"""Integrity audit, paired summaries, and Chinese research report for G V4."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import load_evaluation_log
from .replay import load_game_log
from .training import load_g_checkpoint
from .v3_budget import cpu_seconds_committed, active_wall_seconds_committed
from .v3_evaluation import build_v3_tactical_puzzles, evaluate_v3_tactics, summarize_v3_matches
from .v4_budget import _load_rows, ensure_cpu_room, record_cpu_event
from .v4_config import V3_CHAMPION_ID, V3_CHAMPION_REL, load_v4_config
from .v4_runner import phase_a_job, phase_b_job, phase_c_job, _job_paths, _sha

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    if complete != raw:
        raise ValueError(f"JSONL has an uncommitted tail: {path}")
    result = []
    for index, line in enumerate(complete.splitlines(), start=1):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL record at {path}:{index}") from exc
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _group(matches: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in matches:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for key, subset in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        output.append({**dict(zip(keys, key, strict=True)), **summarize_v3_matches(subset)})
    return output


def _verify_opening(row: dict[str, Any], openings: dict[str, list[dict[str, Any]]]) -> None:
    opening_id = str(row.get("opening_group_id", ""))
    if opening_id not in openings or row.get("opening") != openings[opening_id]:
        raise ValueError(f"evaluation opening does not match the frozen V4 schedule: {opening_id}")


def validate_v4_results(root: str | Path) -> dict[str, Any]:
    """Replay every complete log and verify quota, unique IDs, and schedule isolation."""
    base = Path(root).resolve()
    config = load_v4_config(base)
    schedule = json.loads((base / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    openings = {str(group["opening_group_id"]): group["opening"]
                for group in schedule["development_openings"] + schedule["blind_openings"]}
    selfplay: list[dict[str, Any]] = []
    selfplay_ids: set[str] = set()
    state_total = simulation_total = 0
    jobs = [phase_a_job(int(seed)) for seed in config["stages"]["A"]["new_random_seeds"]]
    jobs.append(phase_b_job())
    state_path = base / "research/g_v4/results/selection/experiment_state.json"
    experiment_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if experiment_state.get("phase_c_job"):
        jobs.append(experiment_state["phase_c_job"])
    route_totals: dict[str, int] = {}
    for job in jobs:
        path = _job_paths(base, job)["game_log"]
        games, ids = load_game_log(path)
        if selfplay_ids.intersection(ids):
            raise ValueError("G V4 self-play IDs collide between jobs")
        selfplay_ids.update(ids)
        states = sum(int(game.get("budgeted_environment_steps", -1)) for game in games)
        if states < 0 or states > int(job["new_states"]):
            raise ValueError(f"G V4 job {job['job_id']} exceeds its registered state quota")
        if states and states % 16:
            raise ValueError(f"G V4 job {job['job_id']} state count is not optimizer aligned")
        for game in games:
            if game.get("config_sha256") != config["config_sha256"]:
                raise ValueError("G V4 self-play game config hash mismatch")
            if any(not move.get("opening") and int(move.get("simulations", -1)) != 64 for move in game["moves"]):
                raise ValueError("G V4 self-play game deviates from fixed PUCT-64")
            simulation_total += int(game["stats"]["simulations"])
        route_totals[job["job_id"]] = states
        state_total += states
        selfplay.extend(games)
    expected_selfplay_files = {_job_paths(base, job)["game_log"] for job in jobs}
    actual_selfplay_files = {path for path in (base / "research/g_v4/results/selfplay").glob("*.jsonl")}
    if not actual_selfplay_files <= expected_selfplay_files:
        raise ValueError("G V4 contains self-play logs outside its registered route set")
    if state_total > int(config["stages"]["maximum_total_new_states"]):
        raise ValueError("G V4 total new states exceed the 300K cap")
    dev_path = base / "research/g_v4/results/evaluation/development.jsonl"
    blind_path = base / "research/g_v4/results/evaluation/blind.jsonl"
    development, dev_ids = load_evaluation_log(dev_path)
    blind, blind_ids = load_evaluation_log(blind_path)
    if dev_ids.intersection(blind_ids):
        raise ValueError("G V4 development and blind evaluation IDs collide")
    for row in development + blind:
        if row.get("config_sha256") != config["config_sha256"]:
            raise ValueError("G V4 evaluation row config hash mismatch")
        _verify_opening(row, openings)
        if row["mode"] == "puct":
            agent_moves = sum(move.get("actor") == "agent" for move in row["moves"])
            if int(row["stats"]["agent_simulations"]) != 64 * agent_moves:
                raise ValueError("G V4 PUCT evaluation did not use exactly 64 simulations per agent move")
    promotion_path = base / "research/g_v4/results/selection/promotions.jsonl"
    promotions = _read_jsonl(promotion_path)
    if len({str(row.get("candidate_id")) for row in promotions}) != len(promotions):
        raise ValueError("G V4 promotion records have duplicate candidate IDs")
    training_rows = _read_jsonl(base / "research/g_v4/results/training/stages.jsonl")
    if len({str(row.get("checkpoint_id")) for row in training_rows}) != len(training_rows):
        raise ValueError("G V4 training milestone IDs are duplicated")
    for row in training_rows:
        checkpoint = base / str(row["checkpoint"])
        if not checkpoint.is_file() or _sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("G V4 stage checkpoint SHA mismatch")
    completed = bool(experiment_state.get("completed")) and experiment_state.get("phase") == "DONE"
    if completed:
        for job in jobs[:4]:
            expected = int(job["new_states"])
            if route_totals.get(str(job["job_id"]), 0) != expected:
                raise ValueError(f"completed G V4 route {job['job_id']} did not reach its registered state budget")
        if experiment_state.get("phase_c_job"):
            c_job = experiment_state["phase_c_job"]
            c_states = route_totals.get(str(c_job["job_id"]), 0)
            c_selection_path = _job_paths(base, c_job)["selection"]
            c_selection = json.loads(c_selection_path.read_text(encoding="utf-8")) if c_selection_path.exists() else {}
            if c_selection.get("early_stopped"):
                if c_states < 30_000 or c_states % 10_000:
                    raise ValueError("early-stopped G V4 phase C lacks complete 10K stage checkpoints")
            elif c_states != int(c_job["new_states"]):
                raise ValueError("completed G V4 phase C did not reach its registered state budget")
        if actual_selfplay_files != expected_selfplay_files:
            raise ValueError("completed G V4 is missing a registered route self-play log")
    runtime_path = base / "research/g_v4/results/runtime/cpu_events.jsonl"
    runtime = _load_rows(runtime_path) if runtime_path.exists() else []
    event_ids = [str(row["event_id"]) for row in runtime]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("G V4 CPU event IDs are not unique")
    return {
        "config_sha256": config["config_sha256"], "schedule_sha256": config["opening_schedule_sha256"],
        "experiment_completed": completed,
        "selfplay_games": len(selfplay), "selfplay_game_ids": len(selfplay_ids),
        "budgeted_training_states": state_total, "selfplay_puct_simulations": simulation_total,
        "training_states_by_job": route_totals, "development_games": len(development),
        "blind_games": len(blind), "unique_evaluation_game_ids": len(dev_ids) + len(blind_ids),
        "promotion_records": len(promotions), "milestone_checkpoints": len(training_rows),
        "cpu_events": len(runtime), "formal_cpu_seconds": sum(float(row["cpu_seconds"]) for row in runtime),
        "formal_instrumented_wall_seconds": sum(float(row.get("wall_seconds", 0)) for row in runtime),
        "all_selfplay_and_evaluation_games_replayed": True,
        "all_evaluation_openings_match_frozen_schedule": True,
        "all_training_and_evaluation_config_hashes_match": True,
    }


def _checkpoint_tactics(root: Path, checkpoint_id: str, relative: str, mode: str) -> dict[str, Any]:
    path = root / relative
    model = load_g_checkpoint(path, device="cpu")
    result = evaluate_v3_tactics(model, mode=mode, puzzles=build_v3_tactical_puzzles(), simulations=64)
    result.update({"checkpoint_id": checkpoint_id, "checkpoint": relative,
                   "checkpoint_sha256": _sha(path)})
    return result


def generate_v4_report(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    config = load_v4_config(base)
    audit = validate_v4_results(base)
    result_dir = base / "research/g_v4/results"
    summary_dir = result_dir / "summary"
    figures = base / "research/g_v4/figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    development = _read_jsonl(result_dir / "evaluation/development.jsonl")
    blind = _read_jsonl(result_dir / "evaluation/blind.jsonl")
    selfplay: list[dict[str, Any]] = []
    for path in sorted((result_dir / "selfplay").glob("*.jsonl")):
        selfplay.extend(load_game_log(path)[0])
    stages = _read_jsonl(result_dir / "training/stages.jsonl")
    promotions = _read_jsonl(result_dir / "selection/promotions.jsonl")
    dev_summary = _group(development, ("candidate_checkpoint_id", "opponent_checkpoint_id", "opponent_family", "mode")) if development else []
    blind_summary = _group(blind, ("candidate_checkpoint_id", "opponent_checkpoint_id", "opponent_family", "mode")) if blind else []
    _write_csv(summary_dir / "development_summary.csv", dev_summary)
    _write_csv(summary_dir / "blind_summary.csv", blind_summary)
    _write_csv(summary_dir / "training_stages.csv", stages)
    _write_csv(summary_dir / "promotion_decisions.csv", promotions)
    _write_csv(summary_dir / "selfplay_games.csv", [
        {key: game.get(key) for key in ("game_id", "job_id", "phase", "opening_group_id", "training_opening_kind",
                                        "budgeted_environment_steps", "completion_overrun_steps", "outcome",
                                        "move_count", "runner_cpu_seconds", "runner_wall_seconds", "config_sha256")}
        | {"simulations": game.get("stats", {}).get("simulations"),
           "mean_policy_entropy": float(np.mean([float(row.get("policy_entropy", 0)) for row in game.get("experience_rows", [])]))}
        for game in selfplay
    ])
    _write_csv(summary_dir / "evaluation_games.csv", [
        {key: row.get(key) for key in ("game_id", "phase", "candidate_checkpoint_id", "opponent_checkpoint_id",
                                       "opponent_family", "mode", "opening_group_id", "agent_color",
                                       "winner_color", "outcome_for_agent", "score_for_agent", "move_count",
                                       "runner_cpu_seconds", "runner_wall_seconds")}
        | row.get("stats", {}) for row in development + blind
    ])
    _write_json(summary_dir / "integrity_audit.json", audit)
    state_path = result_dir / "selection/experiment_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    runtime_cpu = cpu_seconds_committed(base)
    runtime_wall = active_wall_seconds_committed(base)
    runtime_rows = _load_rows(base / "research/g_v4/results/runtime/cpu_events.jsonl")
    cpu_by_category = {
        category: float(sum(float(row["cpu_seconds"]) for row in runtime_rows if row["category"] == category))
        for category in sorted({str(row["category"]) for row in runtime_rows})
    }
    formal_cpu = float(sum(value for category, value in cpu_by_category.items() if category != "preflight"))
    ledger_preflight_cpu = float(cpu_by_category.get("preflight", 0.0))
    registered_preflight_cpu = float(json.loads((result_dir / "preflight/preflight.json").read_text())
                                     .get("measurement_cpu_seconds", 0.0))
    total_cpu = runtime_cpu + registered_preflight_cpu
    _write_json(summary_dir / "runtime.json", {
        "formal_training_evaluation_and_orchestration_cpu_seconds": formal_cpu,
        "ledger_preflight_diagnostic_cpu_seconds": ledger_preflight_cpu,
        "registered_initial_preflight_cpu_seconds": registered_preflight_cpu,
        "total_cpu_seconds_including_preflight": total_cpu,
        "formal_instrumented_wall_seconds": runtime_wall,
        "cpu_seconds_by_event_category": cpu_by_category,
        "thread_probe": json.loads((result_dir / "runtime/thread_probe.json").read_text())
        if (result_dir / "runtime/thread_probe.json").exists() else None,
        "hard_cpu_hour_cap": config["hard_cpu_hour_cap"],
    })

    # Every registered checkpoint receives the same small exact-rule tactic set.
    tactics_path = result_dir / "evaluation/tactics.jsonl"
    tactics, tactic_ids = [], set()
    if tactics_path.exists():
        tactics = _read_jsonl(tactics_path)
        tactic_ids = {str(row["record_id"]) for row in tactics}
    expected_candidates = {str(row["checkpoint_id"]): str(row["checkpoint"]) for row in stages}
    for job in [phase_a_job(int(seed)) for seed in config["stages"]["A"]["new_random_seeds"]] + [phase_b_job()] + ([state["phase_c_job"]] if state.get("phase_c_job") else []):
        job_state = _job_paths(base, job)["selection"]
        if not job_state.is_file():
            continue
        selection = json.loads(job_state.read_text(encoding="utf-8"))
        start_id, start_path = str(selection["start_checkpoint_id"]), str(selection["start_checkpoint"])
        expected_candidates.setdefault(start_id, start_path)
    for checkpoint_id, relative in expected_candidates.items():
        for mode in ("network", "puct"):
            record_id = f"{checkpoint_id}:{mode}"
            if record_id in tactic_ids:
                continue
            result = _checkpoint_tactics(base, checkpoint_id, relative, mode)
            result["record_id"] = record_id
            result["config_sha256"] = config["config_sha256"]
            result["cpu_seconds"] = sum(float(item.get("cpu_seconds", 0)) for item in result["records"])
            row = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            with tactics_path.open("a", encoding="utf-8") as handle:
                handle.write(row); handle.flush()
                import os
                os.fsync(handle.fileno())
            tactics.append(result); tactic_ids.add(record_id)
    _write_csv(summary_dir / "tactical_accuracy.csv", [
        {key: row.get(key) for key in ("checkpoint_id", "mode", "puzzles", "correct", "accuracy", "cpu_seconds", "categories")}
        for row in tactics
    ])

    # Training-state and CPU curves use measured stage records and paired PUCT results.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for job_id in sorted({str(row["job_id"]) for row in stages}):
        job_rows = sorted([row for row in stages if str(row["job_id"]) == job_id], key=lambda row: int(row["new_states"]))
        points = []
        for stage in job_rows:
            summary = next((row for row in dev_summary if row.get("candidate_checkpoint_id") == stage["checkpoint_id"]
                            and row.get("opponent_checkpoint_id") == V3_CHAMPION_ID and row.get("mode") == "puct"), None)
            if summary:
                points.append((int(stage["new_states"]), float(summary["score_rate"])))
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=job_id)
    ax.axhline(.5, color="black", linewidth=1)
    ax.set(xlabel="本阶段新增自我对弈状态", ylabel="对 G0 起点/阶段对手得分率", ylim=(0, 1), title="G0 阶段冠军挑战成绩")
    ax.grid(alpha=.25); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "g_v4_growth_by_states.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for job_id in sorted({str(row["job_id"]) for row in stages}):
        job_rows = sorted([row for row in stages if str(row["job_id"]) == job_id], key=lambda row: int(row["new_states"]))
        points = []
        for stage in job_rows:
            summary = next((row for row in dev_summary if row.get("candidate_checkpoint_id") == stage["checkpoint_id"]
                            and row.get("opponent_checkpoint_id") == V3_CHAMPION_ID and row.get("mode") == "puct"), None)
            if summary:
                cpu = (float(stage.get("selfplay_cpu_seconds", 0)) + float(stage.get("optimizer_cpu_seconds", 0))) / 3600
                points.append((cpu, float(summary["score_rate"])))
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=job_id)
    ax.axhline(.5, color="black", linewidth=1)
    ax.set(xlabel="路线累计训练 CPU 小时", ylabel="对 G0 起点/阶段对手得分率", ylim=(0, 1), title="棋力相对开发对手随训练 CPU 时间变化")
    ax.grid(alpha=.25); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "g_v4_growth_by_cpu.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for family in ("A", "B", "F2"):
        for mode, style, label in (("network", "--o", "裸网络"), ("puct", "-s", "PUCT-64")):
            points = []
            for stage in sorted(stages, key=lambda row: (str(row["job_id"]), int(row["new_states"]))):
                summary = next((row for row in dev_summary if row.get("candidate_checkpoint_id") == stage["checkpoint_id"]
                                and row.get("opponent_family") == family and row.get("mode") == mode), None)
                if summary:
                    points.append((len(points) + 1, float(summary["score_rate"])))
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], style, label=f"{family} {label}")
    ax.axhline(.5, color="black", linewidth=1)
    ax.set(xlabel="已评测里程碑序号（按训练任务顺序）", ylabel="开发集得分率", ylim=(0, 1), title="训练变化与固定搜索收益（开发赛）")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figures / "g_v4_network_vs_puct.png", dpi=160); plt.close(fig)

    outcome_counts: dict[str, dict[str, int]] = {}
    for job in [phase_a_job(int(seed)) for seed in config["stages"]["A"]["new_random_seeds"]] + [phase_b_job()] + ([state["phase_c_job"]] if state.get("phase_c_job") else []):
        subset = [game for game in selfplay if game.get("job_id") == job["job_id"]]
        outcome_counts[job["job_id"]] = dict(Counter(str(game["outcome"]) for game in subset))
    total_states = audit["budgeted_training_states"]
    phase_c = state.get("phase_c_decision") or {"started": False, "reason": "not reached"}
    final_candidate = state.get("final_candidate", {})
    blind_ci_note = "paired opening bootstrap 95% interval; exploratory sample size is 20 groups per listed matchup"
    report = [
        "# WuziQ G V4：从零强化学习的稳定成长实验", "",
        f"- 分支：`{config['branch']}`；基线 G V3 提交：`{config['base_commit']}`。",
        f"- 注册配置 SHA-256：`{config['config_sha256']}`；V3 20K 权重：`{V3_CHAMPION_REL}`（`{config['v3_champion']['checkpoint_sha256']}`）。",
        "- 规则/算法：9×9 自由五子棋，32 通道、2 残差块策略价值网络，固定 PUCT-64；只用 MCTS 访问分布与终局结果训练。没有教师标签、模仿样本、人工战术奖励或推理规则补丁。",
        f"- 实测新增状态：{total_states:,}；自我对弈 {audit['selfplay_games']} 局、{audit['selfplay_puct_simulations']:,} 次 MCTS 模拟。阶段状态：`{json.dumps(audit['training_states_by_job'], ensure_ascii=False)}`。",
        f"- 正式评估：开发赛 {audit['development_games']} 局；盲测 {audit['blind_games']} 局。盲测使用 {config['evaluation']['blind_opening_groups']} 个开发未见开局、双方换色，按 {blind_ci_note}。",
        "", "## 资源预检、训练预算和恢复依据", "",
        f"- 环境：{config['estimated_resources']['cpu']}；峰值内存预检 {config['estimated_resources']['peak_rss_bytes_measured_preflight']:,} B；预检磁盘可用 {config['estimated_resources']['disk_free_bytes_measured_preflight']:,} B；CPU-only。",
        f"- 预检后登记估算（CPU 时间）：乐观 {config['runtime_estimate']['hours']['optimistic']:.2f} 小时，中位 {config['runtime_estimate']['hours']['central']:.2f} 小时，保守 {config['runtime_estimate']['hours']['conservative']:.2f} 小时；硬上限 {config['hard_cpu_hour_cap']} CPU 小时。",
        f"- 正式训练、评估和协调 CPU {formal_cpu / 3600:.2f} 小时；包括首轮预检与线程诊断的总 CPU {total_cpu / 3600:.2f} 小时。有效工作 wall time {runtime_wall / 3600:.2f} 小时。",
        "- 初始 node-0 评估时发现 PyTorch 默认使用3线程。用相同4局固定种子分别重放3线程和1线程后，A/B/F2/G0 对阵的完整落子序列均与正式记录逐步一致；之后的正式训练/评估固定1线程。逐局耗时仍使用实际值，探针另计入 CPU 账本，详细数据见 `results/runtime/thread_probe.json`。",
        f"- V3 没有 20K 对应的优化器/经验池/随机态快照（当时最新状态为 {config['v3_champion']['latest_training_state_budgeted_states']:,} 状态、{config['v3_champion']['latest_training_state_optimizer_updates']:,} 次更新）。因此 B 阶段采用权重继承、AdamW/余弦调度器重置，并从完整 V3 对局记录重建 20K 经验状态；不称为精确续训。",
        f"- A 三个随机种子：{json.dumps(config['stages']['A']['new_random_seeds'])}；B 权重继承 V3 G0@20K，最多新增 100K；C 启动状态为 `{phase_c.get('started')}`，原因：{phase_c.get('reason', phase_c.get('candidate', {}).get('champion_checkpoint_id', ''))}。总预算上限 300K。",
        "", "## 成长曲线", "",
        "![随状态变化](figures/g_v4_growth_by_states.png)", "",
        "![随 CPU 时间变化](figures/g_v4_growth_by_cpu.png)", "",
        "![裸网络与 PUCT-64](figures/g_v4_network_vs_puct.png)", "",
        "图只绘制有真实配对开发数据的里程碑。每个点来自固定开发开局换色比赛；训练状态本身不会被当作棋力提升。", "",
        "## 随机种子与续训成长", "",
        "`results/summary/development_summary.csv` 给出每个候选 checkpoint 对阶段起点/当前冠军、V3 G0@20K、A、B、F2 的网络及 PUCT-64 对局汇总。置信区间按配对开局 bootstrap 计算。晋级需要得分率≥62.5%、95%区间下限>50%、执黑/执白均≥50%，只由开发赛决定。",
    ]
    for row in promotions:
        report.append(f"- `{row.get('candidate_id')}` 对 `{row.get('incumbent_id')}`：{row.get('wins', '—')}胜/平/负摘要以完整 CSV 为准；得分率 {float(row.get('score_rate', 0)):.3f}，配对95% CI {row.get('paired_opening_bootstrap_ci95')}，晋级={row.get('promote')}。")
    report += ["", "## 最终盲测", "", "盲测未用于选择 checkpoint。开发集选出的冻结模型为：",
               f"- `{final_candidate.get('checkpoint_id')}`，SHA-256 `{final_candidate.get('checkpoint_sha256')}`。"]
    for row in blind_summary:
        report.append(f"- `{row.get('candidate_checkpoint_id')}` 对 `{row.get('opponent_checkpoint_id')}`（{row.get('opponent_family')}），{row.get('mode')}：{row['wins']}胜/{row['draws']}平/{row['losses']}负，得分率 {row['score_rate']:.3f}，95% CI {row['paired_opening_bootstrap_ci95']}；黑 `{row['by_color']['B']['score_rate']}`，白 `{row['by_color']['W']['score_rate']}`。")
    report += ["", "## 战术题、训练统计与解释边界", ""]
    for row in sorted(tactics, key=lambda x: (str(x.get("checkpoint_id")), str(x.get("mode")))):
        report.append(f"- `{row['checkpoint_id']}` {row['mode']} 精确规则战术题 {row['correct']}/{row['puzzles']}（{row['accuracy']:.3f}），类别 `{json.dumps(row['categories'], ensure_ascii=False)}`。题集由少量母题做对称变换，不能当作大量独立战术样本。")
    report += [
        "", f"自我对弈胜负分布按路线见 `{json.dumps(outcome_counts, ensure_ascii=False)}`。策略熵、状态数、迭代更新、模拟数和逐局耗时见 `results/summary/selfplay_games.csv`、`training_stages.csv`。",
        "- 学习收益：相同 PUCT-64 对手/预算下，训练 checkpoint 与同路线起点的差异。搜索收益：同一 checkpoint 下裸网络与 PUCT-64 的差异。二者分列，不能用训练损失下降代替棋力证据。",
        "- G0 是否跨随机种子稳定成长，需要多个随机路线在配对开发赛中重复晋级；A 阶段仅三个新种子且每种子30K。即使都未晋级，也只说明此预算/门槛下没有稳定提升证据。B 是一个旧冠军权重续训路线，单路线不能估计种子方差。",
        "- 盲测每个对局通常只有20组配对开局，CI 可能很宽；对 A/B/F2 的成绩只说明当前协议下的探索性结果，不宣称统计显著。A、B 的实力受其固定规则/50ms配置限制。",
        "- 若 C 未运行，是因为没有 A 阶段新随机种子 champion 通过注册的 V3 G0@20K 挑战门槛；未执行的 110K 预算不计作负例。若任务受 12 CPU 小时硬上限停止，报告中的部分状态会小于规划值。",
        "", "## 完整性、复现与结论", "",
        f"完整性审计通过：{audit['selfplay_games']} 局自我对弈及 {audit['unique_evaluation_game_ids']} 场评估棋谱已逐手回放；唯一评估 ID、开局计划和配置 hash 校验均通过。结果见 `results/summary/integrity_audit.json`。",
        "保存内容：`results/selfplay/` 原始棋谱、`results/evaluation/` 逐局对局、`checkpoints/` 模型和可精确恢复训练状态、`config/` 预注册参数及开局池、`figures/` 趋势图。",
        "复现：`python scripts/run_g_v4.py --initialize`，随后重复 `python scripts/run_g_v4.py --advance` 直到输出 `completed=true`；生成报告：`python scripts/report_g_v4.py`。空目录快照恢复：`python scripts/restore_g_v4_snapshot.py --archive <zip> --destination <empty-dir>`。",
        "结果判断仅由实测配对棋局、路线晋级记录和盲测组成。V4没有神经网络外部教师，也没有执行 G-F2 新训练、15×15或 G V5。",
    ]
    report_path = base / "research/g_v4/REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"report": report_path.relative_to(base).as_posix(),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "audit": audit, "development_summary_rows": len(dev_summary),
            "blind_summary_rows": len(blind_summary), "tactical_runs": len(tactics),
            "formal_cpu_seconds": formal_cpu, "total_cpu_seconds_including_preflight": total_cpu}
