"""Audit and report the preregistered G V3 random-vs-F2 comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .evaluation import load_evaluation_log
from .replay import load_game_log
from .training import load_g_checkpoint
from .v3_budget import active_wall_seconds_committed, cpu_seconds_committed
from .v3_config import evaluation_game_counts
from .v3_evaluation import (
    ARMS,
    build_v3_blind_plan,
    build_v3_tactical_puzzles,
    summarize_v3_matches,
    v3_value_and_entropy_diagnostics,
)
from .v3_runner import load_v3_config
from .v3_schedule import validate_v3_schedule


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ci_text(ci: list[float] | None) -> str:
    return "未估计" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _group_matches(matches: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in matches:
        groups[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for key, subset in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        result.append({**dict(zip(keys, key, strict=True)), **summarize_v3_matches(subset)})
    return result


def _row_belongs_to_arm(row: dict[str, Any], arm: str) -> bool:
    """Match a route family to its versioned checkpoint IDs in summary rows."""
    arm_id = str(row.get("arm_id", ""))
    return arm_id == arm or arm_id.startswith(f"{arm}-seed-")


def _final_blind_comparisons(
    blind_summary: list[dict[str, Any]], final_models: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve each selected checkpoint against the actual opponent families in match logs."""
    result: dict[str, dict[str, Any]] = {}
    for model_key, model in sorted(final_models.get("models", {}).items()):
        arm, _seed = str(model_key).split(":", maxsplit=1)
        checkpoint_id = str(model["checkpoint_id"])
        own_family = "G0" if arm == "G0" else "F2"
        other_family = "G-F2" if arm == "G0" else "G0"
        rows = [row for row in blind_summary if row.get("arm_id") == checkpoint_id]
        by_family = {
            family: {
                mode: next((row for row in rows
                            if row.get("opponent_family") == family and row.get("mode") == mode), None)
                for mode in ("network", "puct")
            }
            for family in ("A", "B", "F2", "G0", "G-F2")
        }
        result[str(model_key)] = {
            "checkpoint_id": checkpoint_id,
            "own_initial": by_family[own_family],
            "other_route": by_family[other_family],
            "opponents": by_family,
        }
    return result


def validate_v3_results(root: str | Path) -> dict[str, Any]:
    """Replay and check all formal training/evaluation data against registration."""
    root_path = Path(root).resolve()
    config = load_v3_config(root_path)
    schedule = json.loads((root_path / config["opening_schedule_path"]).read_text(encoding="utf-8"))
    validate_v3_schedule(schedule)
    expected = evaluation_game_counts(
        seed_count=len(config["seeds"]), stage_count=len(config["stage_nodes"]),
        development_groups=int(config["evaluation"]["milestone_opening_groups_per_opponent"]),
        promotion_groups=int(config["promotion"]["opening_groups"]),
        blind_groups=int(config["evaluation"]["blind_opening_groups_per_matchup_seed"]),
    )
    all_selfplay: dict[tuple[str, int], list[dict[str, Any]]] = {}
    selfplay_ids: set[str] = set()
    selfplay_games = selfplay_states = selfplay_simulations = 0
    for seed in map(int, config["seeds"]):
        for arm in ARMS:
            path = root_path / f"research/g_v3/results/selfplay/{arm}-seed-{seed}.jsonl"
            games, ids = load_game_log(path)
            if selfplay_ids.intersection(ids):
                raise ValueError("self-play game IDs collide across route/seed logs")
            selfplay_ids.update(ids)
            accepted = sum(int(game.get("budgeted_environment_steps", -1)) for game in games)
            if accepted != int(config["states_per_seed"]):
                raise ValueError(f"{arm} seed {seed} has {accepted} registered states; "
                                 f"expected {config['states_per_seed']}")
            for game in games:
                if game.get("config_sha256") != config["config_sha256"]:
                    raise ValueError("self-play record differs from the registered config")
                for move in game["moves"]:
                    if not move.get("opening") and int(move.get("simulations", -1)) != 64:
                        raise ValueError("self-play has a move without exactly 64 PUCT simulations")
                selfplay_simulations += int(game["stats"]["simulations"])
            all_selfplay[(arm, seed)] = games
            selfplay_games += len(games)
            selfplay_states += accepted

    eval_root = root_path / "research/g_v3/results/evaluation"
    development, _ = load_evaluation_log(eval_root / "development.jsonl")
    blind, _ = load_evaluation_log(eval_root / "blind.jsonl")
    promotions = [row for row in development if row.get("phase") == "promotion"]
    expected_dev = expected["development"] + expected["promotion"]
    if len(development) != expected_dev:
        raise ValueError(f"development/promotion count {len(development)} differs from {expected_dev}")
    if len(blind) != expected["blind"]:
        raise ValueError(f"blind count {len(blind)} differs from {expected['blind']}")
    blind_plan = build_v3_blind_plan(schedule)
    if {row["game_id"] for row in blind} != {row["game_id"] for row in blind_plan}:
        raise ValueError("blind records do not exactly match the frozen blind plan")
    for row in development + blind:
        if row.get("config_sha256") != config["config_sha256"]:
            raise ValueError("evaluation record differs from the registered config hash")
        stats = row["stats"]
        candidate_moves = [move for move in row["moves"] if move.get("actor") == "agent"]
        if row["mode"] == "puct" and int(stats["agent_simulations"]) != 64 * len(candidate_moves):
            raise ValueError("PUCT evaluation game does not use 64 simulations on each candidate move")
        if row["mode"] == "network" and int(stats["agent_simulations"]) != 0:
            raise ValueError("bare-network evaluation unexpectedly performed MCTS simulations")

    tactics = _read_jsonl(eval_root / "tactics.jsonl")
    expected_tactics = (len(config["stage_nodes"]) + 1) * len(config["seeds"]) * len(ARMS) * 2
    puzzle_count = len(build_v3_tactical_puzzles())
    tactic_ids = [row.get("run_id") for row in tactics]
    if len(tactics) != expected_tactics or len(set(tactic_ids)) != len(tactic_ids):
        raise ValueError(f"tactical run count/IDs invalid: {len(tactics)} expected {expected_tactics}")
    if any(row.get("puzzles") != puzzle_count or len(row.get("records", [])) != puzzle_count
           for row in tactics):
        raise ValueError("a checkpoint is missing one or more tactical puzzle predictions")

    stages = _read_jsonl(root_path / "research/g_v3/results/training/stages.jsonl")
    expected_stages = len(config["stage_nodes"]) * len(config["seeds"]) * len(ARMS)
    if len(stages) != expected_stages:
        raise ValueError(f"checkpoint-stage count {len(stages)} differs from {expected_stages}")
    selection = json.loads((root_path / "research/g_v3/results/selection/state.json").read_text())
    final_manifest = json.loads((root_path / "research/g_v3/results/selection/final_models.json").read_text())
    if final_manifest.get("config_sha256") != config["config_sha256"]:
        raise ValueError("selected-model manifest differs from the registered config")
    final_ids = set()
    for seed in map(int, config["seeds"]):
        for arm in ARMS:
            key = f"{arm}:{seed}"
            chosen = selection["champions"][key]
            final = final_manifest["models"][key]
            if (final["checkpoint_id"] != chosen["checkpoint_id"]
                    or final["sha256"] != chosen["checkpoint_sha256"]):
                raise ValueError("final model does not point at the selected champion")
            path = root_path / final["checkpoint"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != final["sha256"]:
                raise ValueError("selected final checkpoint checksum mismatch")
            load_g_checkpoint(path, device="cpu")
            final_ids.add(final["checkpoint_id"])
    if any(row.get("candidate_checkpoint_id") not in final_ids for row in blind):
        raise ValueError("blind matches do not use the final selected champion models")

    return {
        "valid": True, "config_sha256": config["config_sha256"],
        "registered_evaluation_games": expected,
        "selfplay_games": selfplay_games, "selfplay_unique_game_ids": len(selfplay_ids),
        "budgeted_training_states": selfplay_states, "selfplay_puct_simulations": selfplay_simulations,
        "development_games": len(development), "promotion_games": len(promotions),
        "blind_games": len(blind),
        "evaluation_unique_game_ids": len({row["game_id"] for row in development + blind}),
        "tactical_runs": len(tactics), "tactical_predictions": len(tactics) * puzzle_count,
        "stage_checkpoints": len(stages), "selected_final_models": len(final_manifest["models"]),
    }


def _stage_rows(config: dict[str, Any], games: dict[tuple[str, int], list[dict[str, Any]]],
                stage_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = list(map(int, config["stage_nodes"]))
    rows = []
    for record in sorted(stage_records, key=lambda row: (int(row["seed"]), row["arm"], int(row["node"]))):
        arm, seed, node = str(record["arm"]), int(record["seed"]), int(record["node"])
        stage_index = nodes.index(node)
        stage_games = [game for game in games[(arm, seed)] if int(game["stage_index"]) == stage_index]
        samples = [row for game in stage_games for row in game["experience_rows"]]
        metrics = record.get("training_metrics") or {}
        progress = record.get("progress", {})
        rows.append({
            "arm": arm, "seed": seed, "node_states": node, "stage_index": stage_index,
            "selfplay_games_in_stage": len(stage_games),
            "cumulative_selfplay_games": int(progress.get("game_count", 0)),
            "cumulative_budgeted_states": int(progress.get("budgeted_environment_steps", node)),
            "cumulative_optimizer_updates": int(record.get("optimizer_updates", 0)),
            "stage_selfplay_cpu_seconds": sum(float(game.get("runner_cpu_seconds", 0.0)) for game in stage_games),
            "cumulative_selfplay_cpu_seconds": float(progress.get("selfplay_cpu_seconds", 0.0)),
            "cumulative_training_cpu_seconds": float(progress.get("selfplay_cpu_seconds", 0.0))
            + float(progress.get("optimizer_cpu_seconds", 0.0)),
            "cumulative_optimizer_cpu_seconds": float(progress.get("optimizer_cpu_seconds", 0.0)),
            "mean_policy_entropy": float(np.mean([x["policy_entropy"] for x in samples])) if samples else None,
            "mean_root_value": float(np.mean([x["root_value"] for x in samples])) if samples else None,
            "black_winning_target_fraction": float(np.mean([x["value_target"] == 1.0 for x in samples
                                                              if x["player"] == "B"])) if any(x["player"] == "B" for x in samples) else None,
            "white_winning_target_fraction": float(np.mean([x["value_target"] == 1.0 for x in samples
                                                              if x["player"] == "W"])) if any(x["player"] == "W" for x in samples) else None,
            "policy_loss": metrics.get("policy_loss"), "value_loss": metrics.get("value_loss"),
            "total_loss": metrics.get("total_loss"), "learning_rate": metrics.get("learning_rate"),
            "checkpoint_id": record.get("checkpoint_id"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
        })
    return rows


def generate_v3_report(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = load_v3_config(root_path)
    audit = validate_v3_results(root_path)
    result_dir = root_path / "research/g_v3/results"
    report_dir = root_path / "research/g_v3"
    summary_dir = result_dir / "summary"
    figures_dir = report_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    games: dict[tuple[str, int], list[dict[str, Any]]] = {}
    outcomes: Counter[tuple[str, str]] = Counter()
    entropy: list[float] = []
    for seed in map(int, config["seeds"]):
        for arm in ARMS:
            data, _ = load_game_log(result_dir / "selfplay" / f"{arm}-seed-{seed}.jsonl")
            games[(arm, seed)] = data
            for game in data:
                outcomes[(arm, str(game["outcome"]))] += 1
                entropy.extend(float(row["policy_entropy"]) for row in game["experience_rows"])

    stage_records = _read_jsonl(result_dir / "training/stages.jsonl")
    stages = _stage_rows(config, games, stage_records)
    _write_csv(summary_dir / "training_stages.csv", stages)
    development = _read_jsonl(result_dir / "evaluation/development.jsonl")
    blind = _read_jsonl(result_dir / "evaluation/blind.jsonl")
    matches = development + blind
    tactical = _read_jsonl(result_dir / "evaluation/tactics.jsonl")
    match_summaries = _group_matches(
        matches, ("phase", "arm_id", "seed", "candidate_checkpoint_id", "opponent_family", "mode"),
    )
    _write_csv(summary_dir / "match_summaries.csv", match_summaries)
    match_games = [{
        "phase": row.get("phase"), "game_id": row.get("game_id"), "seed": row.get("seed"),
        "arm_id": row.get("arm_id"), "candidate_checkpoint_id": row.get("candidate_checkpoint_id"),
        "opponent_family": row.get("opponent_family"), "opponent_checkpoint_id": row.get("opponent_checkpoint_id"),
        "mode": row.get("mode"), "opening_group_id": row.get("opening_group_id"),
        "agent_color": row.get("agent_color"), "winner_color": row.get("winner_color"),
        "outcome_for_agent": row.get("outcome_for_agent"), "score_for_agent": row.get("score_for_agent"),
        "move_count": row.get("move_count"), **row.get("stats", {}),
    } for row in matches]
    _write_csv(summary_dir / "match_games.csv", match_games)
    tactical_rows = [{
        key: row.get(key) for key in (
            "arm", "seed", "node", "mode", "puzzles", "correct", "accuracy",
            "cpu_seconds", "wall_seconds", "checkpoint_sha256", "categories",
        )
    } for row in tactical]
    _write_csv(summary_dir / "tactical_accuracy.csv", tactical_rows)
    _json(summary_dir / "value_entropy_diagnostics.json",
          v3_value_and_entropy_diagnostics([row for row in development if row.get("phase") == "development"]))
    selection = json.loads((result_dir / "selection/state.json").read_text(encoding="utf-8"))
    decisions = list(selection.get("decisions", []))
    _write_csv(summary_dir / "promotion_decisions.csv", decisions)
    _json(summary_dir / "integrity_audit.json", audit)
    _json(summary_dir / "selfplay_diagnostics.json", {
        "outcomes": {
            arm: {outcome: count for (row_arm, outcome), count in outcomes.items() if row_arm == arm}
            for arm in ARMS
        },
        "mean_policy_entropy": float(np.mean(entropy)) if entropy else None,
        "training_states": audit["budgeted_training_states"],
        "selfplay_puct_simulations": audit["selfplay_puct_simulations"],
        "mean_terminal_game_plies": float(np.mean([
            len(game["moves"]) for arm_games in games.values() for game in arm_games
        ])) if audit["selfplay_games"] else None,
        "opening_mix": config["training_opening_mix"],
    })

    dev_rows = [row for row in development if row.get("phase") == "development"]
    dev_summary = _group_matches(
        dev_rows, ("arm_id", "seed", "candidate_checkpoint_id", "opponent_family", "mode"),
    )
    # States -> score curves. Both routes use the same PUCT-64 budget and
    # compare to A and their registered initial model (F2 is G-F2's alias).
    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    for arm in ARMS:
        base_opponent = "G0" if arm == "G0" else "F2"
        for opponent, marker in ((base_opponent, "o"), ("A", "s")):
            points = []
            for node in map(int, config["stage_nodes"]):
                rows = [r for r in dev_summary if _row_belongs_to_arm(r, arm)
                        and r["opponent_family"] == opponent and r["mode"] == "puct"
                        and str(r["candidate_checkpoint_id"]).endswith(f"@{node}")]
                if rows:
                    points.append((node, float(np.mean([r["score_rate"] for r in rows]))))
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], f"-{marker}",
                        label=f"{arm} vs {opponent} · PUCT-64")
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.6)
    ax.set(xlabel="累计自我对弈状态数", ylabel="得分率", ylim=(0, 1),
           title="棋力随自我对弈状态增长（开发集）")
    ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures_dir / "g_v3_ability_by_states.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    for arm in ARMS:
        base_opponent = "G0" if arm == "G0" else "F2"
        points = []
        for node in map(int, config["stage_nodes"]):
            sr = [r for r in stages if r["arm"] == arm and int(r["node_states"]) == node]
            mr = [r for r in dev_summary if _row_belongs_to_arm(r, arm)
                  and r["opponent_family"] == base_opponent and r["mode"] == "puct"
                  and str(r["candidate_checkpoint_id"]).endswith(f"@{node}")]
            if sr and mr:
                cpu = float(np.mean([r["cumulative_training_cpu_seconds"] for r in sr]))
                points.append((cpu / 3600.0, float(np.mean([r["score_rate"] for r in mr]))))
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], "-o",
                    label=f"{arm} vs 初始化模型 · PUCT-64")
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.6)
    ax.set(xlabel="累计自我对弈与优化 CPU 小时", ylabel="得分率", ylim=(0, 1),
           title="棋力随训练 CPU 计算时间增长（开发集）")
    ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures_dir / "g_v3_ability_by_cpu.png", dpi=160); plt.close(fig)

    core_opponents = {"A", "B", "F2"}
    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    for arm in ARMS:
        for mode, marker, label in (("network", "--o", "裸网络"), ("puct", "-s", "PUCT-64")):
            points = []
            for node in map(int, config["stage_nodes"]):
                rows = [r for r in dev_summary if _row_belongs_to_arm(r, arm)
                        and r["mode"] == mode and r["opponent_family"] in core_opponents
                        and str(r["candidate_checkpoint_id"]).endswith(f"@{node}")]
                if rows:
                    points.append((node, float(np.mean([r["score_rate"] for r in rows]))))
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], marker,
                        label=f"{arm} {label}")
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.6)
    ax.set(xlabel="累计自我对弈状态数", ylabel="对 A/B/F2 平均得分率", ylim=(0, 1),
           title="固定 checkpoint 下网络与 PUCT-64 的差异")
    ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures_dir / "g_v3_network_vs_puct.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 5.3))
    for arm in ARMS:
        for mode, marker, label in (("network", "--o", "裸网络"), ("puct", "-s", "PUCT-64")):
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in tactical:
                if row["arm"] == arm and row["mode"] == mode:
                    grouped[int(row["node"])].append(float(row["accuracy"]))
            if grouped:
                nodes = sorted(grouped)
                ax.plot(nodes, [float(np.mean(grouped[node])) for node in nodes], marker,
                        label=f"{arm} {label}")
    ax.set(xlabel="累计自我对弈状态数", ylabel="精确战术题正确率", ylim=(0, 1.05),
           title="精确胜点、唯一防守、边线和伪威胁题")
    ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures_dir / "g_v3_tactical_accuracy.png", dpi=160); plt.close(fig)

    blind_summary = _group_matches(blind, ("arm_id", "opponent_family", "mode"))
    _write_csv(summary_dir / "blind_summary.csv", blind_summary)
    final_models = json.loads((result_dir / "selection/final_models.json").read_text(encoding="utf-8"))
    blind_comparisons = _final_blind_comparisons(blind_summary, final_models)

    runtime = {
        "formal_cpu_seconds": cpu_seconds_committed(root_path),
        "formal_instrumented_wall_seconds": active_wall_seconds_committed(root_path),
        "preflight_cpu_seconds": float(json.loads((result_dir / "preflight/preflight.json").read_text())
                                       .get("measurements_cpu_seconds", 0.0)),
        "bn_probe_cpu_seconds": float(json.loads((result_dir / "bn_probe/probe.json").read_text())
                                       .get("cpu_seconds", 0.0)),
    }
    runtime["total_cpu_seconds_including_preflight"] = (
        runtime["formal_cpu_seconds"] + runtime["preflight_cpu_seconds"] + runtime["bn_probe_cpu_seconds"]
    )
    _json(summary_dir / "runtime_costs.json", runtime)

    report = [
        "# WuziQ G V3：从零强化学习与模仿预训练强化学习对照", "",
        f"- 分支：{config['branch']}；基线 G V2 commit：{config['base_commit']}。",
        f"- 注册配置 SHA-256：{config['config_sha256']}；冻结 F2 权重 SHA-256：{config['f2_sha256']}。",
        "- 研究范围：9×9 自由五子棋；两条路线都采用 F2 的 32 通道、2 残差块策略价值 CNN、PUCT-64、自我对弈访问分布与终局胜负监督。",
        "- G0 从两个独立随机种子初始化；G-F2 从同一冻结 F2 权重初始化。两组优化器、经验池和随机状态相互独立。",
        "- 没有 C/E5 教师、F 监督样本、人工战术奖励或推理阶段战术补丁；没有神经网络监督训练教师数据。",
        f"- 训练完成：{audit['budgeted_training_states']:,} 个预算状态，{audit['selfplay_games']} 局完整自我对弈，PUCT 模拟 {audit['selfplay_puct_simulations']:,} 次；每条路线/种子达到 {config['states_per_seed']:,} 状态。",
        f"- 评估验收：开发赛 {audit['development_games']} 局（含 node0），冠军挑战 {audit['promotion_games']} 局，最终盲测 {audit['blind_games']} 局，战术预测 {audit['tactical_predictions']} 次。",
        "",
        "## 环境、公平性和训练方案", "",
        f"正式训练与逐局评估记录 CPU 时间合计 {runtime['formal_cpu_seconds'] / 3600:.2f} 小时；逐任务计时器合计 wall time {runtime['formal_instrumented_wall_seconds'] / 3600:.2f} 小时。预检/BN 探针额外 CPU 时间约 {(runtime['preflight_cpu_seconds'] + runtime['bn_probe_cpu_seconds']) / 3600:.2f} 小时。注册总上限为 {config['hard_cpu_hour_cap']:.1f} CPU 小时。",
        f"预检修正后的估算为乐观 {config['runtime_estimate']['optimistic_cpu_seconds']/3600:.2f}、中心 {config['runtime_estimate']['central_cpu_seconds']/3600:.2f}、保守 {config['runtime_estimate']['conservative_cpu_seconds']/3600:.2f} CPU 小时。估算含 {config['runtime_estimate']['evaluation_games']} 组完整评估对局：开发 {config['runtime_estimate']['evaluation_games_by_phase']['development']}、冠军挑战 {config['runtime_estimate']['evaluation_games_by_phase']['promotion']}、盲测 {config['runtime_estimate']['evaluation_games_by_phase']['blind']}。",
        f"BatchNorm 采用实测短探针中冻结 running statistics 的共同策略。探针每路线 256 状态，不能据此宣称根因已完全确认；采用冻结是预注册 tie-break。完整结果记录于 results/bn_probe/probe.json。",
        f"G-F2 的 node0 权重与冻结 F2 完全相同，所以对该初始模型只进行一次实际对阵，并以 F2 同时标记起点基线；G0 则另外与自己的随机 node0 对照。",
        "开局训练 50% 空盘、50% 合法四手随机开局；开发与盲测使用按 D4 对称去重的未见开局，盲测为 20 组并换色。每步 MCTS 固定 64 次模拟；CPU-only 单线程。", "",
        "## 成长曲线", "",
        "![棋力随状态数变化](figures/g_v3_ability_by_states.png)", "",
        "![棋力随累计训练 CPU 时间变化](figures/g_v3_ability_by_cpu.png)", "",
        "![固定 checkpoint 裸网络与 PUCT-64 对照](figures/g_v3_network_vs_puct.png)", "",
        "![精确战术题表现](figures/g_v3_tactical_accuracy.png)", "",
        "数值明细见 results/summary/training_stages.csv、match_summaries.csv、blind_summary.csv 和 tactical_accuracy.csv。", "",
        "## 最终盲测结果", "",
    ]
    for model_key, comparisons in sorted(blind_comparisons.items()):
        for mode in ("network", "puct"):
            row = comparisons["own_initial"][mode]
            if row is None:
                report.append(f"- {model_key} {mode}：缺少与初始化模型的盲测数据。")
            else:
                report.append(
                    f"- {model_key} ({comparisons['checkpoint_id']}) {mode} 对自身初始化："
                    f"{row['wins']} 胜、{row['draws']} 平、{row['losses']} 负；得分率 {row['score_rate']:.3f}，"
                    f"按20个配对开局的 95% bootstrap 区间 {_ci_text(row['paired_opening_bootstrap_ci95'])}。"
                )
        cross = comparisons["other_route"]["puct"]
        if cross:
            report.append(
                f"- {model_key} PUCT-64 对另一条路线最终冠军：{cross['wins']}-{cross['draws']}-{cross['losses']}，"
                f"得分率 {cross['score_rate']:.3f}，95% CI {_ci_text(cross['paired_opening_bootstrap_ci95'])}。"
            )
        for opponent in ("A", "B", "F2"):
            row = comparisons["opponents"][opponent]["puct"]
            if row:
                report.append(
                    f"- {model_key} PUCT-64 对 {opponent}：{row['wins']}-{row['draws']}-{row['losses']}，"
                    f"得分率 {row['score_rate']:.3f}，95% CI {_ci_text(row['paired_opening_bootstrap_ci95'])}。"
                )
    report += ["", "## 关键问题与证据边界", ""]
    for model_key, comparisons in sorted(blind_comparisons.items()):
        row = comparisons["own_initial"]["puct"]
        if row:
            lower = row["paired_opening_bootstrap_ci95"][0] if row["paired_opening_bootstrap_ci95"] else None
            report.append(
                f"- {model_key} 训练后相对初始化模型的 PUCT-64 得分率为 {row['score_rate']:.3f}，"
                f"95% CI {_ci_text(row['paired_opening_bootstrap_ci95'])}；"
                f"{'区间下限高于 50%' if lower is not None and lower > 0.5 else '没有得到区间下限高于 50% 的证据'}。"
            )
    report.append("- G0 与 G-F2 谁更强只按最终盲测同预算 PUCT-64 和配对区间判断；两种初始化各只有两个训练种子，仍无法充分估计种子方差。")
    report.append("- 裸网络与同 checkpoint 的 PUCT-64 差值是搜索收益；训练前后同为 PUCT-64 的差值是网络学习收益。训练损失下降和搜索次数固定不替代棋力证据。")
    report.append("- 晋级测试只用开发集；盲测仅在所有候选模型和晋级决策冻结后执行一次，没有用盲测挑选 checkpoint。")
    report.append("- 战术题由规则枚举直接胜点和唯一防守，覆盖直接胜点、阻挡、边缘、伪威胁；32 个题目来自 4 个母题的 D4 变换，不能视为 32 个独立题型。")
    report += [
        "", "## 自我对弈、损失与价值诊断", "",
        f"自我对弈经验平均策略熵 {float(np.mean(entropy)):.3f}。训练损失、每阶段胜负目标比例、终局长度见 results/summary/training_stages.csv 和 selfplay_diagnostics.json。",
        "价值校准使用开发对局终局结果，以待走方视角比较每步网络 value/root_value；数据不用于训练，但属于 checkpoint 选择用开发集，不把它当作独立盲测误差。结果见 results/summary/value_entropy_diagnostics.json。",
        "冠军—挑战者在每阶段用 12 个固定开发开局组、双方换色。门槛：得分率至少 62.5%、配对 bootstrap 区间下限大于 50%、黑白得分率各至少 50%。未晋级不淘汰后续候选；最终盲测用当前冠军而非自动采用最后训练权重。",
        "", "## 数据完整性与复现", "",
        f"全部 self-play 棋谱及评估棋谱回放通过。唯一正式评估对局 ID 共 {audit['evaluation_unique_game_ids']} 个；每个自我对弈状态来自完成局，所有实际着法都逐步记录，空位/胜负检查通过。完整核验结果见 results/summary/integrity_audit.json。",
        "原始数据：results/selfplay/*.jsonl、results/evaluation/development.jsonl、blind.jsonl、tactics.jsonl；checkpoint 与优化器状态在 checkpoints/。配置、开局、预检和 BatchNorm 探针在 config/ 与 results/preflight、bn_probe/。",
        "复现命令：python scripts/run_g_v3.py --through-node 30000；生成报告与图表：python scripts/report_g_v3.py。最终模型通过 load_g_checkpoint(path, device='cpu') 加载，文件 checkpoints/{G0,G-F2}/seed-{seed}/final.pt 指向晋级筛选后的冠军权重。",
        "恢复入口：先 fetch 分支到研究工作区，再运行 python scripts/restore_g_v3_snapshot.py --archive <zip> --destination <empty-dir>，校验 manifest SHA-256 后将恢复树合并至对应分支工作树；Drive 文件夹只存 V3 最新有效快照，不更改其他实验备份。",
        "本项目没有神经网络教师或 G V4/15×15实验。本轮属于 G-F2 与 G0 的 AlphaZero 式自我对弈训练与棋力评估。",
        "",
    ]
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return {
        "report": report_path.relative_to(root_path).as_posix(),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "audit": audit,
        "runtime": runtime,
        "blind_summary_rows": len(blind_summary),
        "summary_rows": len(match_summaries),
        "tactical_runs": len(tactical),
    }
