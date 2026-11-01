"""Validate raw G V2 artifacts and render the Chinese research report."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ...gomoku import GomokuBoard
from .evaluation import load_evaluation_log, summarize_matches
from .replay import load_game_log
from .training import F2_SHA256, load_g_checkpoint
from .v2_blind import build_v2_blind_plan, summarize_v2_results
from .v2_schedule import validate_v2_schedule

LR_LABELS = {"lr_1e4": "1e-4", "lr_3e5": "3e-5", "lr_1e3": "1e-3"}
LR_COLORS = {"lr_1e4": "#315B7D", "lr_3e5": "#D28A3B", "lr_1e3": "#7868A6"}
MODE_COLORS = {"network": "#315B7D", "puct": "#D28A3B"}
LR_ORDER = ("lr_3e5", "lr_1e4", "lr_1e3")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _exact_tactics(records: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    counts = {
        mode: {"direct_win_opportunities": 0, "direct_wins_taken": 0,
               "single_block_opportunities": 0, "single_blocks_taken": 0}
        for mode in ("network", "puct")
    }

    def wins(board: GomokuBoard, player: str) -> list[tuple[int, int]]:
        result = []
        for row in range(9):
            for col in range(9):
                if board.get((row, col)) is not None:
                    continue
                board.place_stone(player, (row, col))
                winning = board.check_victory((row, col), player)
                board.undo()
                if winning:
                    result.append((row, col))
        return result

    for record in records:
        mode = str(record.get("mode"))
        if mode not in counts or record.get("candidate_id") not in LR_LABELS:
            continue
        board = GomokuBoard(9)
        for move in record["moves"]:
            player = move["player"]
            coord = (int(move["row"]), int(move["col"]))
            if move.get("actor") == "agent":
                own = wins(board, player)
                enemy = "W" if player == "B" else "B"
                threat = wins(board, enemy)
                row = counts[mode]
                if own:
                    row["direct_win_opportunities"] += 1
                    if coord in own:
                        row["direct_wins_taken"] += 1
                elif len(threat) == 1:
                    row["single_block_opportunities"] += 1
                    if coord == threat[0]:
                        row["single_blocks_taken"] += 1
            board.place_stone(player, coord)
        # Match records are validated by the loader before this diagnostic runs.
    for values in counts.values():
        values["direct_win_accuracy"] = (
            values["direct_wins_taken"] / values["direct_win_opportunities"]
            if values["direct_win_opportunities"] else None
        )
        values["single_block_accuracy"] = (
            values["single_blocks_taken"] / values["single_block_opportunities"]
            if values["single_block_opportunities"] else None
        )
    return counts


def _value_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    squared: dict[str, list[float]] = {"network": [], "puct": []}
    absolute: dict[str, list[float]] = {"network": [], "puct": []}
    for record in records:
        mode = str(record.get("mode"))
        if mode not in squared:
            continue
        for move in record["moves"]:
            if move.get("actor") != "agent":
                continue
            field = "value" if mode == "network" else "root_value"
            if field not in move:
                continue
            agent_score = float(record["score_for_agent"])
            side_score = agent_score if move["player"] == record["agent_color"] else 1.0 - agent_score
            target = 2.0 * side_score - 1.0
            error = float(move[field]) - target
            squared[mode].append(error * error)
            absolute[mode].append(abs(error))
    return {
        mode: {"moves": len(squared[mode]),
               "outcome_value_mse": float(np.mean(squared[mode])) if squared[mode] else None,
               "outcome_value_mae": float(np.mean(absolute[mode])) if absolute[mode] else None}
        for mode in squared
    }


def _dev_summary(records: list[dict[str, Any]], checkpoint_id: str, mode: str, opponent: str) -> dict[str, Any] | None:
    selected = [row for row in records if row.get("candidate_checkpoint_id") == checkpoint_id
                and row.get("mode") == mode and row.get("opponent") == opponent]
    if not selected:
        return None
    if any(row.get("evaluation_phase") != "development" for row in selected):
        raise ValueError("development curve includes a non-development game")
    return summarize_matches(selected)


def _training_milestones(root: Path, config: dict[str, Any], dev_records: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = list(map(int, config["stage_nodes"]))
    candidates: dict[str, Any] = {}
    for lr_id in LR_ORDER:
        log_path = root / "research/g_v2/results/selfplay" / f"{lr_id}.jsonl"
        games, ids = load_game_log(log_path) if log_path.exists() else ([], set())
        if len(games) != len(ids):
            raise ValueError(f"{lr_id} has duplicate self-play game IDs")
        progress_path = root / "research/g_v2/checkpoints" / lr_id / "progress.json"
        progress = _read_json(progress_path) if progress_path.exists() else {
            "budgeted_environment_steps": 0, "optimizer_updates": 0,
            "active_cpu_seconds": 0.0, "training_history": [], "game_count": 0,
        }
        if int(progress.get("game_count", 0)) != len(games):
            raise ValueError(f"{lr_id} progress count differs from its durable self-play games")
        cumulative_budgeted = int(progress.get("budgeted_environment_steps", 0))
        cumulative_updates = int(progress.get("optimizer_updates", 0))
        if cumulative_budgeted > int(config["states_per_candidate"]):
            raise ValueError(f"{lr_id} exceeds the registered self-play state quota")
        if cumulative_budgeted == int(config["states_per_candidate"]):
            if cumulative_updates != int(config["updates_per_candidate"]):
                raise ValueError(f"{lr_id} update count differs from its equal registered quota")
        history = list(progress.get("training_history", []))
        milestones = []
        consumed = 0
        actual_cpu = 0.0
        actual_games = 0
        experience: list[dict[str, Any]] = []
        accepted_games: list[dict[str, Any]] = []
        for node in nodes:
            while consumed < node and actual_games < len(games):
                game = games[actual_games]
                used = int(game.get("budgeted_environment_steps", 0))
                take = min(used, node - consumed)
                if take <= 0:
                    break
                consumed += take
                actual_cpu += float(game.get("runner_cpu_seconds", game.get("stats", {}).get("cpu_seconds", 0.0)))
                experience.extend(game["experience_rows"][:take])
                accepted_games.append(game)
                actual_games += 1
            node_history = [row for row in history if int(row.get("states_seen", 0)) <= node]
            updates = sum(int(row.get("updates", 0)) for row in node_history)
            actual_cpu += sum(float(row.get("cpu_seconds", 0.0)) for row in node_history
                              if not row.get("_cpu_counted"))
            # Histories are cumulative; add each update exactly once across milestones.
            if node_history:
                actual_cpu = sum(
                    float(game.get("runner_cpu_seconds", game.get("stats", {}).get("cpu_seconds", 0.0)))
                    for game in accepted_games
                ) + sum(float(row.get("cpu_seconds", 0.0)) for row in node_history)
            checkpoint = root / "research/g_v2/checkpoints" / lr_id / f"node_{node}.pt"
            checkpoint_present = checkpoint.is_file()
            if checkpoint_present:
                load_g_checkpoint(checkpoint)
            checkpoint_id = f"{lr_id}@{node}"
            node_records = [row for row in dev_records if row.get("candidate_checkpoint_id") == checkpoint_id]
            value = _value_metrics(node_records)
            entropy = [float(row.get("policy_entropy", row.get("search_entropy", 0.0)))
                       for match in node_records for row in match["moves"] if row.get("actor") == "agent"
                       and ("policy_entropy" in row or "search_entropy" in row)]
            if not entropy:
                entropy = [float(row.get("policy_entropy", 0.0)) for row in experience]
            loss = node_history[-1] if node_history else {}
            puct_f2 = _dev_summary(dev_records, checkpoint_id, "puct", "F2")
            network_f2 = _dev_summary(dev_records, checkpoint_id, "network", "F2")
            by_color = {color: None for color in ("B", "W")}
            if puct_f2:
                by_color = {color: puct_f2["by_color"][color] for color in ("B", "W")}
            row = {
                "node": node, "executed": checkpoint_present and consumed >= node,
                "checkpoint": str(checkpoint) if checkpoint_present else None,
                "games": actual_games if consumed >= node else None,
                "states": consumed if consumed >= node else None,
                "optimizer_updates": updates if consumed >= node else None,
                "training_cpu_seconds": actual_cpu if consumed >= node else None,
                "puct_vs_f2": puct_f2, "network_vs_f2": network_f2,
                "puct_vs_a": _dev_summary(dev_records, checkpoint_id, "puct", "A"),
                "puct_vs_b": _dev_summary(dev_records, checkpoint_id, "puct", "B"),
                "by_color": by_color,
                "policy_entropy": float(np.mean(entropy)) if entropy else None,
                "value_metrics": value,
                "policy_loss": float(loss["policy_loss"]) if "policy_loss" in loss else None,
                "value_loss": float(loss["value_loss"]) if "value_loss" in loss else None,
                "total_loss": float(loss["total_loss"]) if "total_loss" in loss else None,
                "selfplay_simulations": sum(int(sample.get("simulations", 0)) for sample in experience),
                "outcomes": {
                    "B_win": sum(game["outcome"] == "B_win" for game in accepted_games),
                    "W_win": sum(game["outcome"] == "W_win" for game in accepted_games),
                    "draw": sum(game["outcome"] == "draw" for game in accepted_games),
                },
                "mean_complete_game_plies": (
                    float(np.mean([len(game["moves"]) - len(game["opening"]) for game in accepted_games]))
                    if accepted_games else None
                ),
            }
            milestones.append(row)
        final_node = max(nodes)
        candidates[lr_id] = {
            "candidate_id": lr_id,
            "learning_rate": float(config["candidate_learning_rates"][lr_id]),
            "games": len(games), "unique_game_ids": len(ids),
            "states": cumulative_budgeted, "actual_environment_steps": sum(len(game["experience_rows"]) for game in games),
            "optimizer_updates": cumulative_updates,
            "training_cpu_seconds": float(progress.get("active_cpu_seconds", 0.0)),
            "training_wall_seconds": float(progress.get("active_wall_seconds", 0.0)),
            "completed": cumulative_budgeted == int(config["states_per_candidate"]),
            "milestones": milestones,
        }
    return {"candidates": candidates,
            "unexecuted_nodes": [node for node in nodes if not any(
                candidate["milestones"][nodes.index(node)]["executed"] for candidate in candidates.values()
            )]}


def _tactical_and_values(dev_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"exact_tactics": _exact_tactics(dev_records), "outcome_value_metrics": _value_metrics(dev_records)}


def build_v2_report(root: str | Path) -> dict[str, Any]:
    """Validate registered training/evaluation artifacts and build reproducible report data."""
    root_path = Path(root).resolve()
    base = root_path / "research/g_v2"
    config = _read_json(base / "config/experiment.json")
    schedule = _read_json(base / "config/openings.json")
    if config.get("registered") is not True or config.get("base_commit") != "6e2bf48aa6b630c0fa4f20284ceb3b5aab295080":
        raise ValueError("G V2 report refuses an unregistered or changed baseline config")
    validate_v2_schedule(schedule)
    if schedule["schedule_sha256"] != config["opening_schedule_sha256"]:
        raise ValueError("G V2 report schedule differs from the registered config")
    f2_path = root_path / "research/f_v2/checkpoints/best_f2.pt"
    if hashlib.sha256(f2_path.read_bytes()).hexdigest() != F2_SHA256:
        raise ValueError("report input F2 checkpoint differs from the frozen V2 baseline")
    preflight = _read_json(base / "results/preflight/preflight.json")
    dev_records, _ = load_evaluation_log(base / "results/evaluation/development.jsonl")
    challenge_records, _ = load_evaluation_log(base / "results/evaluation/champion.jsonl")
    blind_records, _ = load_evaluation_log(base / "results/evaluation/blind.jsonl")
    plan = build_v2_blind_plan(schedule)
    if {row["game_id"] for row in blind_records} != {row["game_id"] for row in plan}:
        raise ValueError("blind log does not contain exactly the registered complete comparison plan")
    if any(row.get("evaluation_phase") != "development" for row in dev_records):
        raise ValueError("development log contains records from another evaluation phase")
    if any(row.get("evaluation_phase") != "blind" for row in blind_records):
        raise ValueError("blind log contains records from another evaluation phase")
    state_path = base / "results/selection/state.json"
    stage_path = base / "results/selection/stages.jsonl"
    state = _read_json(state_path)
    stage_lines = [json.loads(line) for line in stage_path.read_text().splitlines() if line.strip()]
    if len({int(row["node"]) for row in stage_lines}) != len(stage_lines):
        raise ValueError("selection history contains duplicate milestone nodes")
    if int(state.get("last_completed_node", -1)) not in [0, *map(int, config["stage_nodes"])]:
        raise ValueError("selection state is not at a registered development milestone")
    blind_summary = summarize_v2_results(blind_records)
    stored_blind = _read_json(base / "results/evaluation/blind_summary.json")
    if stored_blind.get("comparisons") != blind_summary["comparisons"]:
        raise ValueError("stored blind summary differs from the raw match records")
    training = _training_milestones(root_path, config, dev_records)
    tactics_values = _tactical_and_values(dev_records)
    total_cpu_seconds = sum(float(row.get("stats", {}).get("total_cpu_seconds", 0.0))
                            for row in [*dev_records, *challenge_records, *blind_records])
    total_cpu_seconds += sum(float(item["training_cpu_seconds"]) for item in training["candidates"].values())
    candidate_records = sum(len(load_game_log(base / "results/selfplay" / f"{lr}.jsonl")[0])
                            for lr in LR_ORDER if (base / "results/selfplay" / f"{lr}.jsonl").exists())
    if len(challenge_records) != len(stage_lines) * 3 * 16:
        raise ValueError("champion challenge logs do not match all candidates and milestone nodes")
    summary = {
        "metadata": {
            "experiment_id": config["experiment_id"], "base_commit": config["base_commit"],
            "registered_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "schedule_sha256": schedule["schedule_sha256"],
            "f2_sha256": F2_SHA256,
            "preflight_wall_seconds": float(preflight["measurements_wall_seconds"]),
            "preflight_cpu_seconds": float(preflight["measurements_cpu_seconds"]),
            "resources": preflight["resources"],
            "optimistic_cpu_seconds": float(config["estimates"]["optimistic_cpu_seconds"]),
            "central_cpu_seconds": float(config["estimates"]["central_cpu_seconds"]),
            "conservative_cpu_seconds": float(config["estimates"]["conservative_cpu_seconds"]),
            "hard_cpu_hour_cap": float(config["hard_cpu_hour_cap"]),
            "unexecuted_nodes": [node for node in (20000,) if node not in config["stage_nodes"]],
            "formal_selfplay_game_records": candidate_records,
            "development_games": len(dev_records), "challenge_games": len(challenge_records),
            "blind_games": len(blind_records), "total_actual_cpu_seconds": total_cpu_seconds,
        },
        "training": training,
        "selection": {
            "champion": state["champion"],
            "best_candidate_for_blind": state.get("best_candidate_for_blind"),
            "last_completed_node": state["last_completed_node"],
            "stages": stage_lines,
            "exact_tactics": tactics_values["exact_tactics"],
            "outcome_value_metrics": tactics_values["outcome_value_metrics"],
        },
        "blind": blind_summary,
        "integrity": {
            "all_games_valid": True,
            "duplicate_game_ids": 0,
            "selfplay_games": candidate_records,
            "evaluation_games": len(dev_records) + len(challenge_records) + len(blind_records),
        },
    }
    return summary


def _plot_growth_by_states(root: Path, training: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for lr_id in LR_ORDER:
        points = [row for row in training["candidates"][lr_id]["milestones"] if row["puct_vs_f2"]]
        if not points:
            continue
        x = [row["node"] for row in points]
        y = [row["puct_vs_f2"]["score_rate"] for row in points]
        cis = [row["puct_vs_f2"]["paired_opening_bootstrap_ci95"] for row in points]
        low = [max(0.0, rate - interval[0]) for rate, interval in zip(y, cis, strict=True)]
        high = [max(0.0, interval[1] - rate) for rate, interval in zip(y, cis, strict=True)]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, linewidth=2,
                    color=LR_COLORS[lr_id], label=f"学习率 {LR_LABELS[lr_id]}")
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="与冻结F2得分相同")
    ax.set_title("PUCT-64棋力随自我对弈训练状态变化")
    ax.set_xlabel("累计自我对弈训练状态（每个学习率候选）")
    ax.set_ylabel("对冻结F2的得分率")
    ax.set_ylim(0, 1)
    ax.set_xticks([1000, 2000, 5000, 10000])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = root / "research/g_v2/figures/growth_by_states.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.relative_to(root).as_posix()


def _plot_growth_by_cpu(root: Path, training: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for lr_id in LR_ORDER:
        points = [row for row in training["candidates"][lr_id]["milestones"]
                  if row["puct_vs_f2"] and row["training_cpu_seconds"] is not None]
        if not points:
            continue
        x = [row["training_cpu_seconds"] for row in points]
        y = [row["puct_vs_f2"]["score_rate"] for row in points]
        cis = [row["puct_vs_f2"]["paired_opening_bootstrap_ci95"] for row in points]
        low = [max(0.0, rate - interval[0]) for rate, interval in zip(y, cis, strict=True)]
        high = [max(0.0, interval[1] - rate) for rate, interval in zip(y, cis, strict=True)]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, linewidth=2,
                    color=LR_COLORS[lr_id], label=f"学习率 {LR_LABELS[lr_id]}")
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="与冻结F2得分相同")
    ax.set_title("PUCT-64棋力随候选训练CPU时间变化")
    ax.set_xlabel("累计训练CPU秒（不含评估对局）")
    ax.set_ylabel("对冻结F2的得分率")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = root / "research/g_v2/figures/growth_by_cpu.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.relative_to(root).as_posix()


def _plot_network_vs_puct(root: Path, training: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    for lr_id in LR_ORDER:
        points = [row for row in training["candidates"][lr_id]["milestones"]
                  if row["puct_vs_f2"] and row["network_vs_f2"]]
        for mode, field in (("network", "network_vs_f2"), ("puct", "puct_vs_f2")):
            selected = [row for row in points if row[field]]
            if not selected:
                continue
            ax.plot(
                [row["node"] for row in selected],
                [row[field]["score_rate"] for row in selected],
                color=LR_COLORS[lr_id], linestyle="--" if mode == "network" else "-",
                marker="o", linewidth=1.7,
                label=f"{LR_LABELS[lr_id]} · {'裸网络' if mode == 'network' else 'PUCT-64'}",
            )
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1, label="与冻结F2得分相同")
    ax.set_title("相同训练阶段的裸网络与PUCT-64对照")
    ax.set_xlabel("累计自我对弈训练状态（每个学习率候选）")
    ax.set_ylabel("对冻结F2的得分率")
    ax.set_ylim(0, 1)
    ax.set_xticks([1000, 2000, 5000, 10000])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    path = root / "research/g_v2/figures/network_vs_puct.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.relative_to(root).as_posix()


def render_v2_report(root: str | Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Write three Chinese trend charts, source tables, summary data and REPORT.md."""
    root_path = Path(root).resolve()
    figures = [
        _plot_growth_by_states(root_path, summary["training"]),
        _plot_growth_by_cpu(root_path, summary["training"]),
        _plot_network_vs_puct(root_path, summary["training"]),
    ]
    training_rows = []
    for lr_id in LR_ORDER:
        candidate = summary["training"]["candidates"][lr_id]
        for stage in candidate["milestones"]:
            training_rows.append({
                "candidate": lr_id, "learning_rate": candidate["learning_rate"],
                "node": stage["node"], "executed": stage["executed"],
                "games": stage["games"], "states": stage["states"],
                "updates": stage["optimizer_updates"], "training_cpu_seconds": stage["training_cpu_seconds"],
                "network_vs_f2": stage["network_vs_f2"]["score_rate"] if stage["network_vs_f2"] else None,
                "puct_vs_f2": stage["puct_vs_f2"]["score_rate"] if stage["puct_vs_f2"] else None,
                "policy_entropy": stage["policy_entropy"], "policy_loss": stage["policy_loss"],
                "value_loss": stage["value_loss"], "value_mse": stage["value_metrics"]["network"]["outcome_value_mse"],
                "selfplay_simulations": stage["selfplay_simulations"],
            })
    growth_csv = root_path / "research/g_v2/results/growth.csv"
    _write_csv(growth_csv, training_rows, [
        "candidate", "learning_rate", "node", "executed", "games", "states", "updates",
        "training_cpu_seconds", "network_vs_f2", "puct_vs_f2", "policy_entropy", "policy_loss",
        "value_loss", "value_mse", "selfplay_simulations",
    ])
    blind_rows = []
    for name, item in sorted(summary["blind"]["comparisons"].items()):
        blind_rows.append({
            "comparison": name, "games": item["games"], "wins": item["wins"],
            "draws": item["draws"], "losses": item["losses"], "points": item["points"],
            "score_rate": item["score_rate"],
            "ci95_low": item["paired_opening_bootstrap_ci95"][0] if item["paired_opening_bootstrap_ci95"] else None,
            "ci95_high": item["paired_opening_bootstrap_ci95"][1] if item["paired_opening_bootstrap_ci95"] else None,
            "black_score_rate": item["by_color"]["B"]["score_rate"],
            "white_score_rate": item["by_color"]["W"]["score_rate"],
            "mean_agent_seconds": item["mean_agent_seconds_per_game"],
            "total_cpu_seconds": item["total_cpu_seconds"],
            "agent_simulations": item["total_agent_simulations"],
        })
    blind_csv = root_path / "research/g_v2/results/blind_comparisons.csv"
    _write_csv(blind_csv, blind_rows, list(blind_rows[0]) if blind_rows else ["comparison"])
    _write_json(root_path / "research/g_v2/results/final_summary.json", summary)

    meta = summary["metadata"]
    candidates = summary["training"]["candidates"]
    total_states = sum(item["states"] for item in candidates.values())
    total_updates = sum(item["optimizer_updates"] for item in candidates.values())
    blind_games = meta["blind_games"]
    primary = summary["blind"]["learning"].get("F2-PUCT64_to_G-F2-PUCT64")
    if primary:
        lo, hi = primary["g_view"]["paired_opening_bootstrap_ci95"] or (None, None)
        ci = "不可估" if lo is None else f"[{lo:.3f}, {hi:.3f}]"
        growth_conclusion = (
            f"G-F2 PUCT-64 对冻结 F2 的成对得分率为 **{primary['g_score_rate']:.3f}**，"
            f"差值 {primary['g_minus_f2_score_rate']:+.3f}，95% 配对开局区间 {ci}。"
        )
    else:
        growth_conclusion = "最终盲测未提供冻结 F2 与 G-F2 PUCT-64 的同预算配对结果。"
    champion = summary["selection"].get("champion", {})
    best = summary["selection"].get("best_candidate_for_blind", {})
    lines = [
        "# G-F2自我对弈在本轮预算下是否产生可复核成长",
        "",
        "## 结论摘要",
        "",
        f"- 本轮使用同一个冻结 F2 网络作为三条学习率候选的初始化；固定每步 PUCT-64，不引入教师标签、F 监督样本或人工战术奖励。正式训练完成 {total_states:,} 个预算内状态和 {total_updates:,} 次更新，实际盲测 {blind_games} 局。",
        f"- {growth_conclusion}",
        f"- 开发赛冠军仍为 `{champion.get('checkpoint_id', '未知')}`；供盲测的最佳开发候选为 `{best.get('checkpoint_id', '未知')}`。只有满足预注册冠军门槛的候选才替换 F2。",
        f"- 实际总 CPU 时间约 {meta['total_actual_cpu_seconds'] / 3600:.2f} 小时；2 CPU 小时上限未扩大。20,000 状态节点为**未执行**，本轮每个学习率候选的登记上限为 10,000 状态。",
        "",
        "## 实验设计与计算预算",
        "",
        f"- 基线提交：`{meta['base_commit']}`；冻结 F2 SHA-256：`{meta['f2_sha256']}`；开局表 SHA-256：`{meta['schedule_sha256']}`。",
        f"- 预检实测 {meta['preflight_wall_seconds']:.2f} 秒墙钟 / {meta['preflight_cpu_seconds']:.2f} CPU 秒；正式估算乐观、中位、保守分别为 {meta['optimistic_cpu_seconds']:.0f}/{meta['central_cpu_seconds']:.0f}/{meta['conservative_cpu_seconds']:.0f} CPU 秒。资源：CPU `{meta['resources'].get('cpu_count', '未知')}`，设备 `{meta['resources'].get('device', '未知')}`。",
        "- 三个独立候选从同一 F2 权重开始，学习率分别为 3e-5、1e-4、1e-3；每候选相同状态配额、经验池规则、批次和更新频率。策略目标是自我对弈访问分布，价值目标是终局胜负。",
        "- 训练里程碑为 1k、2k、5k、10k；20,000 状态节点**未执行**。三候选总计并行比较，不把增加搜索次数当作学习收益。",
        "",
        "## 训练规模、过程损失与自我对弈分布",
        "",
        "| 候选学习率 | 完整自我对弈局 | 状态数 | 更新数 | 训练CPU秒 | 执行状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for lr_id in LR_ORDER:
        item = candidates[lr_id]
        lines.append(f"| {LR_LABELS[lr_id]} | {item['games']} | {item['states']:,} | {item['optimizer_updates']} | {item['training_cpu_seconds']:.1f} | {'完成' if item['completed'] else '未完成'} |")
    lines.extend([
        "",
        "各里程碑的训练损失、策略熵、价值误差、完整棋局数与模拟数保存在 [`growth.csv`](results/growth.csv) 和 [`final_summary.json`](results/final_summary.json)。自我对弈胜负颜色、局长、每步访问分布可从原始棋谱复算。",
        "",
        "## 棋力随训练状态与CPU时间变化",
        "",
        "下图中的每个点是该检查点在固定开发开局上的 PUCT-64 对冻结 F2 得分率；误差线为按开局组重采样的 95% 百分位区间。CPU 曲线仅计该候选的训练 CPU，不含模型评估开销。",
        "",
        f"![PUCT-64 对冻结 F2 的得分率随训练状态变化]({figures[0].replace('research/g_v2/', '')})",
        "",
        f"![PUCT-64 对冻结 F2 的得分率随训练 CPU 时间变化]({figures[1].replace('research/g_v2/', '')})",
        "",
        f"![相同阶段裸网络与 PUCT-64 对冻结 F2 的得分率]({figures[2].replace('research/g_v2/', '')})",
        "",
        "## 开发赛与冠军晋级",
        "",
        "每个阶段的三个候选均挑战同一个冻结 incumbent，换色、成对开局。晋级门槛是得分率至少 0.625、配对开局 bootstrap 下界高于 0.5、黑白得分均不低于 0.5，并且对 A/B 的 PUCT 得分相对 F2 未下降超过 0.125。最终盲测不参与选择。",
        f"当前 incumbent：`{champion.get('checkpoint_id', '未知')}`。盲测候选：`{best.get('checkpoint_id', '未知')}`（选择节点 {best.get('selection_node', '未知')}）。如果没有候选通过晋级标准，冠军保留 F2，但仍盲测开发集上表现最好的候选。",
        "",
        "## 独立盲测：同预算学习收益与固定检查点搜索收益",
        "",
        "盲测使用 20 个未参与训练或模型选择的开局，每个比较双方各执黑/白一局。F2 与 G 棋手分别使用裸网络和 PUCT-64；各条件再对 A、B（B 固定 50ms）进行对弈。下表点数按胜 1、平 0.5、负 0 计算；置信区间按开局配对 bootstrap。",
        "",
        "| 比较（左侧为统计视角） | W-D-L | 得分率（95%区间） | 黑方得分 | 白方得分 | 平均决策秒/局 | CPU秒 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, item in sorted(summary["blind"]["comparisons"].items()):
        ci = item["paired_opening_bootstrap_ci95"]
        ci_text = "不可估" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            f"| {name} | {item['wins']}-{item['draws']}-{item['losses']} | {item['score_rate']:.3f} {ci_text} | "
            f"{item['by_color']['B']['score_rate']:.3f} | {item['by_color']['W']['score_rate']:.3f} | "
            f"{item['mean_agent_seconds_per_game']:.3f} | {item['total_cpu_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "- 学习收益：比较 `F2-PUCT64` 与 `G-F2-PUCT64`，即同等搜索预算下比较训练前后网络。",
        "- 搜索收益：比较同一 checkpoint 的 network 与 PUCT-64，分别计算 F2 和 G。",
        "- 裸网络或 PUCT 对 A/B 的胜负和颜色分项见 [`blind_comparisons.csv`](results/blind_comparisons.csv)。两种收益不会相互混算。",
        "",
        "## 攻防、价值估计与局限",
        "",
        "攻防诊断仅统计精确规则可证的即时五连机会与唯一必防点，不修改网络落子。策略准确性由开发对局之外的战术结果共同衡量；网络推理仍只采用合法落点掩码。价值 MSE 按每个决策方的最终胜负映射到 [-1,1] 计算。",
        f"精确战术计数：`{json.dumps(summary['selection']['exact_tactics'], ensure_ascii=False, sort_keys=True)}`。价值误差：`{json.dumps(summary['selection']['outcome_value_metrics'], ensure_ascii=False, sort_keys=True)}`。",
        "- 本轮只训练 F2 初始化候选，未训练 G0 随机初始化；因此不能回答从随机权重出发是否成长，也不能量化 F2 初始化相对随机初始化的加速。",
        "- 两小时预算下 20k 训练节点未运行；候选只到 10k。开发集仅8组开局，用于晋级，区间不适合作稳定棋力排序。最终盲测每比较20组开局、40局，仍属有限样本。",
        "- 训练中自我对弈对手是同一当前网络，可能出现共同弱点；候选机制和冻结 A/B/F2 评估能暴露部分退化，但不等价于更广泛对手群。",
        "- 若盲测置信区间覆盖零变化，不能把点估计差异宣称为确证的学习提升。一个候选晋级也只表示通过本轮预注册开发门槛。",
        "",
        "## 数据、复现和恢复",
        "",
        "原始自我对弈、开发赛、冠军挑战赛和盲测棋谱以逐局 JSONL 保存，并由程序完整回放验证；局 ID 去重。`integrity.json` 对配置、模型、原始结果、报告和图表给出 SHA-256。",
        "",
        "```bash",
        "python scripts/run_g_v2_growth.py --root .",
        "python scripts/report_g_v2.py --root .",
        "python -m pytest tests/test_alphazero_g_v2_*.py -q",
        "```",
        "",
        f"Drive 恢复快照：{summary.get('drive_status', {}).get('status', '待写入或未验证')}。本研究没有神经网络训练之外的模型训练目标；报告只汇总算法开发、监督自对弈优化和实战评估，不包含虚构的 G0、20k 节点或未执行成绩。",
        "",
        "## 附件",
        "",
        "- `results/growth.csv`：各学习率里程碑数据。",
        "- `results/blind_comparisons.csv`：全部盲测比较统计。",
        "- `results/final_summary.json`：可复算摘要。",
        "- `results/integrity.json`：数据与图表哈希。",
    ])
    report_path = root_path / "research/g_v2/REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(report_path), "figures": figures,
            "growth_csv": str(growth_csv), "blind_csv": str(blind_csv)}


def _integrity_manifest(root: Path) -> dict[str, Any]:
    paths = [
        "research/g_v2/config/experiment.json", "research/g_v2/config/openings.json",
        "research/g_v2/results/preflight/preflight.json", "research/g_v2/results/selection/state.json",
        "research/g_v2/results/selection/stages.jsonl", "research/g_v2/results/evaluation/development.jsonl",
        "research/g_v2/results/evaluation/champion.jsonl", "research/g_v2/results/evaluation/blind.jsonl",
        "research/g_v2/results/evaluation/blind_summary.json", "research/g_v2/results/final_summary.json",
        "research/g_v2/results/growth.csv", "research/g_v2/results/blind_comparisons.csv",
        "research/g_v2/REPORT.md", "research/g_v2/figures/growth_by_states.png",
        "research/g_v2/figures/growth_by_cpu.png", "research/g_v2/figures/network_vs_puct.png",
        "research/f_v2/checkpoints/best_f2.pt",
    ]
    paths += [f"research/g_v2/results/selfplay/{lr}.jsonl" for lr in LR_ORDER]
    paths += [f"research/g_v2/checkpoints/{lr}/progress.json" for lr in LR_ORDER]
    artifacts = {}
    for relative in paths:
        file = root / relative
        if not file.is_file():
            raise FileNotFoundError(f"integrity input is missing: {relative}")
        data = file.read_bytes()
        artifacts[relative] = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return {"format_version": 1, "artifacts": artifacts}


def write_v2_integrity_manifest(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest = _integrity_manifest(root_path)
    path = root_path / "research/g_v2/results/integrity.json"
    _write_json(path, manifest)
    for relative, item in manifest["artifacts"].items():
        data = (root_path / relative).read_bytes()
        if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"artifact changed during V2 integrity audit: {relative}")
    return manifest
