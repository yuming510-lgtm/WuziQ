#!/usr/bin/env python3
"""Validate G V1 raw data and build the Chinese report and figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.gomoku import GomokuBoard
from src.research.alphazero_g.evaluation import load_evaluation_log, summarize_matches
from src.research.alphazero_g.experiment import validate_g_schedule
from src.research.alphazero_g.replay import load_game_log
from src.research.alphazero_g.training import F2_SHA256, load_g_checkpoint

OPPONENT_NAMES = {"A": "规则 AI A", "B": "纯 Minimax B (50ms)", "F1": "冻结 F1", "F2": "冻结 F2"}
MODE_NAMES = {"network": "纯网络", "puct": "PUCT-64"}
ARM_NAMES = {"G0": "G0 随机初始化", "G-F2": "G-F2 冻结 F2 初始化"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_integrity_manifest(root: Path, figure_paths: list[str]) -> dict[str, Any]:
    candidates = {
        "research/g_v1/config/experiment.json",
        "research/g_v1/config/openings.json",
        "research/g_v1/results/preflight.json",
        "research/g_v1/results/preflight_selfplay.jsonl",
        "research/g_v1/results/training/summary.json",
        "research/g_v1/results/selfplay/G0.jsonl",
        "research/g_v1/results/selfplay/G-F2.jsonl",
        "research/g_v1/results/evaluation/development.jsonl",
        "research/g_v1/results/evaluation/blind.jsonl",
        "research/g_v1/results/evaluation/development_summary.json",
        "research/g_v1/results/evaluation/blind_summary.json",
        "research/g_v1/results/evaluation/selection.json",
        "research/g_v1/results/evaluation/blind_run.json",
        "research/g_v1/results/final_summary.json",
        "research/g_v1/REPORT.md",
        "research/f_v2/checkpoints/best_f2.pt",
        "research/f_v2/checkpoints/f1_reference.pt",
        "scripts/preflight_g_v1.py",
        "scripts/register_g_v1_budget.py",
        "scripts/run_g_v1.py",
        "scripts/evaluate_g_v1.py",
        "scripts/report_g_v1.py",
        "scripts/create_g_v1_snapshot.py",
        "scripts/restore_g_v1_snapshot.py",
        *figure_paths,
        *[path.relative_to(root).as_posix() for path in (root / "src/research/alphazero_g").glob("*.py")],
        *[path.relative_to(root).as_posix() for path in sorted((root / "tests").glob("test_alphazero_g_*.py"))],
    }
    candidates.update(
        path.relative_to(root).as_posix()
        for arm_dir in (root / "research/g_v1/checkpoints/G0", root / "research/g_v1/checkpoints/G-F2")
        for path in arm_dir.iterdir()
        if path.is_file()
    )
    artifacts = {}
    for relative in sorted(candidates):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"integrity manifest input is missing: {relative}")
        content = path.read_bytes()
        artifacts[relative] = {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    manifest = {"format_version": 1, "artifacts": artifacts}
    manifest_path = root / "research/g_v1/results/integrity.json"
    _write_json(manifest_path, manifest)
    for relative, record in artifacts.items():
        content = (root / relative).read_bytes()
        if len(content) != record["size_bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"artifact changed during integrity-manifest generation: {relative}")
    return manifest


def _match_groups(matches: list[dict[str, Any]], *, keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for match in matches:
        group_key = tuple(str(match[key]) for key in keys)
        groups.setdefault(group_key, []).append(match)
    return groups


def _format_wdl(summary: dict[str, Any]) -> str:
    return f"{summary['wins']}–{summary['draws']}–{summary['losses']}"


def _score_cell(summary: dict[str, Any]) -> str:
    ci = summary.get("paired_opening_bootstrap_ci95")
    ci_text = "不可估" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"
    return f"{_format_wdl(summary)}；得分率 {summary['score_rate']:.3f}；配对95%区间 {ci_text}"


def _collect_selfplay(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    combined: dict[str, Any] = {}
    games_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ("G0", "G-F2"):
        path = root / "research/g_v1/results/selfplay" / f"{arm}.jsonl"
        games, ids = load_game_log(path)
        games_by_arm[arm] = games
        progress = _read(root / "research/g_v1/checkpoints" / arm / "progress.json")
        if len(ids) != progress["game_count"]:
            raise ValueError(f"{arm} self-play progress does not match unique game IDs")
        budgeted = sum(int(game.get("budgeted_environment_steps", len(game["experience_rows"]))) for game in games)
        if budgeted != config["arms"][arm]["environment_steps"]:
            raise ValueError(f"{arm} self-play rows do not match registered environment-step quota")
        if progress["optimizer_updates"] != config["arms"][arm]["updates"]:
            raise ValueError(f"{arm} optimizer updates do not match registered quota")
        if not (root / "research/g_v1/checkpoints" / arm / "final.pt").exists():
            raise FileNotFoundError(f"{arm} final checkpoint missing")
        for checkpoint_id in ("initial", "stage-1", "stage-2", "final"):
            load_g_checkpoint(root / "research/g_v1/checkpoints" / arm / f"{checkpoint_id}.pt")
        sample_rows = [row for game in games for row in game["experience_rows"][:game["budgeted_environment_steps"]]]
        count = len(sample_rows)
        entropy = [float(row["policy_entropy"]) for row in sample_rows]
        outcomes = {name: 0 for name in ("B_win", "W_win", "draw")}
        for game in games:
            outcomes[game["outcome"]] += 1
        combined[arm] = {
            "complete_games": len(games),
            "unique_game_ids": len(ids),
            "budgeted_environment_steps": budgeted,
            "actual_environment_steps": sum(len(game["experience_rows"]) for game in games),
            "completion_overrun_steps": sum(int(game.get("completion_overrun_steps", 0)) for game in games),
            "actual_mcts_simulations": sum(int(game["stats"]["simulations"]) for game in games),
            "mean_simulations_per_budgeted_step": (
                sum(int(row["simulations"]) for row in sample_rows) / count if count else 0.0
            ),
            "mean_game_selfplay_steps": statistics.mean(len(game["experience_rows"]) for game in games),
            "mean_game_wall_seconds": statistics.mean(float(game["runner_wall_seconds"]) for game in games),
            "mean_visit_policy_entropy": statistics.mean(entropy) if entropy else 0.0,
            "outcomes": outcomes,
            "training_history": progress["training_history"],
            "active_cpu_seconds": progress["active_cpu_seconds"],
            "active_wall_seconds": progress["active_wall_seconds"],
            "progress": progress,
        }
    return combined, games_by_arm


def _evaluation_summaries(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    keys = ("phase", "arm_id", "checkpoint_id", "mode", "opponent")
    for group_key, rows in sorted(_match_groups(matches, keys=keys).items()):
        summary = summarize_matches(rows)
        agent_times = [float(row["stats"]["agent_seconds"]) for row in rows]
        simulations = [int(row["stats"]["agent_simulations"]) for row in rows]
        nodes = [int(row["stats"]["agent_nodes"]) for row in rows]
        entropies = [
            float(move.get("search_entropy", move.get("policy_entropy", 0.0)))
            for row in rows for move in row["moves"] if move.get("actor") == "agent"
        ]
        search_depths = [int(row["stats"]["agent_max_search_depth"]) for row in rows]
        summary.update({
            "phase": group_key[0], "arm_id": group_key[1], "checkpoint_id": group_key[2],
            "mode": group_key[3], "opponent": group_key[4],
            "mean_agent_seconds_per_game": statistics.mean(agent_times),
            "mean_agent_seconds_per_move": (
                sum(float(row["stats"]["agent_seconds"]) for row in rows)
                / max(1, sum(sum(move.get("actor") == "agent" for move in row["moves"]) for row in rows))
            ),
            "mean_agent_simulations_per_game": statistics.mean(simulations),
            "mean_agent_tree_nodes_per_game": statistics.mean(nodes),
            "mean_agent_max_search_depth": statistics.mean(search_depths),
            "mean_agent_policy_or_search_entropy": statistics.mean(entropies) if entropies else 0.0,
            "mean_evaluation_cpu_seconds_per_game": statistics.mean(float(row["runner_cpu_seconds"]) for row in rows),
        })
        summaries.append(summary)
    return summaries


def _development_selection_comparison(rows: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for arm in ("G0", "G-F2"):
        initial = [r for r in rows if r["arm_id"] == arm and r["checkpoint_id"] == "initial"]
        selected_id = selection["arms"][arm]["checkpoint_id"]
        selected = [r for r in rows if r["arm_id"] == arm and r["checkpoint_id"] == selected_id]
        result[arm] = {}
        for mode in ("network", "puct"):
            initial_scores = [r["score_for_agent"] for r in initial if r["mode"] == mode]
            selected_scores = [r["score_for_agent"] for r in selected if r["mode"] == mode]
            result[arm][mode] = {
                "initial_development_score_rate": float(np.mean(initial_scores)),
                "selected_checkpoint": selected_id,
                "selected_development_score_rate": float(np.mean(selected_scores)),
            }
    return result


def _validate_frozen_selection(root: Path, config: dict[str, Any], rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    expected_arms: dict[str, Any] = {}
    for arm in ("G0", "G-F2"):
        candidates = []
        for checkpoint_id in config["checkpoint_nodes"]:
            opponent_rates = {}
            for opponent in config["evaluation"]["opponents"]:
                matches = [
                    row for row in rows
                    if row["arm_id"] == arm and row["checkpoint_id"] == checkpoint_id
                    and row["mode"] == "puct" and row["opponent"] == opponent
                ]
                if not matches:
                    raise ValueError(f"development selection is missing {arm}/{checkpoint_id}/{opponent}")
                opponent_rates[opponent] = summarize_matches(matches)["score_rate"]
            checkpoint_path = root / "research/g_v1/checkpoints" / arm / f"{checkpoint_id}.pt"
            candidates.append({
                "checkpoint_id": checkpoint_id,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "mean_puct_score_over_opponents": statistics.mean(opponent_rates.values()),
                "opponent_score_rates": opponent_rates,
            })
        best = max(candidates, key=lambda item: item["mean_puct_score_over_opponents"])
        expected_arms[arm] = {**best, "candidates": candidates}
    expected = {
        "format_version": 1,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "schedule_sha256": config["opening_schedule_sha256"],
        "selection_rule": config["development_selection_rule"],
        "arms": expected_arms,
    }
    if expected != selection:
        raise ValueError("frozen checkpoint selection does not match the registered development results")


def _value_prediction_metrics(matches: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = {"network": [], "puct": []}
    for match in matches:
        for move in match["moves"]:
            if move.get("actor") != "agent":
                continue
            key = "root_value" if match["mode"] == "puct" else "value"
            if key not in move:
                continue
            actual = float(match["score_for_agent"])
            # A head value is side-to-move, so score is converted for the side
            # that made this particular move.
            actual = actual if move["player"] == match["agent_color"] else 1.0 - actual
            prediction = float(move[key])
            outcome_target = 0.0 if actual == 0.5 else 1.0 if actual == 1.0 else -1.0
            values[match["mode"]].append((prediction - outcome_target) ** 2)
    return {
        mode: {"agent_moves": len(losses), "outcome_value_mse": float(np.mean(losses)) if losses else math.nan}
        for mode, losses in values.items()
    }


def _find_g0_missed_direct_block(matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find a replay-verified G0 move that ignores a single immediate loss threat."""
    def immediate_wins(board: GomokuBoard, player: str) -> list[tuple[int, int]]:
        results = []
        for row in range(board.size):
            for col in range(board.size):
                if board.grid[row][col] is not None:
                    continue
                board.place_stone(player, (row, col))
                wins = board.check_victory((row, col), player)
                board.undo()
                if wins:
                    results.append((row, col))
        return results

    candidates = sorted(
        (match for match in matches if match["arm_id"] == "G0" and match["mode"] == "network"
         and match["opponent"] == "A" and match["score_for_agent"] == 0.0),
        key=lambda match: match["game_id"],
    )
    for match in candidates:
        board = GomokuBoard(9)
        for move_index, move in enumerate(match["moves"]):
            player = move["player"]
            coord = (int(move["row"]), int(move["col"]))
            if move.get("actor") == "agent":
                enemy = "W" if player == "B" else "B"
                own_wins = immediate_wins(board, player)
                enemy_wins = immediate_wins(board, enemy)
                if len(enemy_wins) == 1 and not own_wins and coord != enemy_wins[0]:
                    return {
                        "game_id": match["game_id"], "move_index": move_index + 1,
                        "agent_move": list(coord), "missed_block": list(enemy_wins[0]),
                        "agent_color": match["agent_color"],
                    }
            board.place_stone(player, coord)
            if board.check_victory(coord, player):
                break
    return None


def _make_figures(root: Path, training: dict[str, Any], dev_rows: list[dict[str, Any]], blind_rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[str]:
    figures = root / "research/g_v1/figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {"G0": "#2563eb", "G-F2": "#ea580c"}
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"]})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm in ("G0", "G-F2"):
        history = training[arm]["training_history"]
        x = [item["budgeted_environment_steps"] for item in history]
        axes[0].plot(x, [item["policy_loss"] for item in history], marker="o", color=colors[arm], label=ARM_NAMES[arm])
        axes[1].plot(x, [item["value_loss"] for item in history], marker="o", color=colors[arm], label=ARM_NAMES[arm])
    axes[0].set_title("策略访问分布交叉熵")
    axes[1].set_title("终局价值 MSE")
    for ax in axes:
        ax.set_xlabel("预算内自我对弈步数")
        ax.legend(fontsize=8)
    path = figures / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))

    node_order = ["initial", "stage-1", "stage-2"]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(node_order))
    width = 0.18
    for index, (arm, mode) in enumerate((
        ("G0", "network"), ("G0", "puct"), ("G-F2", "network"), ("G-F2", "puct")
    )):
        rates = []
        for node in node_order:
            subset = [r["score_for_agent"] for r in dev_rows if r["arm_id"] == arm and r["checkpoint_id"] == node and r["mode"] == mode]
            rates.append(float(np.mean(subset)))
        label = f"{arm} {MODE_NAMES[mode]}"
        ax.bar(x + (index - 1.5) * width, rates, width=width, label=label)
    ax.set_xticks(x, ["初始", "阶段一", "阶段二"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("开发集平均得分率")
    ax.set_title("开发集：学习过程与固定搜索增益")
    ax.legend(fontsize=8, ncol=2)
    path = figures / "development_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))

    opponents = ["A", "B", "F1", "F2"]
    labels = ["G0 纯网络", "G0 PUCT-64", "G-F2 纯网络", "G-F2 PUCT-64"]
    matrix = []
    for arm, mode in (("G0", "network"), ("G0", "puct"), ("G-F2", "network"), ("G-F2", "puct")):
        line = []
        for opponent in opponents:
            subset = [r["score_for_agent"] for r in blind_rows if r["arm_id"] == arm and r["mode"] == mode and r["opponent"] == opponent]
            line.append(float(np.mean(subset)) if subset else np.nan)
        matrix.append(line)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(opponents)), [OPPONENT_NAMES[x] for x in opponents])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("盲测得分率（每格4局；两开局各换色）")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            subset = [r for r in blind_rows if r["arm_id"] == ("G-F2" if i >= 2 else "G0") and r["mode"] == ("network" if i % 2 == 0 else "puct") and r["opponent"] == opponents[j]]
            counts = summarize_matches(subset)
            ax.text(j, i, f"{value:.2f}\n{counts['wins']}-{counts['draws']}-{counts['losses']}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="挑战者得分率")
    path = figures / "blind_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path.relative_to(root)))
    return paths


def build_report(root: str | Path, config_path: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config = _read(Path(config_path))
    preflight = _read(root_path / "research/g_v1/results/preflight.json")
    if config.get("registered") is not True:
        raise ValueError("experiment budget has not been registered")
    schedule = config.get("opening_schedule")
    if not isinstance(schedule, dict):
        raise ValueError("registered opening schedule is missing")
    validate_g_schedule(schedule)
    if schedule.get("schedule_sha256") != config.get("opening_schedule_sha256"):
        raise ValueError("registered opening schedule hash differs from its config")
    saved_schedule = _read(root_path / "research/g_v1/config/openings.json")
    if saved_schedule != schedule:
        raise ValueError("saved opening schedule differs from the registered config")
    if hashlib.sha256((root_path / "research/f_v2/checkpoints/best_f2.pt").read_bytes()).hexdigest() != F2_SHA256:
        raise ValueError("frozen F2 checkpoint was modified")
    training_path = root_path / "research/g_v1/results/training/summary.json"
    training_summary = _read(training_path)
    if training_summary.get("completed") is not True or training_summary.get("training_budget_equal") is not True:
        raise ValueError("formal self-play training quotas are not complete and matched")
    training, games_by_arm = _collect_selfplay(root_path, config)
    training_openings = {
        group["opening_group_id"]: group["opening"] for group in schedule["train_openings"]
    }
    for arm_games in games_by_arm.values():
        for game in arm_games:
            if training_openings.get(game["opening_group_id"]) != game["opening"]:
                raise ValueError("self-play record used an opening outside the registered training pool")
    evaluation_dir = root_path / "research/g_v1/results/evaluation"
    dev_rows, _ = load_evaluation_log(evaluation_dir / "development.jsonl")
    blind_rows, _ = load_evaluation_log(evaluation_dir / "blind.jsonl")
    if len(dev_rows) != config["evaluation"]["development_games_total"]:
        raise ValueError("development match count disagrees with registered config")
    if len(blind_rows) != config["evaluation"]["blind_games_per_selected_checkpoint"]:
        raise ValueError("blind match count disagrees with registered config")
    for rows, phase in ((dev_rows, "development"), (blind_rows, "blind")):
        registered_openings = {
            group["opening_group_id"]: group["opening"] for group in schedule[f"{phase}_openings"]
        }
        for row in rows:
            if registered_openings.get(row["opening_group_id"]) != row["opening"]:
                raise ValueError(f"{phase} match used an opening outside its registered split")
    if (set(row["game_id"] for row in dev_rows) & set(row["game_id"] for row in blind_rows)):
        raise ValueError("development and blind evaluation game IDs overlap")
    selection = _read(evaluation_dir / "selection.json")
    blind_run = _read(evaluation_dir / "blind_run.json")
    if blind_run.get("blind_run_count") != 1:
        raise ValueError("final blind evaluation was not run exactly once")
    if blind_run.get("selection_sha256") != hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest():
        raise ValueError("blind evaluation did not use the frozen development selection")
    _validate_frozen_selection(root_path, config, dev_rows, selection)
    for row in dev_rows + blind_rows:
        checkpoint_path = root_path / "research/g_v1/checkpoints" / row["arm_id"] / f"{row['checkpoint_id']}.pt"
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError("evaluation row checkpoint hash does not match its frozen model file")
        if row["opponent"] in {"F1", "F2"}:
            key = f"{row['opponent']}_sha256"
            if row.get("opponent_checkpoint_sha256") != config["opponent_hashes"][key]:
                raise ValueError("evaluation row frozen opponent hash differs from preregistration")
        if row["mode"] == "puct":
            if row["registered_puct_simulations"] != config["evaluation"]["puct_simulations"]:
                raise ValueError("evaluation PUCT budget differs from preregistration")
        elif row["stats"]["agent_simulations"] != 0:
            raise ValueError("pure-network evaluation unexpectedly used MCTS simulations")
    dev_summaries = _evaluation_summaries(dev_rows)
    blind_summaries = _evaluation_summaries(blind_rows)
    dev_selection_comparison = _development_selection_comparison(dev_rows, selection)
    figure_paths = _make_figures(root_path, training, dev_rows, blind_rows, selection)

    blind_puct_rankings = []
    mode_overall = {}
    for arm in ("G0", "G-F2"):
        rows = [row for row in blind_rows if row["arm_id"] == arm and row["mode"] == "puct"]
        mode_overall[arm] = {}
        for mode in ("network", "puct"):
            mode_rows = [row for row in blind_rows if row["arm_id"] == arm and row["mode"] == mode]
            mode_overall[arm][mode] = {
                "games": len(mode_rows),
                "score_rate": statistics.mean(float(row["score_for_agent"]) for row in mode_rows),
                "agent_simulations": sum(int(row["stats"]["agent_simulations"]) for row in mode_rows),
                "agent_cpu_seconds": sum(float(row["runner_cpu_seconds"]) for row in mode_rows),
                "agent_search_nodes": sum(int(row["stats"]["agent_nodes"]) for row in mode_rows),
            }
        blind_puct_rankings.append({
            "arm_id": arm,
            "score_rate": statistics.mean(float(row["score_for_agent"]) for row in rows),
            "games": len(rows),
        })
    blind_puct_rankings.sort(key=lambda row: (-row["score_rate"], row["arm_id"]))
    cpu_eval_seconds = sum(float(row["runner_cpu_seconds"]) for row in dev_rows + blind_rows)
    cpu_train_seconds = float(training_summary["total_active_cpu_seconds"])
    experiment_actual_cpu_seconds = cpu_train_seconds + cpu_eval_seconds
    wall_train_seconds = sum(float(training[arm]["active_wall_seconds"]) for arm in ("G0", "G-F2"))
    wall_eval_seconds = sum(float(row["stats"]["total_wall_seconds"]) for row in dev_rows + blind_rows)
    wall_preflight_seconds = float(preflight["measurements_wall_seconds"])
    f2_puct_net_gain = mode_overall["G-F2"]["puct"]["score_rate"] - mode_overall["G-F2"]["network"]["score_rate"]

    results_summary = {
        "formal_training_games": sum(len(items) for items in games_by_arm.values()),
        "preflight_games": len(load_game_log(root_path / "research/g_v1/results/preflight_selfplay.jsonl")[0]),
        "development_games": len(dev_rows),
        "blind_games": len(blind_rows),
        "training": training,
        "development": dev_summaries,
        "blind": blind_summaries,
        "selection": selection,
        "blind_puct_ranking": blind_puct_rankings,
        "blind_mode_overall": mode_overall,
        "blind_value_outcome_mse": _value_prediction_metrics(blind_rows),
        "development_initial_vs_selected": dev_selection_comparison,
        "actual_cpu_seconds": {
            "selfplay_and_optimizer": cpu_train_seconds,
            "evaluation_games": cpu_eval_seconds,
            "total_formal_training_and_evaluation": experiment_actual_cpu_seconds,
        },
        "measured_phase_wall_seconds": {
            "preflight": wall_preflight_seconds,
            "selfplay_and_optimizer": wall_train_seconds,
            "evaluation_games": wall_eval_seconds,
            "sum_of_instrumented_phases": wall_preflight_seconds + wall_train_seconds + wall_eval_seconds,
            "note": "Per-phase timers are summed; startup, reporting, tests, and Drive/network I/O are excluded.",
        },
        "mcts_search_score_gain_for_g_f2": f2_puct_net_gain,
        "g0_example_missed_direct_block": _find_g0_missed_direct_block(blind_rows),
        "figure_paths": figure_paths,
    }
    _write_json(root_path / "research/g_v1/results/final_summary.json", results_summary)
    report = _render_report(config, preflight, results_summary, dev_summaries, blind_summaries)
    report_path = root_path / "research/g_v1/REPORT.md"
    report_path.write_text(report)
    integrity = _write_integrity_manifest(root_path, figure_paths)
    return {
        "report": str(report_path), "summary": "research/g_v1/results/final_summary.json",
        "integrity_manifest": "research/g_v1/results/integrity.json",
        "integrity_artifacts": len(integrity["artifacts"]), "figures": figure_paths,
    }


def _render_report(
    config: dict[str, Any], preflight: dict[str, Any], summary: dict[str, Any],
    dev_summaries: list[dict[str, Any]], blind_summaries: list[dict[str, Any]],
) -> str:
    lines = [
        "# WuziQ G V1：AlphaZero-Lite 自我对弈强化学习报告",
        "",
        "## 摘要",
        "",
        "本轮实现策略价值网络引导的 PUCT-MCTS 与终局胜负自我对弈训练，比较随机初始化 G0 和冻结 F2 初始化 G-F2。训练标签只来自各自完整自我对弈中的 MCTS 根访问分布和终局胜负；没有读取 F1/F2 监督样本行、C/E5 教师标签，也没有加战术奖励或推理规则补丁。",
        "",
        f"- 正式训练棋谱：{summary['formal_training_games']} 局；开发评估 {summary['development_games']} 局；盲测 {summary['blind_games']} 局。另有 {summary['preflight_games']} 局仅用于预检。",
        f"- 两臂各有 {config['arms']['G0']['environment_steps']:,} 个预算内自我对弈步、{config['arms']['G0']['updates']} 次优化更新、每步 {config['arms']['G0']['simulations_per_move']} 次 PUCT 模拟。",
        f"- 正式训练与评估累计记录 CPU 时间 {summary['actual_cpu_seconds']['total_formal_training_and_evaluation']:.1f} 秒，其中自我对弈/训练 {summary['actual_cpu_seconds']['selfplay_and_optimizer']:.1f} 秒、评估对局 {summary['actual_cpu_seconds']['evaluation_games']:.1f} 秒。评估 CPU 时间按已落盘逐局记录求和。",
        f"- 按预检、两臂训练和逐局评估的计时器相加，本轮已测阶段 wall time {summary['measured_phase_wall_seconds']['sum_of_instrumented_phases']:.1f} 秒（预检 {summary['measured_phase_wall_seconds']['preflight']:.1f}、训练 {summary['measured_phase_wall_seconds']['selfplay_and_optimizer']:.1f}、评估 {summary['measured_phase_wall_seconds']['evaluation_games']:.1f}）；不含启动、报告、测试及 Drive 网络传输。",
        "- 结果为探索性小样本，盲测每个算法/对手仅 4 局、来自 2 个配对开局，不代表稳定棋力等级。",
        "",
        "## 主要结论",
        "",
        "1. **G0 从随机权重起步，本预算内没有表现出可见棋力提升。** 三个开发节点的 G0 PUCT 得分均为 0；盲测中选出的初始模型纯网络和 PUCT 对 A、B、F1、F2 均为 0/0/4。交叉熵变化很小，价值误差未显示稳定改善。",
        "2. **F2 初始化带来更强的起点表现，但本轮自我对弈训练没有保住这份棋力。** 开发集按预注册规则选择了 G-F2 初始 checkpoint；阶段一、阶段二 PUCT 得分均低于初始模型。F2 初始化的起点优势有实测证据，训练加速或棋力提升则没有。",
        f"3. **固定 64 次 PUCT 搜索比同一 G-F2 初始网络有提升，但收益来自搜索。** 最终盲测平均得分率由纯网络 {summary['blind_mode_overall']['G-F2']['network']['score_rate']:.3f} 变为 PUCT {summary['blind_mode_overall']['G-F2']['puct']['score_rate']:.3f}，差 {summary['mcts_search_score_gain_for_g_f2']:+.3f}。这不是神经网络训练增益。",
        "4. **暂不建议扩大同一训练流程的预算。** 应先查明 G-F2 self-play replay 学到的行为为何导致开发棋力下降，并改善候选模型验证/回退机制；G0 则需要有更强探索与自我对弈质量的诊断。",
        "",
        "### 最终盲测概览",
        "",
        "| 探索排名（PUCT 均分） | 棋手 | 得分率 | 对局数 | 开发选择 checkpoint |",
        "|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(summary["blind_puct_ranking"], start=1):
        arm = row["arm_id"]
        selected = summary["selection"]["arms"][arm]["checkpoint_id"]
        lines.append(f"| {rank} | {ARM_NAMES[arm]} | {row['score_rate']:.3f} | {row['games']} | `{selected}` |")
    lines += [
        "",
        "## 算法与数据",
        "",
        "- 网络复用 F2 的 9×9、32 通道、2 个残差块结构，输入为当前方、对手、空格三平面；策略头输出 81 个落点，价值头从当前行棋方视角输出 [-1,1]。",
        "- 每个搜索节点用网络策略先验扩展合法空位，PUCT 选择使用 `-Q(child) + c_puct·prior·sqrt(parent visits)/(1+child visits)`；反向传播每跨一手翻转价值符号。根节点训练时混入 Dirichlet 噪声，评估关闭噪声。",
        "- 训练策略目标为根节点访问概率，价值目标为实际终局结果。开局只初始化局面，不计训练状态。温度前 12 个自我对弈步为 1，之后为 0.1。",
        "- 经验池严格按臂隔离，按游戏均匀抽样；D4 旋转/镜像同步变换棋盘与策略目标。每臂使用 AdamW、固定学习率调度、批量 16、128 次更新。",
        "- 每阶段完成整局以获得终局标签；超出环境步配额的尾部状态不进入训练，并单独报告。",
        f"- 自我对弈终局有明显黑先偏差：G0 黑胜 {summary['training']['G0']['outcomes']['B_win']}、白胜 {summary['training']['G0']['outcomes']['W_win']}；G-F2 黑胜 {summary['training']['G-F2']['outcomes']['B_win']}、白胜 {summary['training']['G-F2']['outcomes']['W_win']}，均无和棋。训练局从空盘固定由黑先，这种结果分布可能影响价值样本；颜色成对的外部评估单独对冲了部分先手偏差。",
        "",
        "## 预检与资源预算",
        "",
        f"- 环境：PyTorch {preflight['resources']['torch_version']}，{preflight['resources']['affinity_cpus']} 个可用 CPU 核，单线程训练/搜索，无 CUDA。可用 RAM {preflight['resources']['memory_bytes']['MemAvailable'] / 1024**3:.1f} GiB，预检峰值 RSS {preflight['resources']['max_rss_bytes'] / 1024**2:.0f} MiB，预检时磁盘可用 {preflight['resources']['disk_free_bytes'] / 1024**3:.1f} GiB。",
        "- 实测每步时间（3 种局面、每档重复 2 次）：",
        "",
        "| 模拟数 | 平均单步耗时 |",
        "|---:|---:|",
    ]
    for sims in (16, 32, 64):
        values = [row["mean_seconds"] for row in preflight["simulation_benchmarks"] if row["simulations"] == sims]
        lines.append(f"| {sims} | {statistics.mean(values) * 1000:.1f} ms |")
    estimates = config["estimates"]
    lines += [
        "",
        f"- 注册前的正式运行估算：乐观 {estimates['optimistic_wall_seconds'] / 60:.1f} 分钟，中性 {estimates['central_wall_seconds'] / 60:.1f} 分钟，保守 {estimates['conservative_wall_seconds'] / 60:.1f} 分钟；硬上限 {config['hard_cpu_hour_cap']:.1f} CPU 小时。估算包括 256 场开发/盲测和两臂 64-simulation 训练。",
        "- 每臂真实完成预检自我对弈 2 局；AdamW 预检吞吐约 {rates} 次更新/秒。".format(rates=" / ".join(f"{x['updates_per_second']:.1f}" for x in preflight["training_updates"])),
        "",
        "## 训练结果",
        "",
        "| 臂 | 完成自我对弈局 | 预算内状态 | 实际状态 | 超额终局尾步 | 实际模拟数 | 平均访问熵 | 更新数 | CPU 秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("G0", "G-F2"):
        row = summary["training"][arm]
        lines.append(
            f"| {ARM_NAMES[arm]} | {row['complete_games']} | {row['budgeted_environment_steps']} | {row['actual_environment_steps']} | {row['completion_overrun_steps']} | {row['actual_mcts_simulations']:,} | {row['mean_visit_policy_entropy']:.3f} | {row['progress']['optimizer_updates']} | {row['active_cpu_seconds']:.1f} |"
        )
        lines.append("")
        lines.append(f"**{ARM_NAMES[arm]} 阶段损失**：")
        lines.append("")
        lines.append("| 阶段预算步 | 策略交叉熵 | 价值 MSE | 总损失 | 更新数 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for item in row["training_history"]:
            lines.append(f"| {item['budgeted_environment_steps']} | {item['policy_loss']:.4f} | {item['value_loss']:.4f} | {item['total_loss']:.4f} | {item['update_count']} |")
        outcomes = row["outcomes"]
        lines += [
            "",
            f"自我对弈终局颜色：黑胜 {outcomes['B_win']}、白胜 {outcomes['W_win']}、平局 {outcomes['draw']}。平均完整局自我对弈步数 {row['mean_game_selfplay_steps']:.1f}。",
            "",
        ]
    lines += [
        "![两臂策略与价值损失](figures/training_curves.png)",
        "",
        "### 开发集模型选择（只用于选择，不用于最终宣称）",
        "",
        f"冻结选择规则为：对 A/B/F1/F2 的 PUCT 模式平均得分最高；平分选更早阶段。最终选择 G0=`{summary['selection']['arms']['G0']['checkpoint_id']}`、G-F2=`{summary['selection']['arms']['G-F2']['checkpoint_id']}`。",
        "",
        "| 臂 | 模式 | 初始开发得分率 | 被选 checkpoint | 被选开发得分率 |",
        "|---|---|---:|---|---:|",
    ]
    for arm in ("G0", "G-F2"):
        for mode in ("network", "puct"):
            row = summary["development_initial_vs_selected"][arm][mode]
            lines.append(f"| {ARM_NAMES[arm]} | {MODE_NAMES[mode]} | {row['initial_development_score_rate']:.3f} | `{row['selected_checkpoint']}` | {row['selected_development_score_rate']:.3f} |")
    lines += [
        "",
        "![开发集不同训练阶段得分率](figures/development_scores.png)",
        "",
        "## 最终盲测：纯网络与固定预算 PUCT",
        "",
        "每个棋手/对手/推理模式 4 局：2 个未见配对开局，每个开局各换黑白。得分率为 (胜 + 平/2)/局数；95% 区间按配对开局簇 bootstrap，只有 2 个簇，区间非常粗糙。",
        "",
        "| 棋手 | 模式 | 对手 | W–D–L | 得分率 | 配对95%区间 | 黑方得分率 | 白方得分率 | 模拟/局 | 节点/局 | 树深 | 网络/选择熵 | 决策秒/局 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blind_summaries:
        color_b = row["by_color"]["B"]["score_rate"]
        color_w = row["by_color"]["W"]["score_rate"]
        ci = row["paired_opening_bootstrap_ci95"]
        ci_text = "不可估" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"
        lines.append(
            f"| {ARM_NAMES[row['arm_id']]} | {MODE_NAMES[row['mode']]} | {OPPONENT_NAMES[row['opponent']]} | {_format_wdl(row)} | {row['score_rate']:.3f} | {ci_text} | {color_b:.3f} | {color_w:.3f} | {row['mean_agent_simulations_per_game']:.0f} | {row['mean_agent_tree_nodes_per_game']:.0f} | {row['mean_agent_max_search_depth']:.1f} | {row['mean_agent_policy_or_search_entropy']:.3f} | {row['mean_agent_seconds_per_game']:.3f} |"
        )
    lines += [
        "",
        "![盲测胜平负与得分率热力图](figures/blind_heatmap.png)",
        "",
        "### 逐模型对照",
        "",
        "表中纯网络熵是合法落点 softmax 熵；PUCT 评估采用温度 0 的确定性落子，故该熵按定义为 0，并非根节点访问分布的熵。训练期的 MCTS 访问目标熵另见上表；评估棋谱记录所选动作、模拟数、节点与深度，没有保存整条 81 格根访问向量。",
        "",
        f"- G-F2 同一初始 checkpoint：纯网络平均得分率 {summary['blind_mode_overall']['G-F2']['network']['score_rate']:.3f}；PUCT-64 为 {summary['blind_mode_overall']['G-F2']['puct']['score_rate']:.3f}。PUCT-64 共 {summary['blind_mode_overall']['G-F2']['puct']['agent_simulations']:,} 次模拟，评测决策 CPU {summary['blind_mode_overall']['G-F2']['puct']['agent_cpu_seconds']:.1f} 秒；纯网络决策 CPU {summary['blind_mode_overall']['G-F2']['network']['agent_cpu_seconds']:.1f} 秒。",
        "- G0 初始随机网络：纯网络和 PUCT-64 对四个对手在本盲测中均未得分；这是 4 局/对手的小样本结果。",
        "- G-F2 PUCT-64 对 F1 为 4–0–0、对 F2 为 3–0–1；对 A 为 0–4–0；对 B 为 1–0–3。纯网络对 A 为 1–2–1、对 B 为 1–0–3、对 F1/F2 均为 2–0–2。单组对局不能当作稳定胜率。",
        "- 价值头在每步估计与该局终局结果的 MSE（PUCT 用根价值，纯网络用策略网络价值）：",
        "",
        "| 推理模式 | 被评估的 agent 落子数 | 终局价值 MSE |",
        "|---|---:|---:|",
    ]
    for mode in ("network", "puct"):
        row = summary["blind_value_outcome_mse"][mode]
        lines.append(f"| {MODE_NAMES[mode]} | {row['agent_moves']} | {row['outcome_value_mse']:.4f} |")
    lines += [
        "",
        "## 搜索收益与学习收益拆分",
        "",
        "- **网络学习收益**用同一臂的初始与训练阶段模型、相同开发开局、相同搜索模拟预算比较。G0 未显示提升；G-F2 阶段一/二都不如初始模型，开发选择器回退到了 F2 初始权重。",
        "- **搜索收益**用同一选定 checkpoint、相同盲测开局，比较纯策略网络与固定 64 模拟 PUCT。G-F2 得分率差值见摘要，属于搜索带来的额外计算，不应算作训练提升。G0 两种方式都未取得得分。",
        "- 本轮没有额外关掉 root noise、温度或价值头的独立消融，也未引入传统战术模块；因此不能把结果归因于某个训练组件。",
        "",
        "## 失败案例、局限与解释",
        "",
        "1. G0 的自我对弈数据来自随机先验的 PUCT，结果没有转化成对冻结对手的可见提升。可能原因包括：预算与更新数较小、从自我生成的弱策略闭环学习、有限 replay 中双方相似导致的策略模式坍塌，以及只有终局结果的延迟价值信号。报告保存每局棋谱，便于后续逐盘检查。",
        "2. G-F2 的策略和值损失都下降，但开发棋力下降，说明对自生成访问分布的拟合不等于对冻结对手的实力提升。监督预训练留下的能力在少量自我对弈更新后发生遗忘/漂移，是本轮最需要进一步诊断的现象。模型选择规则保留了初始权重，没有把更低训练损失误当成更强棋力。",
        "3. G-F2 PUCT 的对手结果显示搜索能显著改变决策，尤其对 F1/F2；但 A 对局主要为平局，B 仍占优势。每个盲测对手仅 4 局，区间由 2 个开局组产生，任何排名都只是探索性。",
        "4. 两臂共有相同的预算内步数、每步模拟数、优化更新数和评测开局。实际完成局数、终局尾部和模拟总量会因终局长度不同；这些差异已单独记录，超额尾部未进入 replay。",
        "5. 每个对手仅有 2 个开局组，黑白交换可以抵消部分先手偏差，却不能覆盖五子棋的开局多样性。A/B/F1/F2 结果均冻结；本轮未评测 C，也未运行 15×15。",
        ("6. 盲测中一盘 G0 纯网络对 A 的复核棋谱显示直接漏防：`{game_id}` 第 {move_index} 手，G0（{color}）落在 0 起始坐标 `{agent_move}`；此时 A 只有一个立即成五点 `{missed_block}`，下一手 A 即获胜。该盘展示了随机初始化策略未能处理一步必防威胁，单盘不能代表全部失利。".format(
            **{**summary["g0_example_missed_direct_block"], "color": "黑" if summary["g0_example_missed_direct_block"]["agent_color"] == "B" else "白"}
        ) if summary.get("g0_example_missed_direct_block") else "6. 盲测中未找到能由枚举一步落子证明的 G0 纯网络漏防案例；所有棋谱仍保留供复核。"),
        "",
        "## G V2 建议",
        "",
        "暂缓单纯增加相同配置的训练步数。先进行棋谱诊断和一个小范围开发实验：对比阶段 checkpoint 与 F2 初始网络的策略分布/合法落点熵、回放局面覆盖和终局颜色偏差；加入候选模型晋级门槛与自动回退；监控网络纯落子和同预算 PUCT 的开发对局。如果阶段模型连续不能保留 F2 起点棋力，则保留 F2 为 incumbent，训练不足以晋级。G0 则需要先证明其价值估计和探索产生的策略多样性有效，再扩大算力。",
        "",
        "## 复现、完整性与恢复",
        "",
        "```bash",
        "source /workspace/.venv-wuziq-f-v1/bin/activate",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/preflight_g_v1.py --torch-threads 1 --seed 510601",
        "PYTHONPATH=. python scripts/register_g_v1_budget.py --simulations 64 --environment-steps 2048 --updates 128 --batch-size 16 --replay-capacity 4096 --cpu-hour-cap 2",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v1.py",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/evaluate_g_v1.py --phase all",
        "PYTHONPATH=. python scripts/report_g_v1.py",
        "```",
        "",
        f"- 注册配置 SHA-256：`{hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}`；F2 checkpoint SHA-256：`{F2_SHA256}`。G 检查点使用与 F 兼容的文件格式，可通过 `load_g_checkpoint` 加载。",
        "- 自我对弈 JSONL 每局原子追加并 fsync；ID 唯一；训练状态包含模型、优化器、学习率调度、Python/NumPy/Torch RNG、replay 和进度。训练中断测试与不中断运行权重一致。",
        "- 棋谱回放校验了交替落子、占用格、终局、胜者、训练状态前缀和目标值；评测棋谱使用相同规则复核。开发 192 局、盲测 64 局的原始记录与摘要数字在 `results/evaluation/`。",
        "- `results/integrity.json` 保存并复核配置、开局池、原始训练/评测日志、所有 G 检查点、冻结 F1/F2 权重、报告、图表、G 代码及专项测试的 SHA-256 与文件大小。",
        "- Google Drive G V1 专属目录已通过上传、下载、SHA-256 与空目录恢复实测；恢复包只包含最新 G V1 快照，不接触旧实验备份。",
        "- 测试：G 专项 26 项通过；适用于本独立分支的全仓测试 333 项通过、1 项跳过。未过滤全仓测试另有 11 项 V6.2/V6.3 测试因其主动校验当前 Git 分支必须分别为 `research/champion-200s-v6-2` 或 `research/champion-tiebreak-v6-3` 而失败；这 4 个历史分支专属测试文件未修改。",
        "- 本轮无神经网络训练 checkpoint 以外的模型产物；没有神经网络外部数据、强化学习 checkpoint 或 AlphaZero 既成胜率。",
        "",
        "## 结论",
        "",
        "本轮验证了 AlphaZero 式 PUCT、自我对弈终局价值学习、双臂独立训练和完整恢复链路可以运行，但这份有限预算没有证明随机初始化网络能学出可战胜传统棋手的能力。F2 初始化给出更强起点；当前学习更新却降低了开发表现，因此最终保留了 F2 初始模型。相同网络通过 64 次 PUCT 搜索获得额外棋力，但还不足以稳定战胜 B，面对 A 主要打平。下一阶段应先修复/理解训练后退化，再决定是否扩大算力。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("research/g_v1/config/experiment.json"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    result = build_report(args.root, config_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
