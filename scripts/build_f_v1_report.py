#!/usr/bin/env python3
"""Build charts, consistency checks, and the Chinese F V1 research report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.gomoku import GomokuBoard
from src.research.imitation_f.dataset import canonicalize_position, validate_game_record


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_artifacts(root: Path) -> dict:
    base = root / "research/f_v1"
    data, results = base / "data", base / "results"
    partitions = {name: _jsonl(data / f"{name}.jsonl") for name in ("train", "validation", "blind_test")}
    state_sets = {name: {row["state_key"] for row in rows} for name, rows in partitions.items()}
    overlaps = {
        "train_validation": len(state_sets["train"] & state_sets["validation"]),
        "train_blind": len(state_sets["train"] & state_sets["blind_test"]),
        "validation_blind": len(state_sets["validation"] & state_sets["blind_test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"canonical states cross data splits: {overlaps}")
    for name, rows in partitions.items():
        for row in rows:
            policy, board = row["policy_target"], row["canonical_board"]
            if len(policy) != 81 or not math.isclose(sum(policy), 1.0, abs_tol=1e-5):
                raise ValueError(f"malformed policy target in {name}")
            if not -1 <= float(row["value_target"]) <= 1:
                raise ValueError(f"invalid value target in {name}")
            if len(board) != 81 or any(cell not in ".12" for cell in board):
                raise ValueError(f"invalid canonical board in {name}")
            if any(policy[i] > 1e-7 for i, cell in enumerate(board) if cell != "."):
                raise ValueError(f"illegal policy label in {name}")
    split_manifest = json.loads((data / "historical_split_manifest.json").read_text(encoding="utf-8"))
    assignment = split_manifest["opening_group_assignment"]
    games = _jsonl(results / "generated_games.jsonl")
    game_ids = [game["game_id"] for game in games]
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("duplicate generated game ids")
    for game in games:
        validate_game_record(game)
        if game["group_id"] in assignment and assignment[game["group_id"]] != "train":
            raise ValueError("generated game is not training-only")
    annotations = _jsonl(results / "generated_annotations.jsonl")
    annotation_ids = [row["annotation_id"] for row in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("duplicate annotation ids")
    matches = _jsonl(results / "matches.jsonl")
    match_ids = [game["game_id"] for game in matches]
    if len(matches) != 6 or len(match_ids) != len(set(match_ids)):
        raise ValueError("match set must contain six unique games")
    for game in matches:
        validate_game_record(game)
    state_collisions = set()
    for game in matches:
        board = GomokuBoard(9)
        for move in game["moves"]:
            if move.get("model") is not None:
                key, _ = canonicalize_position(board, str(move["player"]))
                if key in state_sets["train"]:
                    state_collisions.add(key)
            board.place_stone(str(move["player"]), (int(move["row"]), int(move["col"])))
    if state_collisions:
        raise ValueError(f"evaluation match states overlap training data: {len(state_collisions)}")
    opening_blocks = {}
    for game in matches:
        opponent = next(model for model in game["pair"] if model != "F")
        opening_blocks.setdefault(opponent, []).append(game)
    for opponent, block in opening_blocks.items():
        if len(block) != 2 or block[0]["opening"] != block[1]["opening"]:
            raise ValueError(f"{opponent} match does not use a color-paired shared opening")
        f_colors = {"B" if game["black_model"] == "F" else "W" for game in block}
        if f_colors != {"B", "W"}:
            raise ValueError(f"{opponent} match lacks a balanced color swap")
    return {
        "partition_counts": {name: len(rows) for name, rows in partitions.items()},
        "canonical_overlap_counts": overlaps,
        "generated_games": len(games),
        "generated_annotations": len(annotations),
        "unique_annotation_ids": len(set(annotation_ids)),
        "matches": len(matches),
        "match_training_state_collisions": len(state_collisions),
        "replay_validation": "generated and match records replayed legally to terminal",
        "policy_legality": "all targets normalized; occupied squares have zero target mass",
    }


def _match_summary(matches: list[dict]) -> tuple[dict, list[dict]]:
    summary, rows = {}, []
    for opponent in ("A", "B", "C"):
        games = [game for game in matches if opponent in game["pair"]]
        points = sum(
            1.0 if game["winner_model"] == "F" else
            0.5 if game["winner_model"] is None else 0.0
            for game in games
        )
        wins = sum(game["winner_model"] == "F" for game in games)
        draws = sum(game["winner_model"] is None for game in games)
        losses = len(games) - wins - draws
        summary[opponent] = {
            "games": len(games), "wins": wins, "draws": draws, "losses": losses,
            "points": points, "score_rate": points / len(games) if games else 0,
        }
        for game in games:
            f_color = "B" if game["black_model"] == "F" else "W"
            outcome = "平" if game["winner_model"] is None else ("F胜" if game["winner_model"] == "F" else f"{opponent}胜")
            own = [move for move in game["moves"] if move.get("model") == "F"]
            other = [move for move in game["moves"] if move.get("model") == opponent]
            rows.append({
                "opponent": opponent, "game_id": game["game_id"], "f_color": f_color,
                "result": outcome, "winner_model": game["winner_model"] or "draw",
                "plies": game["move_count"],
                "f_elapsed_seconds": sum(float(move["elapsed_seconds"]) for move in own),
                "opponent_elapsed_seconds": sum(float(move["elapsed_seconds"]) for move in other),
                "opponent_nodes": sum(int(move.get("nodes", 0)) for move in other),
                "opponent_depth_mean": sum(int(move.get("completed_depth", 0)) for move in other) / len(other) if other else 0,
                "opponent_timeouts": sum(bool(move.get("timed_out")) for move in other),
            })
    return summary, rows


def build_report(root: Path) -> Path:
    base = root / "research/f_v1"
    data, results = base / "data", base / "results"
    config = json.loads((base / "config/training.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    preflight = json.loads((results / "preflight.json").read_text(encoding="utf-8"))
    history = _jsonl(base / "checkpoints/training_history.jsonl")
    matches = _jsonl(results / "matches.jsonl")
    integrity = verify_artifacts(root)
    match_summary, game_rows = _match_summary(matches)
    trained = metrics["blind_test"]["trained"]
    baseline = metrics["blind_test"]["baseline_untrained"]
    blind_tactic = metrics["direct_tactics"]["trained"]
    val_tactic = metrics["validation_direct_tactics"]["trained"]

    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(epochs, [row["train_policy_loss_last_batch"] for row in history], label="Train last batch")
    axes[0].plot(epochs, [row["validation"]["policy_loss"] for row in history], label="Validation full set")
    axes[0].set(title="Policy loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0].grid(alpha=0.25); axes[0].legend()
    axes[1].plot(epochs, [row["validation"]["top1_accuracy"] for row in history])
    axes[1].set(title="Validation top-1", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figures / "training_curve.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, title, key in zip(
        axes, ("Blind top-1", "Target probability mass", "Value MAE"),
        ("top1_accuracy", "target_probability_mass", "value_mae"),
    ):
        values = [baseline[key], trained[key]]
        axis.bar(["Untrained", "F V1"], values, color=["#98a6b3", "#277da1"])
        axis.set_title(title); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(figures / "blind_baseline_vs_f.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    kinds = ("win", "block")
    x = range(2)
    width = 0.19
    for offset, (label, scored) in enumerate((("Validation", val_tactic), ("Blind", blind_tactic))):
        axis.bar([i + (offset * 2 - 1.5) * width for i in x],
                 [scored["by_kind"].get(kind, {}).get("accuracy", 0) for kind in kinds],
                 width=width, label=f"{label} top-1")
        axis.bar([i + (offset * 2 + 0.5) * width for i in x],
                 [scored["by_kind"].get(kind, {}).get("top5_accuracy", 0) for kind in kinds],
                 width=width, alpha=0.55, label=f"{label} top-5")
    axis.set_xticks(list(x), ["Immediate win", "Single-point block"])
    axis.set(title="Exact tactical positions", ylabel="Accuracy", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25); axis.legend(ncols=2, fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "tactical_accuracy.png", dpi=160); plt.close(fig)

    analysis = base / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / "match_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(game_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(game_rows)
    with (analysis / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "updates", "train_policy_loss_last_batch", "train_value_loss_last_batch",
                  "validation_policy_loss", "validation_top1", "validation_target_mass", "validation_value_mae", "learning_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": row["epoch"], "updates": row["updates"],
                "train_policy_loss_last_batch": row["train_policy_loss_last_batch"],
                "train_value_loss_last_batch": row["train_value_loss_last_batch"],
                "validation_policy_loss": row["validation"]["policy_loss"],
                "validation_top1": row["validation"]["top1_accuracy"],
                "validation_target_mass": row["validation"]["target_probability_mass"],
                "validation_value_mae": row["validation"]["value_mae"],
                "learning_rate": row["learning_rate"],
            })

    train_metrics = metrics["training"]
    tactic_win = blind_tactic["by_kind"].get("win", {})
    tactic_block = blind_tactic["by_kind"].get("block", {})
    val_win = val_tactic["by_kind"].get("win", {})
    val_block = val_tactic["by_kind"].get("block", {})
    report = f"""# WuziQ F V1：轻量模仿学习预训练报告

## 结论摘要

F V1 已完成监督式策略/价值预训练；没有运行 G 自我对弈强化学习。盲测集策略 top-1 为 **{trained['top1_accuracy']:.1%}**，同结构未训练网络为 **{baseline['top1_accuracy']:.1%}**；价值 MAE 从 **{baseline['value_mae']:.3f}** 降至 **{trained['value_mae']:.3f}**。策略合法率均为 **100%**，CPU 单状态推理中位数约 **{trained['inference_latency_ms']['median']:.3f} ms**。

6 局探索性实战中，F 对 A 为 {match_summary['A']['wins']}胜{match_summary['A']['draws']}平{match_summary['A']['losses']}负，对 B 为 {match_summary['B']['wins']}胜{match_summary['B']['draws']}平{match_summary['B']['losses']}负，对 C 为 {match_summary['C']['wins']}胜{match_summary['C']['draws']}平{match_summary['C']['losses']}负。样本仅用于确认模型具备基础对弈能力，不建立稳定棋力排名。阻挡仍是短板：验证集 54 个单点必防局面 top-1 为 {val_block.get('accuracy', 0):.1%}；盲测仅 4 题，命中 {tactic_block.get('correct', 0)} 题。

## 数据来源与防泄漏

- 历史来源是 V1–V6.3 登记的 24 个正式 9×9 JSONL 文件，不含 15×15、pilot、archive 或 partial。
- 严格规则回放 {manifest['historical']['source_games']:,} 条原始记录；精确序列去重后 {manifest['historical']['unique_games']:,} 局，移除 {manifest['historical']['duplicate_games']:,} 条重复记录。
- 历史中筛出 C 的 {manifest['historical']['policy_decisions']['C']:,} 个决策和 E5 的 {manifest['historical']['policy_decisions']['E5']:,} 个决策。按旋转、镜像、颜色互换和开局连通组 canonicalize 后得到 {manifest['historical']['positions']:,} 个历史独特状态。
- 新增 36 局混合 A/B/C/E5 轨迹；C/E5 以 50 ms 预算生成或补标 {metrics['dataset']['generated_annotations']:,} 条训练标注（C {metrics['dataset']['teacher_annotation_counts'].get('C', 0)}、E5 {metrics['dataset']['teacher_annotation_counts'].get('E5', 0)}）。
- 最终样本：train {manifest['counts']['train']:,}、validation {manifest['counts']['validation']:,}、blind test {manifest['counts']['blind_test']:,}。完整开局连通组件固定切分，且三分区 canonical state 交集均为 0。
- 历史教师搜索预算跨多版本；新增教师标签为 50 ms 行为，因此结果不代表长预算 C/E5 的完整决策能力。

## 网络与训练

输入为己方、对方和合法空位 3 个 9×9 平面；主干为 32 通道、2 个残差块的轻量 CNN。策略头输出 81 个落点，推理前屏蔽所有占用位置；价值头输出 [-1,1] 的当前行棋方视角价值。策略使用合法位屏蔽后的软标签交叉熵，价值使用已验证终局胜/负/平标签的 MSE。

| 设置 | 结果 |
|---|---|
| 设备 | {train_metrics['torch_version']} CPU；无可用 GPU |
| 资源快照 | {preflight['resource_snapshot']['cpu_logical']} 逻辑 CPU；内存总量 {preflight['resource_snapshot']['memory_bytes'].get('MemTotal', 0)//(1024**3)} GiB；磁盘可用约 {preflight['resource_snapshot']['disk_free_bytes']//(1024**3)} GiB |
| 随机种子 | {config['seed']} |
| 轮数 / 更新数 | {train_metrics['epochs_completed']} / {train_metrics['updates']:,} |
| 优化器 / 批量 | AdamW / {config['batch_size']} |
| 最佳轮次 | {train_metrics['best_epoch']} |
| 训练耗时 | {train_metrics['elapsed_training_seconds']:.2f} 秒实际累计墙钟；100 批次基准 {preflight['training_benchmark']['seconds_per_update']:.4f} 秒/更新，{train_metrics['updates']:,} 次更新约 {train_metrics['updates'] * preflight['training_benchmark']['seconds_per_update']:.1f} 秒基准时间，其余用于验证和 checkpoint I/O |

启动前用 {preflight['pilot']['games']} 局、{preflight['pilot']['annotations']} 条标注做采集试跑（{preflight['pilot']['wall_seconds']:.2f} 秒），并测了 {preflight['training_benchmark']['updates']} 次训练更新。根据最多 {preflight['estimate']['maximum_games']} 局和约 {preflight['estimate']['estimated_training_updates']:,} 次更新，预估全流程乐观 / 中性 / 保守耗时分别为 {preflight['estimate']['optimistic_seconds']:.1f} / {preflight['estimate']['central_seconds']:.1f} / {preflight['estimate']['conservative_seconds']:.1f} 秒；实测训练为 {train_metrics['elapsed_training_seconds']:.1f} 秒，另含采集、评估和实战。

训练曲线在 <code>figures/training_curve.png</code>。其中每轮训练损失记录的是最后一批次，不是全训练集平均；验证指标则覆盖完整验证分区。

## 盲测结果

| 指标 | 未训练 | F V1 |
|---|---:|---:|
| blind 样本数 | {baseline['positions']:,} | {trained['positions']:,} |
| 策略 top-1 | {baseline['top1_accuracy']:.1%} | {trained['top1_accuracy']:.1%} |
| 策略交叉熵 | {baseline['policy_loss']:.3f} | {trained['policy_loss']:.3f} |
| 标签概率质量 | {baseline['target_probability_mass']:.1%} | {trained['target_probability_mass']:.1%} |
| 价值 MAE | {baseline['value_mae']:.3f} | {trained['value_mae']:.3f} |
| 合法预测率 | {baseline['legal_rate']:.1%} | {trained['legal_rate']:.1%} |
| 推理 median / p95 | {baseline['inference_latency_ms']['median']:.3f} / {baseline['inference_latency_ms']['p95']:.3f} ms | {trained['inference_latency_ms']['median']:.3f} / {trained['inference_latency_ms']['p95']:.3f} ms |

F 的策略 top-1 是未训练网络的 {trained['top1_accuracy']/max(1e-9, baseline['top1_accuracy']):.1f} 倍；价值误差也明显降低。指标来自完全相同的盲测局面。

## 直接战术题

题目由精确五连规则筛选：若当前方有一步成五先测取胜；否则只评估对手恰有一个直接胜点的单点阻挡。

| 数据分区 / 类型 | 数量 | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|
| Validation 取胜 | {val_win.get('positions', 0)} | {val_win.get('accuracy', 0):.1%} | {val_win.get('top3_accuracy', 0):.1%} | {val_win.get('top5_accuracy', 0):.1%} |
| Validation 必防 | {val_block.get('positions', 0)} | {val_block.get('accuracy', 0):.1%} | {val_block.get('top3_accuracy', 0):.1%} | {val_block.get('top5_accuracy', 0):.1%} |
| Blind 取胜 | {tactic_win.get('positions', 0)} | {tactic_win.get('accuracy', 0):.1%} | {tactic_win.get('top3_accuracy', 0):.1%} | {tactic_win.get('top5_accuracy', 0):.1%} |
| Blind 必防 | {tactic_block.get('positions', 0)} | {tactic_block.get('accuracy', 0):.1%} | {tactic_block.get('top3_accuracy', 0):.1%} | {tactic_block.get('top5_accuracy', 0):.1%} |

盲测 4 个必防题不足以估计稳定阻挡能力。验证集的必防动作有 {val_block.get('top5_correct', 0)}/{val_block.get('positions', 0)} 进入 top-5，说明可作为 G 搜索的候选先验，但 F 单独落子仍可能漏防。

## 换色实战

使用 3 个训练开局组之外的新开局；F 对 A/B/C 各两局，同开局交换黑白。6 局均完成规则回放，开局组与 train 不重叠，中间 canonical 状态也无 train 碰撞。

| 对手 | F 胜 | 平 | F 负 | F 得分率 |
|---|---:|---:|---:|---:|
| A 规则 AI | {match_summary['A']['wins']} | {match_summary['A']['draws']} | {match_summary['A']['losses']} | {match_summary['A']['score_rate']:.1%} |
| B 纯 Minimax 50 ms | {match_summary['B']['wins']} | {match_summary['B']['draws']} | {match_summary['B']['losses']} | {match_summary['B']['score_rate']:.1%} |
| C Alpha-Beta 50 ms | {match_summary['C']['wins']} | {match_summary['C']['draws']} | {match_summary['C']['losses']} | {match_summary['C']['score_rate']:.1%} |

逐局颜色、胜负、落子数、双方耗时、B/C 节点与深度见 <code>analysis/match_results.csv</code> 和 <code>results/matches.jsonl</code>。每组只有两局，不作显著性或稳定强弱判断。

## 模型、续训与 G 阶段接口

冻结 checkpoint：<code>checkpoints/best.pt</code>。训练状态 <code>checkpoints/training_state.pt</code> 原子保存模型、优化器、调度器、Python/NumPy/PyTorch RNG、采样顺序和游标。<code>load_f_checkpoint(path, device="auto")</code> 可加载模型；<code>predict_policy_value(model, board, to_move)</code> 返回合法掩码后的 81 概率与价值，可作为 G 的策略/价值初始化。本轮未执行 G，也没有强化学习训练或 RL checkpoint。

模型 SHA-256：<code>{train_metrics['checkpoint_sha256']}</code>。数据版本标识（拆分清单 SHA-256）：<code>{train_metrics['dataset_version']}</code>。

~~~bash
python -m venv /workspace/.venv-wuziq-f-v1
/workspace/.venv-wuziq-f-v1/bin/pip install -r requirements-f.txt --index-url https://download.pytorch.org/whl/cpu
/workspace/.venv-wuziq-f-v1/bin/python scripts/prepare_f_v1_dataset.py --root .
/workspace/.venv-wuziq-f-v1/bin/python scripts/generate_f_v1_data.py --root .
/workspace/.venv-wuziq-f-v1/bin/python scripts/prepare_f_v1_dataset.py --root . --annotations research/f_v1/results/generated_annotations.jsonl
/workspace/.venv-wuziq-f-v1/bin/python scripts/train_f_v1.py --root .
/workspace/.venv-wuziq-f-v1/bin/python scripts/evaluate_f_v1.py --root .
/workspace/.venv-wuziq-f-v1/bin/python scripts/play_f_v1_matches.py --root .
~~~

## 完整性、限制与下一步

- 自动验收：{integrity['generated_games']} 局新增局谱、{integrity['matches']} 局实战均合法终局；annotation ID 无重复；拆分交集为 0；所有 policy target 归一化且不指向占用格。
- A/B/C/E5 历史对局有不同预算和开局分布；generated C/E5 补标预算为 50 ms，模型只作为轻量起点。
- 后续 G 可从最佳 checkpoint 初始化，但建议先在独立验证集强化单点必防能力，再开始自我对弈；不要将 6 局实战作为正式评级。
- 本轮只做监督模仿预训练和小规模对弈评估，没有自我对弈强化学习，也没有虚构训练 checkpoint。
"""
    report_path = base / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    (base / "integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "integrity": integrity, "match_summary": match_summary}, ensure_ascii=False, indent=2))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build_report(args.root.resolve())


if __name__ == "__main__":
    main()
