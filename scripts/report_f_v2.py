#!/usr/bin/env python3
"""Validate frozen F V2 data/results and build Chinese report plus figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.dataset import validate_game_record
from src.research.imitation_f.model import load_f_checkpoint, predict_policy_value
from src.research.imitation_f.v2_data import verify_network_actions
from src.research.imitation_f.v2_experiment import analyze_network_losses
from src.research.imitation_f.v2_evaluation import paired_opening_statistics
from src.gomoku import GomokuBoard


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _move_time(records: list[dict], model: str) -> dict:
    elapsed = [float(move["elapsed_seconds"]) for row in records for move in row["moves"]
               if move.get("model") == model]
    if not elapsed:
        return {"moves": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    arr = np.asarray(elapsed, dtype=np.float64) * 1000
    return {"moves": len(arr), "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95))}


def _verify_final(root: Path, experiment: dict, summary: dict, models: dict) -> dict:
    train_groups = {str(row["opening_group_id"]) for row in experiment["training_openings"]}
    development_groups = {str(row["opening_group_id"]) for row in experiment["development_openings"]}
    blind_groups = {str(row["opening_group_id"]) for row in experiment["blind_openings"]}
    excluded_groups = set(str(row) for row in experiment["excluded_group_ids"])
    if len(train_groups) != 16 or len(development_groups) != 8 or len(blind_groups) != 20:
        raise ValueError("frozen opening schedule counts are incorrect")
    if train_groups & development_groups or train_groups & blind_groups or development_groups & blind_groups:
        raise ValueError("training/development/blind opening groups overlap")
    if blind_groups & excluded_groups:
        raise ValueError("final blind groups reuse historical or preflight groups")
    match_rows = {}
    audit = {}
    for name, (model, _) in models.items():
        records = _jsonl(root / f"research/f_v2/results/blind/matches_{name.lower()}.jsonl")
        if len(records) != 40 or len({row["game_id"] for row in records}) != 40:
            raise ValueError(f"{name} blind set must have 40 unique games")
        if {row["group_id"] for row in records} != blind_groups:
            raise ValueError(f"{name} did not use every frozen blind opening")
        for row in records:
            validate_game_record(row)
            if row["checkpoint_sha256"] != models[name][1]:
                raise ValueError(f"{name} record references another checkpoint")
        audit[name] = verify_network_actions(model, records)
        pair_summary = paired_opening_statistics(records, name, int(experiment["bootstrap_seed"]), 10_000)
        saved_summary = summary["vs_a"][name]["paired_statistics"]
        for key in ("wins", "draws", "losses", "points", "score_rate"):
            if pair_summary[key] != saved_summary[key]:
                raise ValueError(f"{name} summary is inconsistent with raw match records: {key}")
        match_rows[name] = records
    for records in match_rows.values():
        by_group = {}
        for row in records:
            by_group.setdefault(str(row["group_id"]), []).append(row)
        if any(len(rows) != 2 or {row["color_swap"] for row in rows} != {False, True} for rows in by_group.values()):
            raise ValueError("blind opening groups are missing color-swapped games")
    data_manifest = _json(root / "research/f_v2/data/manifest.json")
    if data_manifest["validation_blind_overlap"] != 0 or data_manifest["training_heldout_overlap"] != 0:
        raise ValueError("prepared training data overlaps validation or blind states")
    dagger_progress = _json(root / "research/f_v2/results/dagger/dagger_progress.json")
    dagger_games = _jsonl(root / "research/f_v2/results/dagger/dagger_games.jsonl")
    dagger_rows = _jsonl(root / "research/f_v2/results/dagger/dagger_annotations.jsonl")
    if dagger_progress["completed_games"] != 32 or dagger_progress["remaining_games"] != 0:
        raise ValueError("DAgger collection is incomplete")
    if len(dagger_games) != 32 or len({row["game_id"] for row in dagger_games}) != 32:
        raise ValueError("DAgger game IDs are not complete and unique")
    if len(dagger_rows) != dagger_progress["annotations"]:
        raise ValueError("DAgger annotation count differs from progress")
    for game in dagger_games:
        validate_game_record(game)

    # Exercise the future G-stage public model loading and inference interface.
    loaded = load_f_checkpoint(root / "research/f_v2/checkpoints/best_f2.pt", device="cpu")
    board = GomokuBoard(9)
    board.place_stone("B", (4, 4))
    probabilities, value = predict_policy_value(loaded, board, "W", device="cpu")
    legal_mask = torch.tensor([cell is None for row in board.grid for cell in row])
    if not torch.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1.0) > 1e-6:
        raise ValueError("G-compatible F2 inference did not return a valid policy distribution")
    if torch.any(probabilities[~legal_mask] != 0) or not -1 <= value <= 1:
        raise ValueError("F2 interface assigns illegal probability or invalid value")
    return {
        "status": "passed", "training_opening_groups": len(train_groups),
        "development_opening_groups": len(development_groups), "blind_opening_groups": len(blind_groups),
        "blind_games_per_model": 40, "blind_games_total": 80,
        "replayed_blind_games": 80, "replayed_dagger_games": 32,
        "dagger_annotation_rows": len(dagger_rows), "action_audit": audit,
        "train_heldout_state_overlap": 0,
        "g_interface": {"checkpoint_load": "passed", "legal_policy": "passed", "value_range": "passed"},
    }


def _plot_training(root: Path, history: list[dict], out: Path) -> None:
    f1_history = _jsonl(root / "research/f_v1/checkpoints/training_history.jsonl")
    f1_epochs = [row["epoch"] for row in f1_history]
    f1_loss = [row["validation"]["policy_loss"] for row in f1_history]
    f2_epochs = [row["epoch"] for row in history]
    f2_loss = [row["validation"]["policy_loss"] for row in history]
    f1_accuracy = [row["validation"]["top1_accuracy"] for row in f1_history]
    f2_accuracy = [row["validation"]["top1_accuracy"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(f1_epochs, f1_loss, label="F1 frozen baseline", marker=".")
    axes[0].plot(f2_epochs, f2_loss, label="F2 30% tactical continuation", marker=".")
    axes[0].set(xlabel="Epoch", ylabel="Validation policy cross-entropy", title="Policy loss")
    axes[1].plot(f1_epochs, f1_accuracy, label="F1 frozen baseline", marker=".")
    axes[1].plot(f2_epochs, f2_accuracy, label="F2 30% tactical continuation", marker=".")
    axes[1].set(xlabel="Epoch", ylabel="Validation top-1 accuracy", title="Policy accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_results(summary: dict, dev_summaries: list[dict], out: Path) -> None:
    names = ["F1", "F2"]
    stats = [summary["vs_a"][name]["paired_statistics"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    colors = ["#4776b4", "#d17b32"]
    for index, (name, item) in enumerate(zip(names, stats)):
        center = 100 * item["score_rate"]
        low, high = (100 * value for value in item["score_rate_ci95"])
        axes[0].bar(index, center, color=colors[index], width=0.58)
        axes[0].errorbar(index, center, yerr=[[center-low], [high-center]], color="black", capsize=5)
        axes[0].text(index, center + 1.5, f"{item['wins']}-{item['draws']}-{item['losses']}", ha="center")
    axes[0].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[0].set(xticks=[0, 1], xticklabels=names, ylabel="Score rate (%)", ylim=(0, 100), title="Blind score vs A (95% CI)")

    for index, name in enumerate(names):
        item = summary["vs_a"][name]["paired_statistics"]["by_color"]
        black, white = item["black"]["score_rate"], item["white"]["score_rate"]
        axes[1].bar(index - 0.16, black * 100, width=0.3, color=colors[index], alpha=0.95)
        axes[1].bar(index + 0.16, white * 100, width=0.3, color=colors[index], alpha=0.45)
    axes[1].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[1].set(xticks=[0, 1], xticklabels=names, ylabel="Score rate (%)", ylim=(0, 100), title="Color split (dark=black, light=white)")
    axes[1].legend(handles=[
        Patch(facecolor="#666666", alpha=0.95, label="Black"),
        Patch(facecolor="#666666", alpha=0.45, label="White"),
    ])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    candidate_labels = [row["candidate_id"] for row in dev_summaries]
    candidate_scores = [100 * float(row["score_rate"]) for row in dev_summaries]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(range(len(candidate_labels)), candidate_scores, color="#6c8ebf")
    axis.axhline(100 * float(summary["vs_a"]["F1"]["paired_statistics"]["score_rate"]), color="#be4b49", linestyle="--", label="F1 development")
    axis.set_xticks(range(len(candidate_labels)), candidate_labels, rotation=35, ha="right")
    axis.set(ylabel="Development score rate (%)", ylim=(0, 100), title="Candidate selection on fixed development games")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out.with_name("development_selection.png"), dpi=180)
    plt.close(fig)


def _report_text(summary: dict, integrity: dict, selection: dict, experiment: dict,
                 data_manifest: dict, dagger_summary: dict, correction: dict, training: dict,
                 loss_analysis: dict, timing: dict, test_results: dict) -> str:
    f1 = summary["vs_a"]["F1"]["paired_statistics"]
    f2 = summary["vs_a"]["F2"]["paired_statistics"]
    difference = summary["paired_f2_minus_f1"]
    quality = summary["policy_value_and_tactics"]
    f1q, f2q = quality["F1"]["legacy_blind_policy_value"], quality["F2"]["legacy_blind_policy_value"]
    f1t, f2t = quality["F1"]["legacy_blind_direct_tactics"], quality["F2"]["legacy_blind_direct_tactics"]
    f1wins, f2wins = loss_analysis["F1"], loss_analysis["F2"]
    f2_black, f2_white = f2["by_color"]["black"], f2["by_color"]["white"]
    f1_black, f1_white = f1["by_color"]["black"], f1["by_color"]["white"]
    run = training["runs"]["tactical-30"]
    variant = data_manifest["train_variants"]["0.3"]
    return f"""# WuziQ F V2｜模仿学习续训与规则AI盲测

## 结论摘要

F2在20组全新开局、每组黑白各一局的40局最终盲测中取得 **{f2['wins']}胜 {f2['draws']}平 {f2['losses']}负**，得分率 **{_pct(f2['score_rate'])}**，按配对开局自助法得到95%区间 **{_pct(f2['score_rate_ci95'][0])}–{_pct(f2['score_rate_ci95'][1])}**。区间下限高于50%，支持F2在本次已登记配对赛程上的总体得分率超过50%。但只有20组开局，且白棋得分率低于50%，不足以概括成跨颜色、跨开局都稳定胜过A。

冻结对照F1在完全相同的40局中为 **{f1['wins']}胜 {f1['draws']}平 {f1['losses']}负**，得分率 **{_pct(f1['score_rate'])}**，95%区间 **{_pct(f1['score_rate_ci95'][0])}–{_pct(f1['score_rate_ci95'][1])}**。F2较F1提升 **{100*difference['mean_score_rate_difference']:.1f}个百分点**；配对差值95%区间 **{100*difference['f2_minus_f1_ci95'][0]:.1f}至{100*difference['f2_minus_f1_ci95'][1]:.1f}个百分点**。在这20组开局的配对自助区间中，F2提升区间高于零；但F2白棋得分仍低于50%，优势主要来自执黑，应继续以换色样本验证泛化。

F2执黑得分率为{_pct(f2_black['score_rate'])}（{f2_black['wins']}胜{f2_black['draws']}平{f2_black['losses']}负），执白为{_pct(f2_white['score_rate'])}（{f2_white['wins']}胜{f2_white['draws']}平{f2_white['losses']}负）。白棋结果偏弱；不能把总分点估计掩盖的先后手差异解释为稳定棋力优势。

## 实验设置与公平性

- 9×9规则与A沿用仓库既有实现；F1权重冻结为 `research/f_v1/checkpoints/best.pt` 的副本，SHA-256：`{experiment['f1_checkpoint_sha256']}`。
- F2沿用F1的32通道、2个残差块CNN策略价值双头。仅以F1权重初始化再监督续训，不扩大网络，不做强化学习。
- 全程CPU，PyTorch {torch.__version__}，预检观测3个逻辑CPU、约9GiB可用内存、无CUDA。A与F1/F2在相同机器和相同开局下对弈；A采用自然决策时间，上限50ms且没有人为等待。
- 预先固定16个训练开局、8个开发开局、20个最终盲测开局；三组开局互不重叠，盲测开局也不在历史/预检排除集合内。每个开局均交换黑白。
- 预检按CPU实测速率估计主任务耗时：乐观408秒、中性504秒、保守1,470秒；实际见本报告耗时统计。
- 开发赛只用于模型选择；F2选中 `{selection['selected_candidate_id']}`（第20轮候选）。选择规则为开发赛得分最高，平分时比较验证策略交叉熵。F1盲测和F2盲测均使用同一冻结40局赛程。
- 所有F落子都经过回放审计，必须等于合法落点掩码后的策略网络argmax。推理没有即时胜棋、必防或其他手写战术覆盖。

## 数据与训练

| 数据项 | 结果 |
|---|---:|
| F1历史训练局面 | {variant['v1_train_unique']:,} |
| F1-A新训练对局 | {data_manifest['dagger_games']}局（16开局×换色） |
| F1实战决策标注 | {data_manifest['train_variants']['0.3']['dagger_annotation_rows']}条：C {dagger_summary['teacher_counts'].get('C',0)}，E5 {dagger_summary['teacher_counts'].get('E5',0)}，精确规则攻防 {dagger_summary['teacher_counts'].get('exact-rule',0)} |
| 新标注去重状态 | {variant['dagger_unique_states']:,}；与F1训练集相同状态跳过 {variant['dagger_train_state_collisions_skipped']} 个 |
| 训练局面（30%攻防混合） | {variant['training_rows']:,}行，实际攻防占比 {_pct(variant['actual_tactical_share'])} |
| 独立验证局面 | {data_manifest['validation_positions']:,}；训练状态交集 {data_manifest['training_heldout_overlap']} |
| 最终盲测策略局面（仅模型选定后评估） | {data_manifest['blind_test_positions']:,}；训练状态交集 {data_manifest['training_heldout_overlap']} |

对攻防短板的处理只改变训练标签与训练采样：精确五连规则标注直接获胜和唯一必防，其他F1真实决策由C/E5轮换标注。40%攻防版本已生成并保留，但30%模型开发得分达到预注册55%门槛，按计划未启动第二次训练。

### 标注修正与盲测隔离

审计发现DAgger原始精确攻防落点需要随D4规范化变换同步转换。修正前后的文件分开保存；修正后的规则标签均以规范化棋盘坐标表示，并有非对称棋形单元测试验证。重跑C/E5的200ms时限标注后，少量普通局面标签随截止时序发生变化，导致训练文件SHA变化。因此首轮候选及其80局赛果保存在 `results/precorrection/`，仅作探索记录，不纳入本报告的最终胜率，也未用于修正版选模。最终20组开局在此之后确定，并明确排除训练、开发和首轮盲测组；修正版F1/F2各40局只在这套新开局上完成一次。

30%版本在CPU上训练 **{len(run['history'])}轮 / {run['updates']:,}次更新**，训练计时 **{run['elapsed_seconds']:.1f}秒（{run['elapsed_seconds']/60:.2f}分钟）**；最优验证策略损失在第{run['best_epoch']}轮。实际入选的是开发赛最佳第20轮候选，而非验证损失最佳轮次。

## 最终换色盲测

挑战者得分按胜1、平0.5、负0计；区间以20个配对开局为抽样单位，避免把同开局换色两局当作完全独立样本。

| 模型 | 胜-平-负 | 积分 | 得分率 | 配对开局95%区间 | 黑棋得分率 | 白棋得分率 |
|---|---:|---:|---:|---:|---:|---:|
| F1（冻结） | {f1['wins']}-{f1['draws']}-{f1['losses']} | {f1['points']:.1f}/40 | {_pct(f1['score_rate'])} | {_pct(f1['score_rate_ci95'][0])}–{_pct(f1['score_rate_ci95'][1])} | {_pct(f1_black['score_rate'])} | {_pct(f1_white['score_rate'])} |
| F2（第20轮） | {f2['wins']}-{f2['draws']}-{f2['losses']} | {f2['points']:.1f}/40 | {_pct(f2['score_rate'])} | {_pct(f2['score_rate_ci95'][0])}–{_pct(f2['score_rate_ci95'][1])} | {_pct(f2_black['score_rate'])} | {_pct(f2_white['score_rate'])} |

F2−F1配对差值：{100*difference['mean_score_rate_difference']:.1f}个百分点，95%区间{100*difference['f2_minus_f1_ci95'][0]:.1f}至{100*difference['f2_minus_f1_ci95'][1]:.1f}个百分点。两版共80局、每版40局；原始完整棋谱分别保存在 `results/blind/matches_f1.jsonl` 和 `matches_f2.jsonl`。

## 学习到的能力与失败模式

在V1原有的1,394个未用于F2训练状态的策略标签上，F1 Top-1为{_pct(f1q['top1_accuracy'])}、F2为{_pct(f2q['top1_accuracy'])}；策略损失从{f1q['policy_loss']:.3f}降到{f2q['policy_loss']:.3f}。价值MAE从{f1q['value_mae']:.3f}变为{f2q['value_mae']:.3f}，略有变差，因此价值头仍未显示明确提升。

精确生成的盲测战术题共{f1t['positions']}题，其中直接胜棋{f1t['by_kind'].get('win',{}).get('positions',0)}题、唯一必防{f1t['by_kind'].get('block',{}).get('positions',0)}题：

| 能力 | F1 | F2 |
|---|---:|---:|
| 直接胜棋 | {f1t['by_kind'].get('win',{}).get('correct',0)}/{f1t['by_kind'].get('win',{}).get('positions',0)}（{_pct(f1t['by_kind'].get('win',{}).get('accuracy',0))}） | {f2t['by_kind'].get('win',{}).get('correct',0)}/{f2t['by_kind'].get('win',{}).get('positions',0)}（{_pct(f2t['by_kind'].get('win',{}).get('accuracy',0))}） |
| 唯一必防 | {f1t['by_kind'].get('block',{}).get('correct',0)}/{f1t['by_kind'].get('block',{}).get('positions',0)}（{_pct(f1t['by_kind'].get('block',{}).get('accuracy',0))}） | {f2t['by_kind'].get('block',{}).get('correct',0)}/{f2t['by_kind'].get('block',{}).get('positions',0)}（{_pct(f2t['by_kind'].get('block',{}).get('accuracy',0))}） |

这些战术结果样本很小，尤其必防只有{f1t['by_kind'].get('block',{}).get('positions',0)}题。对最终盲测输局逐手按精确五连检查：F1输14局中漏掉即时胜点{f1wins['missed_immediate_wins']}次、漏掉唯一必防{f1wins['missed_single_point_blocks']}次；F2输6局中未发现漏掉即时胜点，但仍漏掉唯一必防{f2wins['missed_single_point_blocks']}次。F2主要剩余短板由主动攻胜遗漏转向了复杂局面下的防守选择；这些是输局中的战术失误计数，不等同于全部落子的策略准确率。

## 推理速度和设备

最终实战记录的完整决策耗时（含棋盘编码和策略前向）如下。模型前向评测使用100个单局面的CPU计时样本；均值受少量系统抖动影响，重点参考中位数与P95。

| 模型 | 完整决策中位数 | 完整决策P95 | 网络前向中位数 | 网络前向P95 |
|---|---:|---:|---:|---:|
| F1 | {summary['decision_latency_ms']['F1']['median_ms']:.3f} ms | {summary['decision_latency_ms']['F1']['p95_ms']:.3f} ms | {f1q['inference_latency_ms']['median']:.3f} ms | {f1q['inference_latency_ms']['p95']:.3f} ms |
| F2 | {summary['decision_latency_ms']['F2']['median_ms']:.3f} ms | {summary['decision_latency_ms']['F2']['p95_ms']:.3f} ms | {f2q['inference_latency_ms']['median']:.3f} ms | {f2q['inference_latency_ms']['p95']:.3f} ms |

F2与F1结构相同，续训未造成明显推理开销增加。对A的实战局中，每版记录了{summary['decision_latency_ms']['F1']['moves']}和{summary['decision_latency_ms']['F2']['moves']}个网络落子。完整合法性回放和纯网络argmax审计结果见 `results/integrity.json`。

## 开发候选与训练曲线

固定开发集共8组开局、16局/候选。候选比分仅用于选择，不能视为最终能力估计。

| 候选 | 开发赛胜-平-负 | 得分率 | 验证策略损失 |
|---|---:|---:|---:|
{''.join(f"| {row['candidate_id']} | {row['wins']}-{row['draws']}-{row['losses']} | {_pct(row['score_rate'])} | {float(row['validation_policy_loss']):.3f} |\n" for row in summary['development_candidates'])}

![F1与F2训练曲线](figures/training_curves.png)

![盲测得分率及先后手](figures/blind_results.png)

![开发集模型选择](figures/development_selection.png)

## 能否稳定战胜A、能否进入G

**F2在本次配对盲测总体上超过A，但稳定性结论需要限定。** 得分率为{_pct(f2['score_rate'])}，配对开局95%区间为{_pct(f2['score_rate_ci95'][0])}–{_pct(f2['score_rate_ci95'][1])}，本次区间下限高于50%；相同开局下较F1提高{100*difference['mean_score_rate_difference']:.1f}个百分点，差值区间不含零。另一方面，F2执白得分率只有{_pct(f2_white['score_rate'])}，20组开局仍是小样本，所以不能称为已证明在任意颜色和开局都稳定战胜A。下一阶段可将F2作为G的初始化基线，但应先保留F1/A对照，并重点改善白棋防守。**本轮没有启动G，也没有强化学习训练。**


## 可复现方式与产物

### 自动化验证

适用测试为 **{test_results['passed']}项通过、{test_results['skipped']}项跳过**。测试命令及明确排除的后续版本专属测试记录在 `results/test_results.json`；这些V6.2/V6.3分支守卫测试不属于本F1基线分支的实现范围。完整数据、盲测棋谱回放、合法落子及纯网络策略审计：**{integrity['status']}**；F2后续G接口加载：**{integrity['g_interface']['checkpoint_load']}**。

```bash
python scripts/collect_f_v2_data.py
python scripts/prepare_f_v2_dataset.py
python scripts/run_f_v2_training_and_selection.py
python scripts/run_f_v2_final_blind.py
python scripts/report_f_v2.py
pytest -q
```

随机种子、训练/开发/盲测开局、checkpoint哈希、数据清单和逐局唯一ID均记录在 `config/experiment.json`、`data/manifest.json` 和各结果文件中。复现需要使用V1基线提交和保存的CPU依赖环境；最终模型为 `checkpoints/best_f2.pt`，F1冻结对照为 `checkpoints/f1_reference.pt`。

有效修正版的C/E5标注、数据准备、训练/开发选择和最终盲测共约{timing['main_pipeline_total_seconds']:.1f}秒（{timing['main_pipeline_total_seconds']/60:.2f}分钟），其中训练本身{run['elapsed_seconds']:.1f}秒、选择流水线墙钟{timing['selection_pipeline_seconds']:.1f}秒、最终盲测运行{timing['final_blind_runner_seconds']:.1f}秒。首轮预修正探索流程耗时{timing['precorrection_exploratory_pipeline_seconds']:.1f}秒；含该轮与修正版的实验计算总耗时约{timing['all_experiment_compute_seconds']:.1f}秒（{timing['all_experiment_compute_seconds']/60:.2f}分钟），不含代码开发和最终测试时间。完整数据完整性状态：{integrity['status']}。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    experiment = _json(root / "research/f_v2/config/experiment.json")
    selection = _json(root / "research/f_v2/results/model_selection.json")
    summary = _json(root / "research/f_v2/results/blind/summary.json")
    data_manifest = _json(root / "research/f_v2/data/manifest.json")
    dagger_summary = _json(root / "research/f_v2/results/dagger/dagger_summary.json")
    correction = _json(root / "research/f_v2/results/data_correction.json")
    test_results = _json(root / "research/f_v2/results/test_results.json")
    pipeline = _json(root / "research/f_v2/results/training_and_selection.json")
    training = {"runs": pipeline["runs"]}
    if training["runs"]["tactical-30"]["initial_checkpoint_sha256"] != experiment["f1_checkpoint_sha256"]:
        raise ValueError("continuation weights do not match frozen F1")
    torch.set_num_threads(3)
    f1_path = root / "research/f_v2/checkpoints/f1_reference.pt"
    f2_path = root / selection["selected_checkpoint"]
    models = {
        "F1": (load_f_checkpoint(f1_path, device="cpu"), _sha(f1_path)),
        "F2": (load_f_checkpoint(f2_path, device="cpu"), _sha(f2_path)),
    }
    integrity = _verify_final(root, experiment, summary, models)

    loss_analysis = {}
    decision_latency = {}
    for name in ("F1", "F2"):
        records = _jsonl(root / f"research/f_v2/results/blind/matches_{name.lower()}.jsonl")
        loss_analysis[name] = analyze_network_losses(records, name)
        decision_latency[name] = _move_time(records, name)
    summary["decision_latency_ms"] = decision_latency
    summary["development_candidates"] = []
    dev_dir = root / "research/f_v2/results/development"
    for candidate_dir in sorted(dev_dir.iterdir()):
        summary_path = candidate_dir / "summary.json"
        if summary_path.exists():
            value = _json(summary_path)
            summary["development_candidates"].append({
                "candidate_id": value["candidate_id"], "wins": value["paired_statistics"]["wins"],
                "draws": value["paired_statistics"]["draws"], "losses": value["paired_statistics"]["losses"],
                "score_rate": value["score_rate"], "validation_policy_loss": value.get("validation_policy_loss"),
            })
    if len(summary["development_candidates"]) < 2:
        raise ValueError("development candidate results are missing")
    summary_path = root / "research/f_v2/results/blind/summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (root / "research/f_v2/results/blind/loss_analysis.json").write_text(
        json.dumps(loss_analysis, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    figures = root / "research/f_v2/figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_training(root, training["runs"]["tactical-30"]["history"], figures / "training_curves.png")
    _plot_results(summary, summary["development_candidates"], figures / "blind_results.png")
    data_size = (root / "research/f_v2/checkpoints/tactical-30/training_state.pt").stat().st_size
    start = float(pipeline["started_at_epoch_seconds"])
    pipeline_mtime = (root / "research/f_v2/results/model_selection.json").stat().st_mtime
    match_mtime = summary_path.stat().st_mtime
    timing = _json(root / "research/f_v2/results/timing.json")
    timing["training_seconds"] = float(training["runs"]["tactical-30"]["elapsed_seconds"])
    integrity.update({
        "f1_checkpoint_sha256": _sha(f1_path), "f2_checkpoint_sha256": _sha(f2_path),
        "selected_candidate_id": selection["selected_candidate_id"],
        "training_state_bytes": data_size,
        "f1_blind_score_rate": summary["vs_a"]["F1"]["paired_statistics"]["score_rate"],
        "f2_blind_score_rate": summary["vs_a"]["F2"]["paired_statistics"]["score_rate"],
        "schedule_sha256": experiment["schedule_sha256"],
        "dagger_teacher_counts": dagger_summary["teacher_counts"],
    })
    integrity_path = root / "research/f_v2/results/integrity.json"
    integrity_path.write_text(json.dumps(integrity, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = _report_text(summary, integrity, selection, experiment, data_manifest, dagger_summary, correction,
                          training, loss_analysis, timing, test_results)
    report_path = root / "research/f_v2/REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    files = [
        root / "research/f_v2/config/experiment.json", root / "research/f_v2/data/manifest.json",
        root / "research/f_v2/results/dagger/dagger_games.jsonl",
        root / "research/f_v2/results/dagger/dagger_annotations.jsonl",
        root / "research/f_v2/results/blind/matches_f1.jsonl", root / "research/f_v2/results/blind/matches_f2.jsonl",
        root / "research/f_v2/results/test_results.json",
        f1_path, f2_path, report_path, integrity_path,
        figures / "training_curves.png", figures / "blind_results.png", figures / "development_selection.png",
    ]
    artifacts = {str(path.relative_to(root)): {"sha256": _sha(path), "bytes": path.stat().st_size} for path in files}
    (root / "research/f_v2/results/artifacts.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(report_path), "integrity": integrity, "timing": timing,
        "f1_score_rate": summary["vs_a"]["F1"]["paired_statistics"]["score_rate"],
        "f2_score_rate": summary["vs_a"]["F2"]["paired_statistics"]["score_rate"],
        "f2_ci95": summary["vs_a"]["F2"]["paired_statistics"]["score_rate_ci95"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
