# WuziQ F V1：轻量模仿学习预训练报告

## 结论摘要

F V1 已完成监督式策略/价值预训练；没有运行 G 自我对弈强化学习。盲测集策略 top-1 为 **41.6%**，同结构未训练网络为 **6.6%**；价值 MAE 从 **0.395** 降至 **0.085**。策略合法率均为 **100%**，CPU 单状态推理中位数约 **0.309 ms**。

6 局探索性实战中，F 对 A 为 0胜0平2负，对 B 为 1胜0平1负，对 C 为 0胜0平2负。样本仅用于确认模型具备基础对弈能力，不建立稳定棋力排名。阻挡仍是短板：验证集 54 个单点必防局面 top-1 为 44.4%；盲测仅 4 题，命中 1 题。

## 数据来源与防泄漏

- 历史来源是 V1–V6.3 登记的 24 个正式 9×9 JSONL 文件，不含 15×15、pilot、archive 或 partial。
- 严格规则回放 2,809 条原始记录；精确序列去重后 2,264 局，移除 545 条重复记录。
- 历史中筛出 C 的 10,296 个决策和 E5 的 4,540 个决策。按旋转、镜像、颜色互换和开局连通组 canonicalize 后得到 11,404 个历史独特状态。
- 新增 36 局混合 A/B/C/E5 轨迹；C/E5 以 50 ms 预算生成或补标 1,496 条训练标注（C 628、E5 868）。
- 最终样本：train 10,332、validation 1,140、blind test 1,394。完整开局连通组件固定切分，且三分区 canonical state 交集均为 0。
- 历史教师搜索预算跨多版本；新增教师标签为 50 ms 行为，因此结果不代表长预算 C/E5 的完整决策能力。

## 网络与训练

输入为己方、对方和合法空位 3 个 9×9 平面；主干为 32 通道、2 个残差块的轻量 CNN。策略头输出 81 个落点，推理前屏蔽所有占用位置；价值头输出 [-1,1] 的当前行棋方视角价值。策略使用合法位屏蔽后的软标签交叉熵，价值使用已验证终局胜/负/平标签的 MSE。

| 设置 | 结果 |
|---|---|
| 设备 | 2.15.0+cpu CPU；无可用 GPU |
| 资源快照 | 3 逻辑 CPU；内存总量 9 GiB；磁盘可用约 27 GiB |
| 随机种子 | 731501 |
| 轮数 / 更新数 | 20 / 3,240 |
| 优化器 / 批量 | AdamW / 64 |
| 最佳轮次 | 20 |
| 训练耗时 | 76.78 秒实际累计墙钟；100 批次基准 0.0222 秒/更新，3,240 次更新约 72.0 秒基准时间，其余用于验证和 checkpoint I/O |

启动前用 2 局、94 条标注做采集试跑（5.90 秒），并测了 100 次训练更新。根据最多 36 局和约 3,300 次更新，预估全流程乐观 / 中性 / 保守耗时分别为 152.9 / 235.2 / 587.9 秒；实测训练为 76.8 秒，另含采集、评估和实战。

训练曲线在 <code>figures/training_curve.png</code>。其中每轮训练损失记录的是最后一批次，不是全训练集平均；验证指标则覆盖完整验证分区。

## 盲测结果

| 指标 | 未训练 | F V1 |
|---|---:|---:|
| blind 样本数 | 1,394 | 1,394 |
| 策略 top-1 | 6.6% | 41.6% |
| 策略交叉熵 | 3.325 | 1.861 |
| 标签概率质量 | 6.3% | 30.8% |
| 价值 MAE | 0.395 | 0.085 |
| 合法预测率 | 100.0% | 100.0% |
| 推理 median / p95 | 0.294 / 0.424 ms | 0.309 / 0.460 ms |

F 的策略 top-1 是未训练网络的 6.3 倍；价值误差也明显降低。指标来自完全相同的盲测局面。

## 直接战术题

题目由精确五连规则筛选：若当前方有一步成五先测取胜；否则只评估对手恰有一个直接胜点的单点阻挡。

| 数据分区 / 类型 | 数量 | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|
| Validation 取胜 | 22 | 45.5% | 68.2% | 86.4% |
| Validation 必防 | 54 | 44.4% | 75.9% | 83.3% |
| Blind 取胜 | 99 | 96.0% | 99.0% | 100.0% |
| Blind 必防 | 4 | 25.0% | 50.0% | 100.0% |

盲测 4 个必防题不足以估计稳定阻挡能力。验证集的必防动作有 45/54 进入 top-5，说明可作为 G 搜索的候选先验，但 F 单独落子仍可能漏防。

## 换色实战

使用 3 个训练开局组之外的新开局；F 对 A/B/C 各两局，同开局交换黑白。6 局均完成规则回放，开局组与 train 不重叠，中间 canonical 状态也无 train 碰撞。

| 对手 | F 胜 | 平 | F 负 | F 得分率 |
|---|---:|---:|---:|---:|
| A 规则 AI | 0 | 0 | 2 | 0.0% |
| B 纯 Minimax 50 ms | 1 | 0 | 1 | 50.0% |
| C Alpha-Beta 50 ms | 0 | 0 | 2 | 0.0% |

逐局颜色、胜负、落子数、双方耗时、B/C 节点与深度见 <code>analysis/match_results.csv</code> 和 <code>results/matches.jsonl</code>。每组只有两局，不作显著性或稳定强弱判断。

## 模型、续训与 G 阶段接口

冻结 checkpoint：<code>checkpoints/best.pt</code>。训练状态 <code>checkpoints/training_state.pt</code> 原子保存模型、优化器、调度器、Python/NumPy/PyTorch RNG、采样顺序和游标。<code>load_f_checkpoint(path, device="auto")</code> 可加载模型；<code>predict_policy_value(model, board, to_move)</code> 返回合法掩码后的 81 概率与价值，可作为 G 的策略/价值初始化。本轮未执行 G，也没有强化学习训练或 RL checkpoint。

模型 SHA-256：<code>8d3cbeefb3b0ec75a4ec769b9852327f7411e8836f87a4add80d9cff2516e51f</code>。数据版本标识（拆分清单 SHA-256）：<code>4944b5d21bb194192eddecf6a64f4ee3f5767c5039355dd0147b7746871cf5de</code>。

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

- 自动验收：36 局新增局谱、6 局实战均合法终局；annotation ID 无重复；拆分交集为 0；所有 policy target 归一化且不指向占用格。
- A/B/C/E5 历史对局有不同预算和开局分布；generated C/E5 补标预算为 50 ms，模型只作为轻量起点。
- 后续 G 可从最佳 checkpoint 初始化，但建议先在独立验证集强化单点必防能力，再开始自我对弈；不要将 6 局实战作为正式评级。
- 本轮只做监督模仿预训练和小规模对弈评估，没有自我对弈强化学习，也没有虚构训练 checkpoint。
