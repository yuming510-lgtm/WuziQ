# V6 E家族高级棋类搜索算法开发与实战评测

## 摘要

本研究在9×9五子棋中新增E1–E5五类非学习型棋手。共有1018局完整且通过回放检查的实际对局，其中V5.1精确复用146局，本轮新增872局；赛程最高上限为2856局。另有911组由预注册的更高预算单调性假设推断，未当作实测胜负。累计记录耗时约38.34小时。

V6模型及预算版本的排序按实际出现的对局平均得分率计算，样本覆盖不同且赛制为逐档早停，因此仅为探索性描述，不能视为完整循环赛或统计显著性排名。A为自然耗时版本；B/C/D/E1–E5各有7档，合计22个ABCD版本和35个E版本。

## 算法与正确性验收

- E1 VCF：对连续冲四强制获胜路线进行OR/AND搜索；不能证明时调用统一A后备。
- E2 VCT：在VCF基础上纳入活三和连续威胁，防守节点保留全部合法关键回应。
- E3 Threat-Space Search：用威胁增益、成本、防守冲突及依赖关系组织搜索；启发式排序不裁掉合法防守。
- E4 Proof-Number Search：以攻方OR/守方AND证明树搜索必胜路线，证明在落子前独立回放证书。
- E5：C上的PVS、置换表、Zobrist哈希、排序、战术延伸和根节点即时胜/必防；E5组件消融单独保存。
固定战术题集包括12个局面、60次决策；43个可评分oracle项目中正确43个，错误证明0，证书独立核验17条；合法落子0、棋盘恢复失败0。超时13次均作为受限搜索记载。
E5消融包含18次运行；结果保留于validation/e5_ablation.json与CSV，不把尚未完成深度的结果混入完成深度比较。

## E5组件消融

| 变体 | 局面数 | 平均完成深度 | 深度3局面数 | 平均节点 | 平均秒数 | 超时数 | TT命中 | PVS重搜 | 战术延伸节点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E5_base | 3 | 2.67 | 2 | 2552.0 | 1.672 | 1 | 0 | 0 | 0 |
| E5_PVS_only | 3 | 2.33 | 1 | 2525.0 | 1.705 | 2 | 0 | 42 | 0 |
| E5_TT_only | 3 | 3.00 | 3 | 2613.7 | 1.555 | 0 | 0 | 0 | 0 |
| E5_ordering_only | 3 | 2.67 | 2 | 2644.7 | 1.645 | 1 | 0 | 0 | 0 |
| E5_extensions_only | 3 | 2.33 | 1 | 2022.0 | 1.847 | 2 | 0 | 0 | 0 |
| E5_all | 3 | 2.33 | 1 | 1807.7 | 1.680 | 2 | 152 | 22 | 0 |

消融使用3个固定局面、每档2.0秒和深度上限3。各组件独立开关及全组合均保留逐局结果；样本很小，且不同变体的截止时间会导致完成深度不同，表中节点/耗时只能描述本题集，不作普遍因果结论。战术延伸实际触发数为0；若为零，代表本组局面没有满足延伸触发条件。

## 实验方法与复现

- 固定V6配置、代码SHA-256、12个开局种子、9×9规则与唯一对局ID；每组同开局换色两局。A自然决策，其他版本按50ms至50s七档截止时间。
- 每完整对局先回放，再逐局写入JSONL；每组结束后原子更新路线状态。断点续跑按唯一ID去重。复用条件只有V5.1中条件完全一致且完整成对的对局。
- 双局得分为胜1、平0.5、负0；路线累计积分由wins + draws/2计算。95%区间使用每局分数作为分数成功的Wilson描述区间，n=2样本很小。
- 只有实测和精确复用的完整配对写入数值矩阵；推断和未运行版本在状态矩阵中单独标记，数值留空。
- 无神经网络或强化学习训练，无训练指标或checkpoint。Codex模型档位和推理强度无法由本运行环境独立读取，故不作验证性声明。

## 探索性算法家族排序

| 探索性排名 | 家族 | 实际游戏样本 | 得分率（95% Wilson） | 覆盖预算版本数 |
|---:|---|---:|---:|---:|
| 1 | C | 202 | 69.8% [63.1%, 75.7%] | 7 |
| 2 | E5 | 208 | 59.4% [52.6%, 65.8%] | 7 |
| 3 | B | 200 | 59.2% [52.3%, 65.8%] | 7 |
| 4 | E2 | 264 | 50.9% [44.9%, 56.9%] | 7 |
| 5 | E1 | 262 | 47.9% [41.9%, 53.9%] | 7 |
| 6 | E3 | 292 | 45.7% [40.1%, 51.5%] | 7 |
| 7 | E4 | 306 | 44.6% [39.1%, 50.2%] | 7 |
| 8 | A | 88 | 44.3% [34.4%, 54.7%] | 1 |
| 9 | D | 214 | 30.8% [25.0%, 37.3%] | 7 |

E家族的探索性顺序为E5 → E2 → E1 → E3 → E4。按所有预算版本汇总，C为69.8%，E5为59.4%，B为59.2%，D为30.8%。预算版本和对手覆盖因路线早停而不同，这些家族总分只作描述，不是平衡条件下的能力估计。

实际E5与C的跨预算对局共18组、36局，E5为16胜–6平–14负，得分率52.8%；E5与D共13组、26局，E5为15胜–5平–6负，得分率67.3%。这些汇总混合不同预算。

同为50s预算的E5@50s与C@50s实测2局，E5以2胜、0平、0负通过该单一开局组（开局种子731112）。样本仅一组换色赛，不能据此认定稳定领先。

### 家族对阵汇总（只含实测组）

| 家族对阵（表中前者） | 实测组/局 | 胜–平–负 | 得分率 |
|---|---:|---:|---:|
| A 对 B | 7/14 | 0–8–6 | 28.6% |
| A 对 C | 6/12 | 0–3–9 | 12.5% |
| A 对 D | 5/10 | 7–1–2 | 75.0% |
| A 对 E1 | 5/10 | 1–8–1 | 50.0% |
| A 对 E2 | 7/14 | 1–13–0 | 53.6% |
| A 对 E3 | 5/10 | 0–9–1 | 45.0% |
| A 对 E4 | 7/14 | 1–13–0 | 53.6% |
| A 对 E5 | 2/4 | 0–3–1 | 37.5% |
| B 对 C | 21/42 | 10–5–27 | 29.8% |
| B 对 D | 17/34 | 28–1–5 | 83.8% |
| B 对 E1 | 9/18 | 8–8–2 | 66.7% |
| B 对 E2 | 15/30 | 13–8–9 | 56.7% |
| B 对 E3 | 8/16 | 9–6–1 | 75.0% |
| B 对 E4 | 9/18 | 13–3–2 | 80.6% |
| B 对 E5 | 14/28 | 10–4–14 | 42.9% |
| C 对 D | 17/34 | 28–2–4 | 85.3% |
| C 对 E1 | 11/22 | 11–8–3 | 68.2% |
| C 对 E2 | 11/22 | 13–5–4 | 70.5% |
| C 对 E3 | 9/18 | 9–8–1 | 72.2% |
| C 对 E4 | 8/16 | 8–7–1 | 71.9% |
| C 对 E5 | 18/36 | 14–6–16 | 47.2% |
| D 对 E1 | 14/28 | 9–5–14 | 41.1% |
| D 对 E2 | 12/24 | 5–5–14 | 31.2% |
| D 对 E3 | 13/26 | 4–13–9 | 40.4% |
| D 对 E4 | 16/32 | 9–12–11 | 46.9% |
| D 对 E5 | 13/26 | 6–5–15 | 32.7% |
| E1 对 E2 | 25/50 | 2–41–7 | 45.0% |
| E1 对 E3 | 25/50 | 6–43–1 | 55.0% |
| E1 对 E4 | 27/54 | 5–49–0 | 54.6% |
| E1 对 E5 | 15/30 | 4–15–11 | 38.3% |
| E2 对 E3 | 28/56 | 5–49–2 | 52.7% |
| E2 对 E4 | 25/50 | 8–41–1 | 57.0% |
| E2 对 E5 | 9/18 | 3–7–8 | 36.1% |
| E3 对 E4 | 43/86 | 1–83–2 | 49.4% |
| E3 对 E5 | 15/30 | 1–22–7 | 40.0% |
| E4 对 E5 | 18/36 | 3–25–8 | 43.1% |

## 探索性版本排序

| 排名 | 模型预算版本 | 对局数 | 胜–平–负 | 得分率（95% Wilson） | 平均每步秒数 | 平均节点 | 平均深度 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | E5@50s | 8 | 6–1–1 | 81.2% [46.7%, 95.5%] | 36.867 | 34244.1 | 4.14 |
| 2 | E5@20s | 12 | 9–1–2 | 79.2% [50.9%, 93.3%] | 14.072 | 19190.0 | 4.01 |
| 3 | C@10s | 18 | 12–3–3 | 75.0% [51.9%, 89.3%] | 9.412 | 66813.6 | 5.16 |
| 4 | C@20s | 12 | 7–4–1 | 75.0% [46.8%, 91.1%] | 18.209 | 144730.3 | 5.54 |
| 5 | E5@1s | 60 | 32–23–5 | 72.5% [60.1%, 82.2%] | 0.879 | 1217.0 | 3.76 |
| 6 | C@5s | 18 | 12–2–4 | 72.2% [49.1%, 87.5%] | 4.804 | 30630.9 | 4.36 |
| 7 | B@1s | 38 | 21–12–5 | 71.1% [55.2%, 83.0%] | 0.957 | 11739.3 | 2.26 |
| 8 | C@50ms | 72 | 40–20–12 | 69.4% [58.0%, 78.9%] | 0.044 | 306.5 | 1.77 |
| 9 | C@1s | 24 | 16–1–7 | 68.8% [48.8%, 83.6%] | 0.982 | 7759.2 | 3.58 |
| 10 | C@200ms | 52 | 28–14–10 | 67.3% [53.8%, 78.5%] | 0.188 | 1451.5 | 2.36 |
| 11 | C@50s | 6 | 4–0–2 | 66.7% [30.0%, 90.3%] | 47.963 | 363095.2 | 8.00 |
| 12 | B@200ms | 38 | 19–9–10 | 61.8% [46.0%, 75.5%] | 0.190 | 2469.6 | 1.63 |
| 13 | B@5s | 22 | 10–7–5 | 61.4% [40.8%, 78.5%] | 4.754 | 59746.2 | 2.47 |
| 14 | B@50s | 10 | 5–2–3 | 60.0% [31.3%, 83.2%] | 47.145 | 607640.2 | 3.29 |
| 15 | E5@5s | 16 | 8–3–5 | 59.4% [35.9%, 79.2%] | 4.193 | 5815.1 | 4.07 |
| 16 | E2@50ms | 70 | 19–45–6 | 59.3% [47.6%, 70.0%] | 0.019 | 13.9 | 2.14 |
| 17 | E1@50ms | 64 | 16–41–7 | 57.0% [44.8%, 68.4%] | 0.001 | 1.3 | 0.06 |
| 18 | E5@10s | 22 | 10–5–7 | 56.8% [36.7%, 74.9%] | 7.834 | 11036.8 | 4.91 |
| 19 | D@50s | 50 | 19–16–15 | 54.0% [40.4%, 67.0%] | 44.808 | 784332.8 | 3.34 |
| 20 | B@50ms | 64 | 29–10–25 | 53.1% [41.1%, 64.8%] | 0.044 | 589.7 | 1.17 |
| 21 | B@20s | 16 | 8–1–7 | 53.1% [30.6%, 74.5%] | 19.668 | 262585.1 | 3.08 |
| 22 | E2@10s | 36 | 10–18–8 | 52.8% [37.0%, 68.0%] | 3.494 | 2799.4 | 3.49 |
| 23 | E2@200ms | 48 | 8–34–6 | 52.1% [38.3%, 65.5%] | 0.073 | 57.8 | 3.15 |
| 24 | E3@50ms | 54 | 8–40–6 | 51.9% [38.9%, 64.6%] | 0.021 | 4.2 | 1.37 |
| 25 | D@20s | 32 | 10–13–9 | 51.6% [35.0%, 67.8%] | 18.498 | 427374.8 | 3.20 |
| 26 | E5@50ms | 56 | 6–45–5 | 50.9% [38.2%, 63.5%] | 0.039 | 52.9 | 1.49 |
| 27 | B@10s | 12 | 5–2–5 | 50.0% [25.4%, 74.6%] | 9.589 | 122976.9 | 2.72 |
| 28 | E1@5s | 36 | 4–27–5 | 48.6% [33.2%, 64.3%] | 0.001 | 1.6 | 0.08 |
| 29 | E4@50ms | 60 | 5–48–7 | 48.3% [36.2%, 60.7%] | 0.021 | 22.4 | 1.09 |
| 30 | E4@200ms | 54 | 2–47–5 | 47.2% [34.5%, 60.3%] | 0.076 | 68.8 | 1.38 |
| 31 | E3@5s | 52 | 4–41–7 | 47.1% [34.2%, 60.4%] | 1.947 | 262.2 | 3.48 |
| 32 | E2@5s | 32 | 3–24–5 | 46.9% [30.9%, 63.6%] | 1.654 | 1402.0 | 3.36 |
| 33 | E1@1s | 46 | 5–33–8 | 46.7% [33.1%, 60.8%] | 0.001 | 1.1 | 0.05 |
| 34 | E2@1s | 40 | 6–25–9 | 46.2% [31.8%, 61.3%] | 0.355 | 277.3 | 3.36 |
| 35 | E3@50s | 40 | 1–35–4 | 46.2% [31.8%, 61.3%] | 16.452 | 1969.8 | 3.34 |
| 36 | E1@200ms | 38 | 5–25–8 | 46.1% [31.3%, 61.5%] | 0.001 | 1.2 | 0.06 |
| 37 | E3@1s | 48 | 1–42–5 | 45.8% [32.6%, 59.7%] | 0.386 | 60.7 | 3.06 |
| 38 | E1@10s | 34 | 3–25–6 | 45.6% [30.2%, 61.9%] | 0.001 | 1.3 | 0.07 |
| 39 | E4@20s | 34 | 3–25–6 | 45.6% [30.2%, 61.9%] | 1.343 | 1063.6 | 2.74 |
| 40 | E4@50s | 50 | 4–37–9 | 45.0% [32.1%, 58.7%] | 1.268 | 1047.1 | 2.68 |
| 41 | E2@50s | 18 | 2–12–4 | 44.4% [24.6%, 66.3%] | 13.382 | 10177.7 | 3.47 |
| 42 | A@natural | 88 | 10–58–20 | 44.3% [34.4%, 54.7%] | 0.000 | 42.1 | 0.00 |
| 43 | D@10s | 38 | 12–9–17 | 43.4% [29.0%, 59.1%] | 8.957 | 196893.1 | 2.86 |
| 44 | E4@10s | 38 | 4–25–9 | 43.4% [29.0%, 59.1%] | 1.414 | 1222.3 | 2.83 |
| 45 | E3@20s | 30 | 1–24–5 | 43.3% [27.4%, 60.8%] | 6.946 | 810.4 | 3.24 |
| 46 | E1@20s | 26 | 4–14–8 | 42.3% [25.5%, 61.1%] | 0.001 | 1.1 | 0.02 |
| 47 | E3@10s | 32 | 1–24–7 | 40.6% [25.5%, 57.7%] | 3.547 | 426.6 | 3.20 |
| 48 | E3@200ms | 36 | 1–27–8 | 40.3% [26.0%, 56.5%] | 0.077 | 12.9 | 2.32 |
| 49 | E5@200ms | 34 | 9–9–16 | 39.7% [25.1%, 56.4%] | 0.166 | 222.5 | 2.25 |
| 50 | E4@1s | 38 | 2–26–10 | 39.5% [25.6%, 55.3%] | 0.357 | 281.4 | 1.96 |
| 51 | E4@5s | 32 | 0–25–7 | 39.1% [24.2%, 56.3%] | 1.177 | 952.0 | 2.58 |
| 52 | E2@20s | 20 | 2–11–7 | 37.5% [20.0%, 59.1%] | 7.091 | 5616.4 | 3.66 |
| 53 | E1@50s | 18 | 0–12–6 | 33.3% [16.3%, 56.3%] | 0.002 | 3.6 | 0.10 |
| 54 | D@5s | 20 | 2–4–14 | 20.0% [8.1%, 41.6%] | 4.554 | 104339.0 | 2.57 |
| 55 | D@1s | 16 | 0–2–14 | 6.2% [1.1%, 28.3%] | 0.901 | 19117.2 | 2.13 |
| 56 | D@200ms | 18 | 1–0–17 | 5.6% [1.0%, 25.8%] | 0.178 | 1823.5 | 1.79 |
| 57 | D@50ms | 40 | 0–0–40 | 0.0% [0.0%, 8.8%] | 0.047 | 458.2 | 1.90 |

得分率区间跨越很宽，且各预算版本的参赛对手数因早停而不同。平均得分率不能消除赛程覆盖偏差；请结合逐组矩阵和挑战路线阅读。

## 挑战路线

| 挑战者版本 | 对手家族 | 最高实测通过档 | 失败档 | inferred pass数 | 路线状态 |
|---|---|---|---|---:|---|
| A@natural | B | 无 | 50ms | 0 | failed |
| B@50ms | A | natural | 无 | 0 | completed |
| B@200ms | A | natural | 无 | 0 | completed |
| B@1s | A | natural | 无 | 0 | completed |
| B@5s | A | natural | 无 | 0 | completed |
| B@10s | A | natural | 无 | 0 | completed |
| B@20s | A | natural | 无 | 0 | completed |
| B@50s | A | natural | 无 | 0 | completed |
| A@natural | C | 无 | 50ms | 0 | failed |
| C@50ms | A | natural | 无 | 0 | completed |
| C@200ms | A | natural | 无 | 0 | completed |
| C@1s | A | natural | 无 | 0 | completed |
| C@5s | A | natural | 无 | 0 | completed |
| C@10s | A | natural | 无 | 0 | completed |
| C@20s | A | natural | 无 | 0 | completed |
| C@50s | A | 无 | 无 | 1 | completed |
| A@natural | D | 5s | 10s | 0 | failed |
| D@50ms | A | 无 | natural | 0 | failed |
| D@200ms | A | 无 | natural | 0 | failed |
| D@1s | A | 无 | natural | 0 | failed |
| D@5s | A | natural | 无 | 0 | completed |
| D@10s | A | natural | 无 | 0 | completed |
| D@20s | A | 无 | 无 | 1 | completed |
| D@50s | A | 无 | 无 | 1 | completed |
| A@natural | E1 | 5s | 10s | 0 | failed |
| E1@50ms | A | natural | 无 | 0 | completed |
| E1@200ms | A | natural | 无 | 0 | completed |
| E1@1s | A | 无 | natural | 0 | failed |
| E1@5s | A | natural | 无 | 0 | completed |
| E1@10s | A | natural | 无 | 0 | completed |
| E1@20s | A | 无 | 无 | 1 | completed |
| E1@50s | A | 无 | 无 | 1 | completed |
| A@natural | E2 | 50s | 无 | 0 | completed |
| E2@50ms | A | 无 | natural | 0 | failed |
| E2@200ms | A | natural | 无 | 0 | completed |
| E2@1s | A | natural | 无 | 0 | completed |
| E2@5s | A | natural | 无 | 0 | completed |
| E2@10s | A | natural | 无 | 0 | completed |
| E2@20s | A | natural | 无 | 0 | completed |
| E2@50s | A | natural | 无 | 0 | completed |
| A@natural | E3 | 5s | 10s | 0 | failed |
| E3@50ms | A | natural | 无 | 0 | completed |
| E3@200ms | A | natural | 无 | 0 | completed |
| E3@1s | A | natural | 无 | 0 | completed |
| E3@5s | A | natural | 无 | 0 | completed |
| E3@10s | A | natural | 无 | 0 | completed |
| E3@20s | A | 无 | 无 | 1 | completed |
| E3@50s | A | 无 | 无 | 1 | completed |
| A@natural | E4 | 50s | 无 | 0 | completed |
| E4@50ms | A | natural | 无 | 0 | completed |
| E4@200ms | A | natural | 无 | 0 | completed |
| E4@1s | A | 无 | natural | 0 | failed |
| E4@5s | A | natural | 无 | 0 | completed |
| E4@10s | A | natural | 无 | 0 | completed |
| E4@20s | A | natural | 无 | 0 | completed |
| E4@50s | A | natural | 无 | 0 | completed |
| A@natural | E5 | 50ms | 200ms | 0 | failed |
| E5@50ms | A | natural | 无 | 0 | completed |
| E5@200ms | A | natural | 无 | 0 | completed |
| E5@1s | A | 无 | 无 | 1 | completed |
| E5@5s | A | 无 | 无 | 1 | completed |
| E5@10s | A | 无 | 无 | 1 | completed |
| E5@20s | A | 无 | 无 | 1 | completed |
| E5@50s | A | 无 | 无 | 1 | completed |
| B@50ms | C | 无 | 50ms | 0 | failed |
| C@50ms | B | 200ms | 1s | 0 | failed |
| C@200ms | B | 10s | 20s | 1 | failed |
| C@1s | B | 1s | 5s | 1 | failed |
| C@5s | B | 50s | 无 | 3 | completed |
| C@10s | B | 20s | 无 | 4 | completed |
| C@20s | B | 5s | 无 | 5 | completed |
| C@50s | B | 50ms | 无 | 6 | completed |
| B@200ms | C | 无 | 50ms | 0 | failed |
| B@1s | C | 200ms | 1s | 0 | failed |
| B@5s | C | 10s | 20s | 1 | failed |
| B@10s | C | 无 | 200ms | 1 | failed |
| B@20s | C | 5s | 10s | 2 | failed |
| B@50s | C | 无 | 5s | 3 | failed |
| B@50ms | D | 5s | 10s | 0 | failed |
| D@50ms | B | 无 | 50ms | 0 | failed |
| D@200ms | B | 无 | 50ms | 0 | failed |
| D@1s | B | 无 | 50ms | 0 | failed |
| D@5s | B | 无 | 50ms | 0 | failed |
| D@10s | B | 200ms | 1s | 0 | failed |
| D@20s | B | 200ms | 1s | 1 | failed |
| D@50s | B | 200ms | 1s | 1 | failed |
| B@200ms | D | 50s | 无 | 3 | completed |
| B@1s | D | 50s | 无 | 3 | completed |
| B@5s | D | 50ms | 无 | 6 | completed |
| B@10s | D | 50ms | 无 | 6 | completed |
| B@20s | D | 50ms | 无 | 6 | completed |
| B@50s | D | 50ms | 无 | 6 | completed |
| B@50ms | E1 | 50ms | 200ms | 0 | failed |
| E1@50ms | B | 无 | 50ms | 0 | failed |
| E1@200ms | B | 50ms | 200ms | 0 | failed |
| E1@1s | B | 无 | 200ms | 1 | failed |
| E1@5s | B | 200ms | 1s | 1 | failed |
| E1@10s | B | 无 | 1s | 2 | failed |
| E1@20s | B | 无 | 1s | 2 | failed |
| E1@50s | B | 无 | 1s | 2 | failed |
| B@200ms | E1 | 1s | 5s | 1 | failed |
| B@1s | E1 | 50s | 无 | 3 | completed |
| B@5s | E1 | 无 | 无 | 7 | completed |
| B@10s | E1 | 无 | 无 | 7 | completed |
| B@20s | E1 | 无 | 无 | 7 | completed |
| B@50s | E1 | 无 | 无 | 7 | completed |
| B@50ms | E2 | 无 | 50ms | 0 | failed |
| E2@50ms | B | 50ms | 200ms | 0 | failed |
| E2@200ms | B | 无 | 200ms | 1 | failed |
| E2@1s | B | 200ms | 1s | 1 | failed |
| E2@5s | B | 200ms | 1s | 1 | failed |
| E2@10s | B | 50s | 无 | 2 | completed |
| E2@20s | B | 无 | 1s | 2 | failed |
| E2@50s | B | 1s | 5s | 2 | failed |
| B@200ms | E2 | 1s | 5s | 0 | failed |
| B@1s | E2 | 50s | 无 | 2 | completed |
| B@5s | E2 | 50s | 无 | 5 | completed |
| B@10s | E2 | 10s | 无 | 6 | completed |
| B@20s | E2 | 10s | 无 | 6 | completed |
| B@50s | E2 | 10s | 无 | 6 | completed |
| B@50ms | E3 | 10s | 20s | 0 | failed |
| E3@50ms | B | 无 | 50ms | 0 | failed |
| E3@200ms | B | 无 | 50ms | 0 | failed |
| E3@1s | B | 无 | 50ms | 0 | failed |
| E3@5s | B | 无 | 50ms | 0 | failed |
| E3@10s | B | 无 | 50ms | 0 | failed |
| E3@20s | B | 50ms | 200ms | 0 | failed |
| E3@50s | B | 无 | 200ms | 1 | failed |
| B@200ms | E3 | 50s | 无 | 5 | completed |
| B@1s | E3 | 无 | 无 | 7 | completed |
| B@5s | E3 | 无 | 无 | 7 | completed |
| B@10s | E3 | 无 | 无 | 7 | completed |
| B@20s | E3 | 无 | 无 | 7 | completed |
| B@50s | E3 | 无 | 无 | 7 | completed |
| B@50ms | E4 | 50s | 无 | 0 | completed |
| E4@50ms | B | 无 | 50ms | 0 | failed |
| E4@200ms | B | 无 | 50ms | 0 | failed |
| E4@1s | B | 50ms | 200ms | 0 | failed |
| E4@5s | B | 无 | 50ms | 0 | failed |
| E4@10s | B | 无 | 50ms | 0 | failed |
| E4@20s | B | 无 | 50ms | 0 | failed |
| E4@50s | B | 50ms | 200ms | 0 | failed |
| B@200ms | E4 | 50s | 无 | 5 | completed |
| B@1s | E4 | 无 | 无 | 7 | completed |
| B@5s | E4 | 无 | 无 | 7 | completed |
| B@10s | E4 | 无 | 无 | 7 | completed |
| B@20s | E4 | 无 | 无 | 7 | completed |
| B@50s | E4 | 无 | 无 | 7 | completed |
| B@50ms | E5 | 200ms | 1s | 0 | failed |
| E5@50ms | B | 无 | 50ms | 0 | failed |
| E5@200ms | B | 50ms | 200ms | 0 | failed |
| E5@1s | B | 200ms | 1s | 0 | failed |
| E5@5s | B | 1s | 5s | 2 | failed |
| E5@10s | B | 50s | 无 | 3 | completed |
| E5@20s | B | 20s | 无 | 5 | completed |
| E5@50s | B | 无 | 无 | 7 | completed |
| B@200ms | E5 | 200ms | 1s | 1 | failed |
| B@1s | E5 | 1s | 5s | 2 | failed |
| B@5s | E5 | 5s | 10s | 3 | failed |
| B@10s | E5 | 10s | 20s | 4 | failed |
| B@20s | E5 | 10s | 20s | 4 | failed |
| B@50s | E5 | 无 | 10s | 4 | failed |
| C@50ms | D | 5s | 10s | 0 | failed |
| D@50ms | C | 无 | 50ms | 0 | failed |
| D@200ms | C | 无 | 50ms | 0 | failed |
| D@1s | C | 无 | 50ms | 0 | failed |
| D@5s | C | 无 | 50ms | 0 | failed |
| D@10s | C | 50ms | 200ms | 0 | failed |
| D@20s | C | 无 | 200ms | 1 | failed |
| D@50s | C | 5s | 10s | 1 | failed |
| C@200ms | D | 50s | 无 | 3 | completed |
| C@1s | D | 50s | 无 | 5 | completed |
| C@5s | D | 50s | 无 | 5 | completed |
| C@10s | D | 50s | 无 | 5 | completed |
| C@20s | D | 50ms | 无 | 6 | completed |
| C@50s | D | 50ms | 无 | 6 | completed |
| C@50ms | E1 | 50s | 无 | 0 | completed |
| E1@50ms | C | 无 | 50ms | 0 | failed |
| E1@200ms | C | 无 | 50ms | 0 | failed |
| E1@1s | C | 50ms | 200ms | 0 | failed |
| E1@5s | C | 无 | 50ms | 0 | failed |
| E1@10s | C | 200ms | 1s | 0 | failed |
| E1@20s | C | 50ms | 200ms | 0 | failed |
| E1@50s | C | 无 | 50ms | 0 | failed |
| C@200ms | E1 | 20s | 无 | 4 | completed |
| C@1s | E1 | 10s | 无 | 6 | completed |
| C@5s | E1 | 无 | 无 | 7 | completed |
| C@10s | E1 | 无 | 无 | 7 | completed |
| C@20s | E1 | 无 | 无 | 7 | completed |
| C@50s | E1 | 无 | 无 | 7 | completed |
| C@50ms | E2 | 无 | 50ms | 0 | failed |
| E2@50ms | C | 50ms | 200ms | 0 | failed |
| E2@200ms | C | 无 | 200ms | 1 | failed |
| E2@1s | C | 200ms | 1s | 1 | failed |
| E2@5s | C | 无 | 200ms | 1 | failed |
| E2@10s | C | 无 | 200ms | 1 | failed |
| E2@20s | C | 1s | 5s | 1 | failed |
| E2@50s | C | 无 | 200ms | 1 | failed |
| C@200ms | E2 | 50s | 无 | 0 | completed |
| C@1s | E2 | 20s | 无 | 5 | completed |
| C@5s | E2 | 20s | 无 | 6 | completed |
| C@10s | E2 | 无 | 无 | 7 | completed |
| C@20s | E2 | 无 | 无 | 7 | completed |
| C@50s | E2 | 无 | 无 | 7 | completed |
| C@50ms | E3 | 50s | 无 | 0 | completed |
| E3@50ms | C | 无 | 50ms | 0 | failed |
| E3@200ms | C | 无 | 50ms | 0 | failed |
| E3@1s | C | 无 | 50ms | 0 | failed |
| E3@5s | C | 200ms | 1s | 0 | failed |
| E3@10s | C | 无 | 50ms | 0 | failed |
| E3@20s | C | 无 | 50ms | 0 | failed |
| E3@50s | C | 无 | 50ms | 0 | failed |
| C@200ms | E3 | 5s | 无 | 6 | completed |
| C@1s | E3 | 5s | 无 | 6 | completed |
| C@5s | E3 | 无 | 无 | 7 | completed |
| C@10s | E3 | 无 | 无 | 7 | completed |
| C@20s | E3 | 无 | 无 | 7 | completed |
| C@50s | E3 | 无 | 无 | 7 | completed |
| C@50ms | E4 | 10s | 20s | 0 | failed |
| E4@50ms | C | 无 | 50ms | 0 | failed |
| E4@200ms | C | 无 | 50ms | 0 | failed |
| E4@1s | C | 无 | 50ms | 0 | failed |
| E4@5s | C | 无 | 50ms | 0 | failed |
| E4@10s | C | 无 | 50ms | 0 | failed |
| E4@20s | C | 50ms | 200ms | 0 | failed |
| E4@50s | C | 无 | 200ms | 1 | failed |
| C@200ms | E4 | 50s | 无 | 5 | completed |
| C@1s | E4 | 无 | 无 | 7 | completed |
| C@5s | E4 | 无 | 无 | 7 | completed |
| C@10s | E4 | 无 | 无 | 7 | completed |
| C@20s | E4 | 无 | 无 | 7 | completed |
| C@50s | E4 | 无 | 无 | 7 | completed |
| C@50ms | E5 | 10s | 20s | 0 | failed |
| E5@50ms | C | 无 | 50ms | 0 | failed |
| E5@200ms | C | 无 | 50ms | 0 | failed |
| E5@1s | C | 无 | 50ms | 0 | failed |
| E5@5s | C | 无 | 50ms | 0 | failed |
| E5@10s | C | 10s | 20s | 0 | failed |
| E5@20s | C | 1s | 10s | 1 | failed |
| E5@50s | C | 50s | 无 | 3 | completed |
| C@200ms | E5 | 20s | 50s | 4 | failed |
| C@1s | E5 | 10s | 20s | 4 | failed |
| C@5s | E5 | 无 | 10s | 4 | failed |
| C@10s | E5 | 50s | 无 | 4 | completed |
| C@20s | E5 | 10s | 50s | 5 | failed |
| C@50s | E5 | 无 | 50s | 6 | failed |
| D@50ms | E1 | 无 | 50ms | 0 | failed |
| E1@50ms | D | 50s | 无 | 0 | completed |
| E1@200ms | D | 10s | 50s | 5 | failed |
| E1@1s | D | 50s | 无 | 6 | completed |
| E1@5s | D | 无 | 50s | 6 | failed |
| E1@10s | D | 无 | 50s | 6 | failed |
| E1@20s | D | 50s | 无 | 6 | completed |
| E1@50s | D | 无 | 50s | 6 | failed |
| D@200ms | E1 | 无 | 50ms | 0 | failed |
| D@1s | E1 | 无 | 50ms | 0 | failed |
| D@5s | E1 | 无 | 50ms | 0 | failed |
| D@10s | E1 | 50ms | 200ms | 0 | failed |
| D@20s | E1 | 无 | 50ms | 0 | failed |
| D@50s | E1 | 50s | 无 | 0 | completed |
| D@50ms | E2 | 无 | 50ms | 0 | failed |
| E2@50ms | D | 10s | 20s | 0 | failed |
| E2@200ms | D | 20s | 50s | 4 | failed |
| E2@1s | D | 50s | 无 | 6 | completed |
| E2@5s | D | 无 | 50s | 6 | failed |
| E2@10s | D | 50s | 无 | 6 | completed |
| E2@20s | D | 无 | 无 | 7 | completed |
| E2@50s | D | 无 | 无 | 7 | completed |
| D@200ms | E2 | 无 | 50ms | 0 | failed |
| D@1s | E2 | 无 | 50ms | 0 | failed |
| D@5s | E2 | 无 | 50ms | 0 | failed |
| D@10s | E2 | 50ms | 200ms | 0 | failed |
| D@20s | E2 | 50ms | 200ms | 0 | failed |
| D@50s | E2 | 5s | 10s | 1 | failed |
| D@50ms | E3 | 无 | 50ms | 0 | failed |
| E3@50ms | D | 10s | 20s | 0 | failed |
| E3@200ms | D | 无 | 20s | 5 | failed |
| E3@1s | D | 无 | 20s | 5 | failed |
| E3@5s | D | 50s | 无 | 5 | completed |
| E3@10s | D | 无 | 50s | 6 | failed |
| E3@20s | D | 50s | 无 | 6 | completed |
| E3@50s | D | 50s | 无 | 6 | completed |
| D@200ms | E3 | 无 | 50ms | 0 | failed |
| D@1s | E3 | 无 | 50ms | 0 | failed |
| D@5s | E3 | 无 | 50ms | 0 | failed |
| D@10s | E3 | 无 | 50ms | 0 | failed |
| D@20s | E3 | 1s | 5s | 0 | failed |
| D@50s | E3 | 50s | 无 | 3 | completed |
| D@50ms | E4 | 无 | 50ms | 0 | failed |
| E4@50ms | D | 5s | 10s | 0 | failed |
| E4@200ms | D | 10s | 20s | 3 | failed |
| E4@1s | D | 10s | 20s | 4 | failed |
| E4@5s | D | 无 | 10s | 4 | failed |
| E4@10s | D | 无 | 10s | 4 | failed |
| E4@20s | D | 10s | 20s | 4 | failed |
| E4@50s | D | 50s | 无 | 5 | completed |
| D@200ms | E4 | 50ms | 200ms | 0 | failed |
| D@1s | E4 | 无 | 50ms | 0 | failed |
| D@5s | E4 | 无 | 50ms | 0 | failed |
| D@10s | E4 | 10s | 20s | 0 | failed |
| D@20s | E4 | 50s | 无 | 3 | completed |
| D@50s | E4 | 50s | 无 | 6 | completed |
| D@50ms | E5 | 无 | 50ms | 0 | failed |
| E5@50ms | D | 5s | 10s | 0 | failed |
| E5@200ms | D | 10s | 20s | 3 | failed |
| E5@1s | D | 50s | 无 | 4 | completed |
| E5@5s | D | 50s | 无 | 6 | completed |
| E5@10s | D | 50s | 无 | 6 | completed |
| E5@20s | D | 无 | 无 | 7 | completed |
| E5@50s | D | 无 | 无 | 7 | completed |
| D@200ms | E5 | 无 | 50ms | 0 | failed |
| D@1s | E5 | 无 | 50ms | 0 | failed |
| D@5s | E5 | 200ms | 1s | 0 | failed |
| D@10s | E5 | 50ms | 200ms | 0 | failed |
| D@20s | E5 | 200ms | 1s | 1 | failed |
| D@50s | E5 | 5s | 10s | 2 | failed |
| E1@50ms | E2 | 5s | 10s | 0 | failed |
| E2@50ms | E1 | 50s | 无 | 0 | completed |
| E2@200ms | E1 | 50s | 无 | 1 | completed |
| E2@1s | E1 | 20s | 无 | 2 | completed |
| E2@5s | E1 | 10s | 无 | 4 | completed |
| E2@10s | E1 | 50ms | 1s | 1 | failed |
| E2@20s | E1 | 1s | 无 | 6 | completed |
| E2@50s | E1 | 1s | 无 | 6 | completed |
| E1@200ms | E2 | 无 | 50ms | 0 | failed |
| E1@1s | E2 | 50s | 无 | 0 | completed |
| E1@5s | E2 | 200ms | 1s | 0 | failed |
| E1@10s | E2 | 1s | 5s | 0 | failed |
| E1@20s | E2 | 200ms | 1s | 0 | failed |
| E1@50s | E2 | 50ms | 200ms | 0 | failed |
| E1@50ms | E3 | 50s | 无 | 0 | completed |
| E3@50ms | E1 | 50s | 无 | 0 | completed |
| E3@200ms | E1 | 无 | 50ms | 0 | failed |
| E3@1s | E1 | 1s | 5s | 0 | failed |
| E3@5s | E1 | 10s | 20s | 0 | failed |
| E3@10s | E1 | 1s | 5s | 1 | failed |
| E3@20s | E1 | 无 | 50ms | 0 | failed |
| E3@50s | E1 | 1s | 5s | 1 | failed |
| E1@200ms | E3 | 1s | 5s | 1 | failed |
| E1@1s | E3 | 50s | 无 | 2 | completed |
| E1@5s | E3 | 50s | 无 | 2 | completed |
| E1@10s | E3 | 5s | 无 | 5 | completed |
| E1@20s | E3 | 5s | 无 | 5 | completed |
| E1@50s | E3 | 50ms | 无 | 6 | completed |
| E1@50ms | E4 | 50s | 无 | 0 | completed |
| E4@50ms | E1 | 10s | 20s | 0 | failed |
| E4@200ms | E1 | 50s | 无 | 0 | completed |
| E4@1s | E1 | 无 | 50ms | 0 | failed |
| E4@5s | E1 | 无 | 50ms | 0 | failed |
| E4@10s | E1 | 200ms | 1s | 0 | failed |
| E4@20s | E1 | 50ms | 200ms | 0 | failed |
| E4@50s | E1 | 50s | 无 | 0 | completed |
| E1@200ms | E4 | 50s | 无 | 2 | completed |
| E1@1s | E4 | 50s | 无 | 3 | completed |
| E1@5s | E4 | 50s | 无 | 4 | completed |
| E1@10s | E4 | 50s | 无 | 4 | completed |
| E1@20s | E4 | 50s | 无 | 4 | completed |
| E1@50s | E4 | 50s | 无 | 5 | completed |
| E1@50ms | E5 | 200ms | 1s | 0 | failed |
| E5@50ms | E1 | 5s | 10s | 0 | failed |
| E5@200ms | E1 | 50ms | 200ms | 0 | failed |
| E5@1s | E1 | 50s | 无 | 0 | completed |
| E5@5s | E1 | 200ms | 无 | 6 | completed |
| E5@10s | E1 | 无 | 无 | 7 | completed |
| E5@20s | E1 | 无 | 无 | 7 | completed |
| E5@50s | E1 | 无 | 无 | 7 | completed |
| E1@200ms | E5 | 1s | 5s | 0 | failed |
| E1@1s | E5 | 50ms | 1s | 1 | failed |
| E1@5s | E5 | 50ms | 1s | 1 | failed |
| E1@10s | E5 | 50ms | 1s | 1 | failed |
| E1@20s | E5 | 无 | 1s | 2 | failed |
| E1@50s | E5 | 无 | 1s | 2 | failed |
| E2@50ms | E3 | 50s | 无 | 0 | completed |
| E3@50ms | E2 | 50ms | 200ms | 0 | failed |
| E3@200ms | E2 | 5s | 10s | 0 | failed |
| E3@1s | E2 | 50s | 无 | 0 | completed |
| E3@5s | E2 | 5s | 10s | 0 | failed |
| E3@10s | E2 | 无 | 50ms | 0 | failed |
| E3@20s | E2 | 50ms | 200ms | 0 | failed |
| E3@50s | E2 | 50s | 无 | 1 | completed |
| E2@200ms | E3 | 50s | 无 | 1 | completed |
| E2@1s | E3 | 5s | 50s | 3 | failed |
| E2@5s | E3 | 1s | 5s | 1 | failed |
| E2@10s | E3 | 50s | 无 | 3 | completed |
| E2@20s | E3 | 50s | 无 | 5 | completed |
| E2@50s | E3 | 50s | 无 | 5 | completed |
| E2@50ms | E4 | 50s | 无 | 0 | completed |
| E4@50ms | E2 | 20s | 50s | 0 | failed |
| E4@200ms | E2 | 5s | 10s | 0 | failed |
| E4@1s | E2 | 50ms | 200ms | 0 | failed |
| E4@5s | E2 | 无 | 50ms | 0 | failed |
| E4@10s | E2 | 5s | 10s | 0 | failed |
| E4@20s | E2 | 无 | 50ms | 0 | failed |
| E4@50s | E2 | 1s | 5s | 0 | failed |
| E2@200ms | E4 | 50s | 无 | 2 | completed |
| E2@1s | E4 | 50s | 无 | 3 | completed |
| E2@5s | E4 | 50s | 无 | 3 | completed |
| E2@10s | E4 | 10s | 无 | 4 | completed |
| E2@20s | E4 | 50ms | 无 | 6 | completed |
| E2@50s | E4 | 50ms | 无 | 6 | completed |
| E2@50ms | E5 | 200ms | 1s | 0 | failed |
| E5@50ms | E2 | 无 | 50ms | 0 | failed |
| E5@200ms | E2 | 无 | 50ms | 0 | failed |
| E5@1s | E2 | 50s | 无 | 0 | completed |
| E5@5s | E2 | 无 | 无 | 7 | completed |
| E5@10s | E2 | 无 | 无 | 7 | completed |
| E5@20s | E2 | 无 | 无 | 7 | completed |
| E5@50s | E2 | 无 | 无 | 7 | completed |
| E2@200ms | E5 | 无 | 1s | 2 | failed |
| E2@1s | E5 | 无 | 1s | 2 | failed |
| E2@5s | E5 | 无 | 1s | 2 | failed |
| E2@10s | E5 | 无 | 1s | 2 | failed |
| E2@20s | E5 | 无 | 1s | 2 | failed |
| E2@50s | E5 | 无 | 1s | 2 | failed |
| E3@50ms | E4 | 50s | 无 | 0 | completed |
| E4@50ms | E3 | 50s | 无 | 0 | completed |
| E4@200ms | E3 | 50s | 无 | 0 | completed |
| E4@1s | E3 | 50s | 无 | 0 | completed |
| E4@5s | E3 | 50s | 无 | 0 | completed |
| E4@10s | E3 | 200ms | 1s | 0 | failed |
| E4@20s | E3 | 50s | 无 | 1 | completed |
| E4@50s | E3 | 50s | 无 | 1 | completed |
| E3@200ms | E4 | 5s | 10s | 0 | failed |
| E3@1s | E4 | 50s | 无 | 0 | completed |
| E3@5s | E4 | 50s | 无 | 1 | completed |
| E3@10s | E4 | 20s | 50s | 1 | failed |
| E3@20s | E4 | 50s | 无 | 1 | completed |
| E3@50s | E4 | 50s | 无 | 1 | completed |
| E3@50ms | E5 | 50ms | 200ms | 0 | failed |
| E5@50ms | E3 | 50s | 无 | 0 | completed |
| E5@200ms | E3 | 50ms | 200ms | 0 | failed |
| E5@1s | E3 | 50s | 无 | 1 | completed |
| E5@5s | E3 | 无 | 无 | 7 | completed |
| E5@10s | E3 | 无 | 无 | 7 | completed |
| E5@20s | E3 | 无 | 无 | 7 | completed |
| E5@50s | E3 | 无 | 无 | 7 | completed |
| E3@200ms | E5 | 200ms | 1s | 0 | failed |
| E3@1s | E5 | 50ms | 1s | 1 | failed |
| E3@5s | E5 | 50ms | 1s | 1 | failed |
| E3@10s | E5 | 50ms | 1s | 1 | failed |
| E3@20s | E5 | 50ms | 1s | 1 | failed |
| E3@50s | E5 | 50ms | 1s | 1 | failed |
| E4@50ms | E5 | 50ms | 200ms | 0 | failed |
| E5@50ms | E4 | 50s | 无 | 0 | completed |
| E5@200ms | E4 | 5s | 10s | 0 | failed |
| E5@1s | E4 | 5s | 10s | 2 | failed |
| E5@5s | E4 | 50s | 无 | 4 | completed |
| E5@10s | E4 | 无 | 无 | 7 | completed |
| E5@20s | E4 | 无 | 无 | 7 | completed |
| E5@50s | E4 | 无 | 无 | 7 | completed |
| E4@200ms | E5 | 50ms | 200ms | 0 | failed |
| E4@1s | E5 | 200ms | 1s | 0 | failed |
| E4@5s | E5 | 200ms | 1s | 0 | failed |
| E4@10s | E5 | 1s | 5s | 0 | failed |
| E4@20s | E5 | 50ms | 5s | 2 | failed |
| E4@50s | E5 | 50ms | 5s | 2 | failed |

路线的最高实测档与最高推断档分别记录；推断通过不表示实战胜利。失败只终止该路线，不影响其他版本。

## 颜色与对局矩阵

- `figures/v6_score_heatmap.svg`：57×57全部模型版本的实际配对得分率。
- `figures/v6_black_score_heatmap.svg` 与 `v6_white_score_heatmap.svg`：按该模型执黑/执白分组的实际得分率；空白表示没有该条件的实测对局。
- `figures/v6_budget_compute.svg`：按预算展示真实对局决策的耗时、节点、搜索深度与模拟量；它不补齐早停后未实测的档位。
- `results/score_matrix.csv`、`status_matrix.csv`、`black_score_matrix.csv`、`white_score_matrix.csv`同时区分实际新跑、V5.1复用、推断和未测试。

## 关键结论与限制

E1–E4的证明输出只表示搜索范围内找到并经独立验证的胜利证书；没有证书表示预算或搜索边界内未找到，不表示不存在必胜路线。E3的威胁排序同样不等于数学证明。E5组件的独立收益应由相同局面、相同深度的消融条目比较；在不同节点数/深度下的墙钟结果同时受截止时间影响。
逐档挑战假设对手更高预算不会更弱，这只用于控制实验规模。传统搜索的时间分配、开局和对手策略会造成非单调观测；任何推断行都没有当作实测。两局配对可以显示方向性，但不足以稳定估计棋力差距。

## Drive恢复快照

Drive终态快照包含1018局、509组；SHA-256：`da11bce4a0fbde597f5acd90c09c702f3348ee94aa65f4eff7d8e86ff9d916d7`。
已从Drive下载并校验后恢复到空目录，恢复计数1018局；Drive目录只保留这一份V6有效快照。上传时间：2026-10-23T16:02:55.927Z。
快照链接：https://drive.google.com/file/d/1my7uwj3gJDbquqtRhQ-jSzaF1NfH11_B/view?usp=drivesdk

## 原始成果

完整记录：`results/9x9_games.jsonl`；进度/来源：`progress.json`、`reuse_manifest.json`、`routes.csv`；统计：`summary.json`、`pairwise_results.csv`、矩阵CSV、`model_usage.csv`；验收：`validation.json`；战术题与E5消融见`validation/`目录。

验收回放：1018局、54296手；赛程最终状态：完整；活动耗时约38.34小时。
