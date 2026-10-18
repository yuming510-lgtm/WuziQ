# 四种五子棋 AI 研究实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 WuziQ 的规则与 UI 基础上，实现 A/B/C/D，完成可恢复且可复现的 9×9 正式循环赛、15×15 补充赛和中文报告。

**Architecture:** `src/ai.py` 提供四算法统一接口；Web 只增加模型选择，AI 自动对弈由研究 runner 负责。`src/research/` 管理配对赛、固定深度效率比较、统计汇总和原子检查点；原始逐局 JSONL 是报告与恢复的唯一事实来源。

**Tech Stack:** Python 3.12、标准库、现有 Flask、pytest。

**Spec:** `docs/superpowers/specs/2026-10-17-four-ai-gomoku-research-design.md`

## Global Constraints

- 不改变 `GomokuBoard` 的规则、胜负定义和存档格式。
- B/C 共用候选点、评价函数、排序、最大深度；算法差异仅为 Alpha-Beta 界限与 cutoff。
- A、B、C、D 全部使用公开棋局；D 不使用神经网络。
- 每步统一 0.05 秒时限和 4 层深度上限；搜索超时保留合法回退并如实记数。
- 9×9 每组 200 局、双颜色各 100 局；15×15 每组 20 局。
- 固定种子、成对开局、交换黑白、唯一对局 ID；每局落盘后才能增加进度。
- 成绩分析报告平局、纯胜率、计半得分率、95% 区间、耗时、节点/模拟数及颜色优势。

## Review Focus

- 满盘、空盘与候选为空：必须明确返回“无合法着法”或唯一中心点，不得索引越界（`test_ai.py`）。
- 只有一步胜着或对手一步胜着：候选顺序不得漏掉赢棋/挡棋点（`test_ai.py`）。
- 时间耗尽发生在根搜索未完成时：仍返回合法回退并标记 timeout（`test_ai.py`）。
- JSONL 尾部截断或中途重启：忽略并清理不完整尾行，已完成局 ID 不重复；其他损坏数据显式报错（`test_research.py`）。
- 搜索在末路局面终止或胜负检查后状态结束：runner 不能再请求 AI 并产生非法落子（`test_research.py`）。

---

### Task 1: 统一 AI 引擎和可重复随机性

**Files:**
- Modify: `src/gomoku.py`
- Create: `src/ai.py`
- Test: `tests/test_ai.py`

**Interfaces:**
- `choose_move(board: GomokuBoard, player: Player, model: str, config: SearchConfig, seed: int) -> SearchResult`
- `SearchResult`: `move`, `score`, `nodes`, `completed_depth`, `cutoffs`, `simulations`, `elapsed_seconds`, `timed_out`
- `generate_candidates(board, player, opponent) -> list[Coordinate]`
- `evaluate_board(board, perspective) -> int`

- [ ] **Step 1: Write failing engine tests**：A-D 合法性与种子重复、空盘中心点、满盘、立即赢与阻挡、B/C 同根值/同最优集合/不同或相等节点、Alpha-Beta cutoff、MCTS 模拟次数、零时限回退。
- [ ] **Step 2: Run `pytest tests/test_ai.py -q`**，确认新接口缺失导致失败。
- [ ] **Step 3: Implement stable common candidates/evaluator and `SearchResult`; retain `select_ai_move` behavior while accepting an optional RNG**。
- [ ] **Step 4: Implement complete-depth iterative Minimax and Alpha-Beta using the same ordered children and evaluation; add deterministic tie-breaking and deadline checks**。
- [ ] **Step 5: Implement UCT selection, expansion, tactical random rollout, and backpropagation; stop on deadline/simulation cap**。
- [ ] **Step 6: Run focused AI tests and `pytest tests/test_gomoku.py -q`**。

### Task 2: Web 模型选择

**Files:**
- Modify: `src/webapp.py`
- Modify: `src/templates/index.html`
- Modify: `src/static/app.js`
- Test: `tests/test_web.py`

**Interfaces:**
- Web state returns `ai_model` in `A/B/C/D` and accepts it on `/config` and `/reset`; omission defaults to A.
- Existing `ai_side` values and game lifecycle remain unchanged.

- [ ] **Step 1: Add API tests** for default A, selecting each model, invalid model rejection, and selected model used on next AI turn.
- [ ] **Step 2: Run focused web tests and confirm failure**。
- [ ] **Step 3: Wire `choose_move` and add accessible model selector without changing board controls**。
- [ ] **Step 4: Run `pytest tests/test_web.py -q` and original UI import test**。

### Task 3: Checkpoint and resumable tournament

**Files:**
- Create: `src/research/__init__.py`
- Create: `src/research/checkpoint.py`
- Create: `src/research/runner.py`
- Create: `configs/four_ai_experiment.json`
- Test: `tests/test_research.py`

**Interfaces:**
- `game_id(board_size, model_pair, opening_index, color_swap) -> str`
- `load_completed_games(path) -> (records_by_id, valid_file_end)`
- `run_schedule(config, output_dir, resume=True) -> list[dict]`
- Every game record contains models/colors/result/moves/timing/search counters/seed and stable game ID.

- [ ] **Step 1: Test paired schedule sizes, color balance, stable IDs, duplicate skipping, deterministic opening seeds, and truncated-tail recovery**。
- [ ] **Step 2: Run `pytest tests/test_research.py -q` and confirm failure**。
- [ ] **Step 3: Implement atomic config/progress writes and append+fsync one complete JSONL record per game**。
- [ ] **Step 4: Implement six unique matchups with 9×9 200 and 15×15 20 games per matchup; re-run skips every completed ID**。
- [ ] **Step 5: Test legal game completion, forfeit-on-illegal, timeout accounting, and interruption/resume without duplicate IDs**。

### Task 4: B/C fixed-depth study and statistical summaries

**Files:**
- Create: `src/research/efficiency.py`
- Create: `src/research/metrics.py`
- Create: `scripts/run_four_ai_efficiency.py`
- Test: `tests/test_research.py`

- [ ] **Step 1: Add tests** for match matrix completeness, pure win rate, draw-half score rate, Wilson and paired-block 95% intervals, black/white counts, and B/C root value equality.
- [ ] **Step 2: Run focused tests and confirm failure**。
- [ ] **Step 3: Implement deterministic fixed-depth position suite and emit exact value, optimal move set, nodes, cutoffs, elapsed time and prune ratio**。
- [ ] **Step 4: Implement per-pair and overall aggregation from raw JSONL only; use paired-opening blocks for score-rate confidence intervals**。
- [ ] **Step 5: Run synthetic result fixtures and verify ties are not spuriously ranked**。

### Task 5: Pilot, estimate, Drive restore drill, and full competition

**Files:**
- Create: `research/results/` generated files
- Modify: `configs/four_ai_experiment.json` only if pilot documents a measured value adjustment before formal run

- [ ] **Step 1: Run small fixed-depth and 12-game pilot**; capture throughput, nodes/s, simulations/s, game length, disk use, and timeouts.
- [ ] **Step 2: Compute estimated 1,320-game wall time and storage from pilot measurements; record estimate before full run**。
- [ ] **Step 3: Create one tiny real-format checkpoint, upload to Drive, fetch/download, restore into a fresh temp directory, prove completed IDs are skipped, compare checksum, and delete only the drill artifact**。
- [ ] **Step 4: Start the full 1,200-game 9×9 cycle and 120-game 15×15 supplement with resume enabled; keep progress visible and check data IDs/counts during execution**。
- [ ] **Step 5: If runtime crosses 24 hours, upload a new minimal ZIP, read back/check SHA-256, then delete the prior snapshot; otherwise do not claim a periodic snapshot occurred**。

### Task 6: Report, verification, and GitHub delivery

**Files:**
- Create: `research/REPORT.md`
- Create: `research/results/summary.json`
- Create: `research/results/efficiency.json`
- Modify: `README.md`
- Commit/push: branch `research/four-ai-gomoku`

- [ ] **Step 1: Generate all result tables and ranking from saved raw games; include full 4×4 matrix and 9×9/15×15 split**。
- [ ] **Step 2: Write Chinese report with method, environment, estimates, results, CIs, B/C separation, representative games, limitations, and next steps; state no model was trained**。
- [ ] **Step 3: Run full `pytest -q`, tournament/result integrity checks, and CLI smoke command**。
- [ ] **Step 4: Review report numbers against JSONL, verify exactly 1,200 + 120 unique games and 100/100 color split per main-series matchup**。
- [ ] **Step 5: Commit all source/tests/config/data/report and push only `research/four-ai-gomoku`; confirm commit SHA and remote branch URL**。
