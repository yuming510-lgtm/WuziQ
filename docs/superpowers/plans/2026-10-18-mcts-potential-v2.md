# MCTS Potential V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate D1 and D2 against A/B/C with reproducible, resumable equal-budget and extreme-compute Gomoku experiments.

**Architecture:** Keep the V1 runner and D path stable. Add D1 as the same UCT engine with expanded rollout/simulation limits, and D2 as a pure MCTS agent with tactical handling, root coverage, full terminal rollouts, and optional runner-managed tree reuse. Add a V2-specific schedule/runner, per-move metrics, statistical summaries, ablation and plotting scripts, and minimal recovery snapshots.

**Tech Stack:** Python 3, existing `GomokuBoard`, pytest, JSON/JSONL, matplotlib only if already available (otherwise SVG/CSV generated with stdlib), Google Drive connector for the verified recovery drill.

**Spec:** `docs/superpowers/specs/2026-10-18-mcts-potential-v2-design.md`

## Global Constraints

- Preserve V1 A/B/C/D source behavior, report, configuration, and result files.
- D1 changes only UCT compute limits and rollout horizon relative to D.
- D2 remains pure MCTS and never calls Minimax/Alpha-Beta.
- Use only public board state and fixed, derivable random seeds.
- Record completed simulations separately from timeouts and truncated rollouts.
- Illegal moves forfeit; a legal move returned after timeout is retained and recorded.
- Persist each complete game with fsync and reject duplicate game IDs on resume.
- Use 9x9 as the primary board and 15x15 as supplement; pre-generate paired color-swapped openings.
- Do not claim the Google Drive drill passed unless upload, read, hash verification, and restore actually pass.
- Push only to `research/mcts-potential-v2`; do not alter `main` or V1.

## Review Focus

- D and D1 must be seed-identical under the same config, proving the D1 route did not change UCT strategy; pin with a result-equality test.
- D2's immediate win and forced block must work even when rollout budget is zero/very small; pin with tactical board tests.
- An interrupted rollout must not be backpropagated or counted as a completed simulation; pin with separate completed/truncated counters.
- Root expansion must visit all tactical candidates before reusing budget on only a subset; pin with a bounded deterministic simulation test.
- Resume and paired seat swaps must be idempotent and use the same B-W-B-W opening; swap only which model owns black/white so the fifth-ply model changes; pin with unique-ID and schedule tests.

---

### Task 1: D1 model and per-search telemetry

**Files:**
- Modify: `src/ai.py`
- Test: `tests/test_mcts_v2.py`

**Interfaces:**
- Consumes: existing `SearchConfig`, `_search_mcts`, `SearchResult`.
- Produces: `choose_move(..., model="D1", ...)`; additive result fields `rollout_plies`, `terminal_rollouts`, `truncated_rollouts`, `root_candidates`, `root_min_visits`, and `reused_nodes` with zero defaults.

- [ ] **Step 1: Write failing D1 parity and telemetry tests**: same D/D1 config+seed returns equal move, score, visits, and counters; D1 with full remaining-ply horizon has terminal/truncated counters; existing D tests remain unchanged.
- [ ] **Step 2: Run `pytest tests/test_mcts_v2.py -q` and confirm the D1/model-field tests fail.**
- [ ] **Step 3: Add D1 routing that calls the unchanged `_search_mcts` with only the configured simulation/time limits and a rollout cap of remaining board cells. Add defaulted telemetry fields without changing D behavior.**
- [ ] **Step 4: Run `pytest tests/test_mcts_v2.py tests/test_ai.py -q`; require all passing.**
- [ ] **Step 5: Commit `feat: add high-budget baseline MCTS model`.**

### Task 2: D2 pure MCTS engine and tactical correctness

**Files:**
- Create: `src/research/mcts_d2.py`
- Modify: `src/ai.py`
- Test: `tests/test_mcts_v2.py`

**Interfaces:**
- Consumes: `GomokuBoard`, `SearchConfig`, `SearchResult`, stable per-move seed.
- Produces: `MCTSAgent.choose_move(board, player, config, seed) -> SearchResult`; D2 `choose_move` routing; options to disable tactical root checks, fair root expansion, tactical rollout, and tree reuse for ablations.

- [ ] **Step 1: Add failing tests** for immediate own win, forced opponent-win block, legal fallback, deterministic seed, candidate root coverage, terminal vs truncated rollout counters, and exact undo of the input board after search.
- [ ] **Step 2: Run the named tests and confirm they fail for missing D2 behavior.**
- [ ] **Step 3: Implement D2 UCT with immediate win/block checks; fair root expansion before repeated root allocation; tactical rollout; terminal-only backpropagation; low-overhead local candidates; max tree/rollout telemetry. Do not import B/C search functions.**
- [ ] **Step 4: Implement optional reuse in `MCTSAgent` by reconciling the prior root with the observed move history; count reused nodes, and reset safely when histories diverge.**
- [ ] **Step 5: Run `pytest tests/test_mcts_v2.py -q`; require tactical, determinism, restoration, and counter tests to pass.**
- [ ] **Step 6: Commit `feat: add tactical pure MCTS engine`.**

### Task 3: V2 schedules, paired openings, and resumable game runner

**Files:**
- Create: `src/research/mcts_v2.py`
- Create: `scripts/run_mcts_v2.py`
- Test: `tests/test_mcts_v2.py`

**Interfaces:**
- Consumes: AI selector, V1 checkpoint helpers, experiment JSON.
- Produces: `build_schedule(config) -> list[dict]`, `paired_opening(board_size, seed, opening_plies=4)`, `run_game(spec, config) -> dict`, `run_experiment(config_path, output_dir, max_games=None) -> dict`.

- [ ] **Step 1: Add failing tests** for 4-ply opening diversity, color-swapped coordinates, stable seeds, unique IDs across budget/track/board/opening, both models receiving equal fair budgets, fixed 50ms C on extreme track, illegal forfeit, timeout retention, and resume skipping completed IDs.
- [ ] **Step 2: Run named tests and confirm schedule/runner tests fail.**
- [ ] **Step 3: Implement deterministic staged schedules for D1/D2 vs A/B/C at 50ms/200ms/1s/5s plus optional 10s, and extreme D2 vs C-50ms. Keep games grouped into matching opening pairs.**
- [ ] **Step 4: Implement per-move records with simulation, tree depth, rollout depth, terminal/truncated counts, actual elapsed time, timeout, and legal/illegal status; append one complete game via V1 fsync helper and reject duplicate IDs.**
- [ ] **Step 5: Add atomic progress/config files and resume validation.**
- [ ] **Step 6: Run `pytest tests/test_mcts_v2.py tests/test_research.py -q`; require passing schedule, resume, and storage tests.**
- [ ] **Step 7: Commit `feat: add resumable V2 MCTS tournament runner`.**

### Task 4: Performance benchmark, ablations, and recovery drill

**Files:**
- Create: `configs/mcts_potential_v2.json`
- Create: `scripts/benchmark_mcts_v2.py`
- Create: `scripts/run_mcts_v2_ablation.py`
- Create: `scripts/verify_mcts_v2_snapshot.py`
- Test: `tests/test_mcts_v2.py`

**Interfaces:**
- Produces: machine-readable benchmark/ablation JSON and a recovery ZIP with config, progress, and result JSONL only.

- [ ] **Step 1: Add tests** for fixed-completed-simulation benchmark records, ablation option coverage, snapshot checksum rejection, successful local restore, and preservation of non-recovery files.
- [ ] **Step 2: Run named tests and confirm the new scripts/behavior are absent.**
- [ ] **Step 3: Add pre-registered budgets, seeds, opening generation, target sample sizes, staged expansion, and explicit output paths in config.**
- [ ] **Step 4: Implement representative 9x9/15x15 positions and real completed-simulation throughput checks at 100/1k/10k/100k (stop at measured practical ceiling and record why); add fixed-position ablation switches.**
- [ ] **Step 5: Run local snapshot round trip, then perform actual Drive upload/read/hash/restore before formal match execution; only remove the previous V2 formal snapshot after the new upload is verified.**
- [ ] **Step 6: Run `pytest tests/test_mcts_v2.py -q`; require local recovery validation to pass.**
- [ ] **Step 7: Commit `feat: add MCTS benchmark and ablation tooling`.**

### Task 5: Staged formal matches and 15x15 supplement

**Files:**
- Create: `research/v2/pilot/` artifacts
- Create: `research/v2/results/` JSONL and summary inputs
- Modify: `scripts/run_mcts_v2.py` only if the pilot exposes a correctness/performance defect

**Interfaces:**
- Consumes: verified config, schedule, benchmark scripts, checkpoint utility.
- Produces: unique, resumable full-game records for all scheduled conditions actually completed.

- [ ] **Step 1: Record CPU, memory, disk and a small 9x9/15x15 D1/D2 benchmark.**
- [ ] **Step 2: Run a small paired pilot for each budget/track and compute measured per-game wall-clock estimate, CPU demand, JSONL bytes/game, and projected total. Save it before formal runs.**
- [ ] **Step 3: Execute 50ms and 200ms conditions first, then 1s, then 5s; add 10s only if measured wall-clock supports it. At each phase, expand sample sizes by paired-opening batches toward 100–200 games/condition.**
- [ ] **Step 4: Prioritize D2-C and D1-C same-budget plus D2-vs-C extreme; execute remaining D1/D2 vs A/B/C fair matchups and the 15x15 supplement from the precomputed schedule.**
- [ ] **Step 5: After each phase verify no duplicate IDs, every requested paired game has its color swap, and each saved game has a legal terminal outcome or explicit forfeit.**
- [ ] **Step 6: At 24-hour boundaries upload and verify the new minimal Drive recovery ZIP, then delete only the previous V2 formal ZIP.**

### Task 6: Statistics, plots, and Chinese research report

**Files:**
- Create: `src/research/mcts_v2_metrics.py`
- Create: `scripts/summarize_mcts_v2.py`
- Create: `research/v2/REPORT.md`
- Create: `research/v2/figures/`
- Test: `tests/test_mcts_v2.py`

**Interfaces:**
- Produces: per-condition W/L/D, pure win rate, half-point score rate and Wilson 95% CI; black/white advantage; simulation/time/depth/timeout/resource metrics; fair and extreme rankings; JSON, CSV/SVG or PNG curves.

- [ ] **Step 1: Add tests** for Wilson intervals, draw half-points, paired color summaries, overlapping-CI rank ties, and blank/incomplete conditions.
- [ ] **Step 2: Run metrics tests and confirm they fail before implementation.**
- [ ] **Step 3: Aggregate only saved JSONL records, flag under-target sample sizes, and label unexecuted conditions as missing rather than zero.**
- [ ] **Step 4: Generate budget-to-simulation/strength curves, six matchup summaries per track/budget, D1/D2 ablations, efficiency-per-CPU-second, and Chinese report tables.**
- [ ] **Step 5: Run `pytest tests/test_mcts_v2.py tests/test_research.py -q`; require passing statistics and all checkpoint validations.**
- [ ] **Step 6: Commit `research: report MCTS V2 potential evaluation`.**

### Task 7: Full verification and V2 push

**Files:**
- Verify all changed V2 files and preserve V1 history.

- [ ] **Step 1: Run the full suite `pytest -q`; preserve and report any pre-existing failure distinctly.**
- [ ] **Step 2: Run result validator, summary generator, `git diff --check`, and verify V1 result files are unchanged from the V1 parent.**
- [ ] **Step 3: Review report tables against raw JSONL counts and cross-check one recorded game by replaying its opening/move sequence.**
- [ ] **Step 4: Commit final report/validation artifacts.**
- [ ] **Step 5: Push `research/mcts-potential-v2` and verify remote SHA; confirm `main` and V1 refs are unchanged.**
