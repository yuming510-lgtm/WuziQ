# MCTS High-Compute V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a resumable 54-game, fixed-opening, 9×9 high-compute challenge using V2's pure MCTS engine and publish reproducible V3 results.

**Architecture:** Add D3 as a name for the existing V2 `MCTSAgent`, and extend the deterministic V2 tournament infrastructure additively for a V3 challenge track and shared registered opening seeds. Keep V1/V2 records untouched; create V3-only config, preflight, Drive snapshot drill, validation, summary, chart, and Chinese report artifacts.

**Tech Stack:** Python 3, pytest, existing Gomoku/V2 MCTS and checkpoint modules, JSON/JSONL, SVG, Git.

**Spec:** `docs/research/mcts-v3-spec.md`

## Global Constraints

- D3 must execute the V2 `MCTSAgent`; do not call Minimax/Alpha-Beta or use trained models.
- Every opponent A/B/C receives 0.05 seconds per move; D3 receives 10, 20, or 50 seconds.
- Run exactly 54 official games: nine conditions, six games per condition, three shared opening pairs and three games per color.
- Register the three opening seeds before results exist and reuse the same positions for every condition.
- Record the complete games, simulations, MCTS tree depth, time, timeouts, and unique IDs; persist one game at a time.
- Treat six-game condition results as exploratory, with uncertainty stated and no claim of stable superiority.
- Preserve `main`, V1, and V2 history and data.

## Review Focus

- D3 alias accidentally selects baseline MCTS or Minimax — test the D3 route through the exact same `MCTSAgent` behavior as D2.
- Schedule silently changes the requested matrix, color balance, or openings — assert 54 games, exact per-condition counts, swaps, budgets, and shared frozen opening positions.
- Resume duplicates or counts off-schedule games — test unique-ID and schedule validation against restored records.
- Report labels configured simulation caps as actual simulations — aggregate only measured per-move results.
- Snapshot upload/readback or checksum verification fails — require a real Drive round trip and local restore before formal runs.

### Task 1: D3 routing and registered schedule

**Files:**
- Modify: `src/ai.py`
- Modify: `src/research/mcts_v2.py`
- Modify: `tests/test_mcts_v2.py`
- Create: `configs/mcts_high_compute_v3.json`

**Interfaces:**
- D3 is accepted by `choose_move(board, player, "D3", config, seed)` and routed to `MCTSAgent`.
- `build_schedule(config)` accepts the V3 `high-compute` track, D3-vs-A/B/C only, and a top-level three-entry `opening_seeds` list shared across phases.

- [ ] Write failing tests for D3 routing, challenge constraints, 54-game count, color balance, and identical coordinates across all nine conditions.
- [ ] Run the focused tests and confirm they fail because the D3/track/seed behavior is missing.
- [ ] Add only the additive routing/scheduling behavior; V2 defaults remain unchanged.
- [ ] Run focused tests and then the complete existing suite.
- [ ] Freeze the config with base seed `20261018`, opening seeds `[731091, 731092, 731093]`, three budgets, and nine registered D3-vs-A/B/C conditions.
- [ ] Commit the verified D3 routing, schedule, tests, config, spec, and plan.

### Task 2: Preflight and recovery artifacts

**Files:**
- Create: `scripts/preflight_mcts_v3.py`
- Create: `scripts/create_mcts_v3_snapshot.py`
- Create: `scripts/restore_mcts_v3_snapshot.py`
- Create: `tests/test_mcts_v3_snapshot.py`
- Create: `research/v3/pilot/` outputs

**Interfaces:**
- Preflight runs one representative legal 9×9 position at each D3 budget and emits measured simulations, elapsed time, tree depth, CPU/memory/disk data, game-time estimate, and storage estimate.
- Snapshot creator consumes a V3 results directory and writes only config, progress, JSONL game records, and the config's code commit; restore verifies checksums and returns completed game IDs.

- [ ] Write failing tests for minimal archive contents, checksum failure rejection, and exact completed-ID recovery.
- [ ] Run the focused tests and confirm they fail for absent V3 wrappers.
- [ ] Implement thin V3 wrappers over existing atomic checkpoint/snapshot helpers.
- [ ] Run focused and full suite checks.
- [ ] Run the hardware/performance preflight and the real Google Drive upload/read/hash/restore drill before formal games.
- [ ] Commit the verified V3 preflight and recovery tools.

### Task 3: Official run, validation, statistics, and chart

**Files:**
- Create: `scripts/run_mcts_v3.py`
- Create: `scripts/validate_mcts_v3_results.py`
- Create: `scripts/summarize_mcts_v3.py`
- Create: `research/v3/results/`
- Create: `research/v3/figures/`
- Test: `tests/test_mcts_v3_results.py`

**Interfaces:**
- Runner resumes through the V2 per-game durable writer and produces exactly the registered schedule.
- Validator replays every official move against the 9×9 rules and checks pair/color/opening/metric invariants.
- Summarizer reports per-condition W/D/L, pure win and score rates, paired 95% intervals, measured compute, depth, timeouts, and budget trends; chart is built from raw results.

- [ ] Write failing tests for matrix aggregation, actual simulation/depth aggregation, trend points, and count/pair validation.
- [ ] Run focused tests and confirm the new report behavior is absent.
- [ ] Implement validator, summary, and trend chart generation.
- [ ] Run focused and full suite checks.
- [ ] Commit/push the tested code and registered config so the preflight and official data are tied to a public code commit.
- [ ] Run the measured 54-game schedule to completion, resuming only by unique ID if needed.
- [ ] Validate all 54 games and derive reports directly from the JSONL records.
- [ ] Commit the official game records, validation, summary, and trend chart.

### Task 4: Research report and delivery

**Files:**
- Create: `research/v3/REPORT.md`
- Create: `research/v3/results/summary.json`
- Create: `research/v3/figures/d3_budget_trend.svg`

- [ ] Include V2 D2 comparison, budget trend, all opponent results, paired seats, compute costs, uncertainty, representative games, and algorithm-vs-compute attribution.
- [ ] State whether D3 beat A/B/C in this sample without claiming statistical superiority from six games per condition; explain the observed saturation or lack of saturation.
- [ ] Run the full test suite and V3 validator/summary checks, inspect the final diff, commit results and report, and push `research/mcts-potential-v3`.
