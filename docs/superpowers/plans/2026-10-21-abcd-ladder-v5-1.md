# V5.1 ABCD Independent-Version Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace V5's family-coupled ladder with 66 independent versioned routes backed by at most 168 unique paired groups, then deliver validated results and a Chinese report.

**Architecture:** Keep the frozen V5 decision engine and opening catalog. Add a V5.1 schedule layer where every B/C/D budget version is a challenger identity and each physical matchup references two independently scored routes. Reuse only V5 groups whose engine, budgets, opening, colors, and RNG seeds match exactly; run and checkpoint the remaining active groups, then replay and analyze from raw records.

**Tech Stack:** Python 3.12, pytest, JSON/JSONL/CSV, SVG, Git, Google Drive connector.

**Spec:** `docs/superpowers/specs/2026-10-21-abcd-ladder-v5-1-design.md`

## Global Constraints

- The board is 9×9; A is natural-time; B/C/D fixed budgets are 50ms, 200ms, 1s, 5s, 10s, 20s, 50s.
- Same-family games are forbidden; each pair is two games on one frozen legal opening with colors swapped.
- A route advances only when its own two-game points are at least 1.0; score equals wins plus half draws.
- One physical version matchup runs at most once and may feed two routes; a route's actual failure never deactivates a different fixed challenger version.
- User steering update: when a fixed challenger fails a lower opponent budget, do not play its higher opponent budget levels; infer the reverse higher-budget versions pass against that same challenger under the stated monotonic-budget assumption. Mark inference separately from actual games and exclude it from standings. Reuse any exact V5 matchup when an active route reaches it.
- Do not change `src/ai.py`, `src/gomoku.py`, or `src/research/mcts_d2.py` decision behavior.
- Reuse V5 results only after exact condition and full replay checks; never choose reuse by outcome.
- Save each complete game once, update route state atomically after groups, and resume by unique game ID.
- Drive must have exactly one latest valid V5.1 snapshot, with old V5 files untouched.
- Do not label inferred dominated outcomes as played, and do not describe two-game promotions as stable rankings.

## Review Focus

1. A route fails while its counterpart still needs a shared game — test that the shared game runs and only the active route advances.
2. A challenger loses to a lower opponent budget — test that higher opponent rungs are skipped and the reverse higher-budget routes are marked as zero-game inferences, never fabricated wins.
3. A paired group is interrupted after one game — test that the same first ID is retained, the second game resumes, and promotion waits for both unless the level becomes inferable.
4. A V5 row differs in any frozen condition — test that it is rejected for reuse and does not enter V5.1 counts.
5. Mixed W/D/L pairwise rows — test both models' cumulative points equal wins + draws/2 and rates use actual games.
6. Budget-grid route ordering and opening assignment — test all 66 ladders are monotone, acyclic, unique, deterministic, and distributed over the frozen pool.

---

### Task 1: Frozen V5.1 config and unique version matchup schedule

**Files:**
- Create: `configs/abcd_ladder_v5_1.json`
- Create: `src/research/abcd_ladder_v5_1.py`
- Test: `tests/test_abcd_ladder_v5_1_schedule.py`

**Interfaces:**
- Produces `load_config(source) -> dict`, `build_routes(config) -> dict[str, dict]`, and `build_schedule(config) -> list[dict]`.
- Each group has `group_id`, two `games`, per-model version/budget maps, opening assignment, and two route references `{route_id, level_index}`.
- Route IDs are `A>B`, `B@50ms>A`, and `{family}@{fixed_budget}>{opponent_family}`.

- [ ] Write tests asserting 66 routes, 168 unique groups, 336 unique game IDs, exactly two routes per group, and no same-family pairing.
- [ ] Run `PYTHONPATH=. pytest -q tests/test_abcd_ladder_v5_1_schedule.py`; confirm failures identify missing V5.1 schedule API.
- [ ] Implement config validation, legal frozen opening validation, deterministic A ladders, and row-major 7×7 B/C/D budget grids.
- [ ] Add tests for monotone route levels, 12-opening coverage, swapped colors, fixed challenger budget, and deterministic output; rerun focused tests.
- [ ] Commit `feat(v5.1): add independent budget-version schedule`.

### Task 2: Exact V5 reuse manifest and independent route adjudication

**Files:**
- Modify: `src/research/abcd_ladder_v5_1.py`
- Create: `src/research/abcd_ladder_v5_1_reuse.py`
- Test: `tests/test_abcd_ladder_v5_1_routes.py`
- Test: `tests/test_abcd_ladder_v5_1_reuse.py`

**Interfaces:**
- Produces `prepare_v5_reuse(schedule, v5_results_dir, v5_config) -> (schedule, manifest)` and `derive_state(records, config, schedule) -> dict`.
- V5.1 route state stores highest pass, failed level, every level status, route score, whether a shared result was counted, and an explicit reason for skipped levels.

- [ ] Write failing tests for baseline pairing in both directions, one-win/one-loss threshold, full draw threshold, post-failure shared-game visibility, and exact source matching.
- [ ] Run both focused test files and confirm expected failures.
- [ ] Implement source replay validation and exact comparisons for engine hashes/settings/openings/budget/color/game seed; record V5 provenance without counting any source twice.
- [ ] Implement independent route progression over the frozen topological schedule; permit a group when at least one referenced route is active and at that level.
- [ ] Test all 28 eligible V5 groups and mutate one condition at a time to verify rejection; test no cycle/duplicate or route-wide early-stop propagation.
- [ ] Commit `feat(v5.1): validate reuse and adjudicate independent routes`.

### Task 3: Durable runner, group progress, and V5.1 Drive snapshots

**Files:**
- Create: `src/research/abcd_ladder_v5_1_runner.py`
- Create: `src/research/abcd_ladder_v5_1_snapshot.py`
- Create: `scripts/run_abcd_ladder_v5_1.py`
- Create: `scripts/create_abcd_ladder_v5_1_snapshot.py`
- Test: `tests/test_abcd_ladder_v5_1_runner.py`
- Test: `tests/test_abcd_ladder_v5_1_snapshot.py`

**Interfaces:**
- Runner reuses `src.research.abcd_ladder_v5.run_game` with the frozen V5 engine configuration.
- Snapshot contains only V5.1 config, materialized schedule, reuse manifest, JSONL records, progress and code identity.

- [ ] Write failing resume tests for one-game partial groups, duplicate IDs, skipped groups with no active routes, and shared groups with exactly one live route.
- [ ] Run runner tests; confirm they fail for absent V5.1 runner.
- [ ] Implement atomic JSONL append, replay validation, completion-only route advancement, atomic progress update after each game and paired group, and code/config/schedule pin checks.
- [ ] Write failing snapshot tests for hash mismatch, non-empty restore destination, empty-dir round-trip, and restore continuation from one unfinished pair.
- [ ] Implement create/restore validation, retaining no generated reports or disposable logs in snapshots.
- [ ] Commit `feat(v5.1): add resumable runner and verified snapshots`.

### Task 4: Preflight, integrity analysis, CSV/SVG, and Chinese report

**Files:**
- Create: `src/research/abcd_ladder_v5_1_analysis.py`
- Create: `src/research/abcd_ladder_v5_1_report.py`
- Create: `src/research/abcd_ladder_v5_1_preflight.py`
- Create: `scripts/preflight_abcd_ladder_v5_1.py`
- Create: `scripts/summarize_abcd_ladder_v5_1.py`
- Test: `tests/test_abcd_ladder_v5_1_analysis.py`
- Test: `tests/test_abcd_ladder_v5_1_report.py`

**Interfaces:**
- Analysis consumes only replay-validated V5.1 results and emits route/version/family/pairwise/color/usage summaries.
- Report consumes the summary and V5 history; SVG renders every fixed challenger version against each opponent family.

- [ ] Write failing tests with synthetic results for correct win/draw/loss points, actual-game denominators, paired color counts, missing rungs, and report/summary agreement.
- [ ] Add an explicit regression test proving the V5 report's A–C, A–D and first-listed B–D/C–D point fields are re-derived from win/draw counts.
- [ ] Run focused analysis/report tests and confirm expected failures.
- [ ] Implement strict game replay, exact-once data checks, group and route sums, descriptive confidence intervals, usage metrics, CSVs, gradient SVG and Chinese narrative with an audited V5 score correction.
- [ ] Implement resource/preflight estimation from actual rung benchmarks, observed decision counts, 39-turn strict ceilings, V5 reuse subtraction, CPU/memory/disk snapshots, and history comparison.
- [ ] Test report rendering, XML validity, deterministic outputs, and every count/rate identity; commit `feat(v5.1): add validated analysis and report`.

### Task 5: Backup drill, full ladder execution, final verification, and push

**Files:**
- Create: `research/v5_1/results/*`
- Create: `research/v5_1/REPORT.md`
- Create: `research/v5_1/figures/ladder_gradient.svg`
- Modify: scripts/tests only if a preflight finding requires a tested correction.

**Interfaces:**
- Uses the committed frozen config and schedule, validated V5 reuse manifest, runner, preflight, analysis and snapshot interfaces above.

- [ ] Run cloud resource checks and a complete per-budget pilot; calculate expected wall time, configured full-board ceiling, longest group and storage before official games.
- [ ] Validate a Drive upload/download/SHA/empty-folder recovery exercise and upload the initial official V5.1 recovery snapshot into a dedicated V5.1 folder.
- [ ] Run every unique group with at least one active route until the route graph is exhausted; checkpoint at every group and perform a Drive snapshot rotation at least every 24 hours.
- [ ] Replay all result records, confirm 28 V5 reuse groups or document exact nonmatches, validate all 66 route trees, 168 max matchup IDs, score invariants, group/color pairs, CSVs, SVG, report claims, and Drive's single current snapshot.
- [ ] Run `PYTHONPATH=. pytest -q` and `git diff --check`; commit complete artifacts, push `research/abcd-ladder-v5-1`, and verify remote HEAD equals local commit.

## Global Verification

- Baseline: `PYTHONPATH=. pytest -q` on V5 before changes.
- Task tests: the focused pytest paths named above, after each red/green step.
- Final: `PYTHONPATH=. pytest -q`, V5.1 replay validator with `require_complete=True`, recalculated point assertions, SVG XML parse, local/Drive SHA-256 checks, `git diff --check`, and `git ls-remote` exact commit comparison.
