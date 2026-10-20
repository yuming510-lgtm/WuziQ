# V5 ABCD Ladder Challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible, resumable 9x9 V5 ladder tournament with verified Drive recovery, validated data, analysis, figures and report.

**Architecture:** Keep the V4C engine untouched and add a dedicated V5 runner, validator/analyzer and report builder. Persist one fsynced JSONL record per complete game; reconstruct route state from complete two-game groups so restart cannot count partial groups. Store only V5 snapshot payloads in a dedicated Drive folder.

**Tech Stack:** Python 3.12, pytest, existing Gomoku AI/search modules, JSON/JSONL, CSV and standalone SVG.

**Spec:** `docs/superpowers/specs/2026-10-20-abcd-ladder-v5-design.md`

## Global Constraints

- 9x9 board; A natural rule-selector time; B/C/D budgets are exactly 50ms, 200ms, 1s, 5s, 10s, 20s, 50s.
- A vs each B/C/D rung is two games from the same opening, with sides swapped.
- Internal B/C/D pairings share 50ms-vs-50ms and then independently climb against the other family fixed at 50ms.
- Promotion threshold is 1 point from two games; failure ends only that directed route.
- Maximum 60 groups / 120 games; no historical game is counted.
- B and C use the same existing search except C's Alpha-Beta switch; D uses existing V4C MCTS and tree reuse.
- No neural-network training, learned weights, or Minimax substitution for D.
- Per-game IDs are unique, records are replay-validated and fsynced before progress updates.
- Drive must pass upload, download, SHA-256 and empty-directory restore before formal play; only the latest verified V5 snapshot remains after each update.

## Review Focus

- Shared baseline is counted once while each active direction makes an independent decision — test both one-sided progression and two-sided draw progression.
- A previously failed route may be a fixed 50ms opponent for the other active direction without being reactivated — test the state transition.
- A crash after the first game of a group cannot trigger promotion — test resume from one persisted game.
- Paired colors use the same exact opening and no duplicate IDs — test schedule invariants.
- Invalid or unfinished games cannot affect results — test replay validation and rejection.

---

### Task 1: Freeze V5 configuration and opening catalog

**Files:**
- Create: `configs/abcd_ladder_v5.json`
- Test: `tests/test_abcd_ladder_v5.py`

**Interfaces:**
- Produces config consumed by `load_config` and `build_schedule`.

- [ ] Add tests for exactly 12 legal, nonterminal openings, seven exact budget rungs, and the maximum 60 unique groups / 120 unique games.
- [ ] Run the focused test to confirm it fails because V5 config/loader do not exist.
- [ ] Add the frozen config, including source hashes for `src/ai.py`, `src/gomoku.py`, and `src/research/mcts_d2.py`.
- [ ] Verify the config loader and schedule tests pass.

### Task 2: Implement execution, route state and durable resume

**Files:**
- Create: `src/research/abcd_ladder_v5.py`
- Create: `scripts/run_abcd_ladder_v5.py`
- Modify: `tests/test_abcd_ladder_v5.py`

**Interfaces:**
- `load_config(source) -> dict`
- `build_schedule(config) -> list[dict]`
- `run_game(spec, config, implementation_commit) -> dict`
- `derive_routes(records, config) -> dict`
- `run_experiment(config_source, output_dir, implementation_commit=...) -> dict`

- [ ] Add failing tests for route promotion/failure, shared baseline accounting, color-swapped IDs, and natural-time A.
- [ ] Add failing restart test where one game of a pair is persisted and only the other is run on resume.
- [ ] Implement game execution with B/C deadlines and persistent V4C D2 agent; fsync each complete game before updating progress.
- [ ] Implement route reconstruction from full pairs only; enforce duplicate and unexpected ID rejection.
- [ ] Run focused tests and the full existing suite.

### Task 3: Implement replay validation, aggregation and report artifacts

**Files:**
- Create: `src/research/abcd_ladder_v5_analysis.py`
- Create: `src/research/abcd_ladder_v5_report.py`
- Create: `scripts/summarize_abcd_ladder_v5.py`
- Create: `tests/test_abcd_ladder_v5_analysis.py`

**Interfaces:**
- `validate_game_record(record, config, spec=None) -> dict`
- `validate_records(records, config, require_complete=False) -> dict`
- `summarize_records(records, config) -> dict`
- `write_analysis(results_dir, output_dir) -> dict`
- `build_report(analysis, results_dir, output_dir) -> Path`

- [ ] Add failing tests for illegal replay, unfinished group exclusion, score threshold, side metrics and route-gradient rendering.
- [ ] Implement CSV/JSON metrics and a standalone SVG budget ladder chart.
- [ ] Build a Chinese report with explicit two-game uncertainty and historical comparisons kept separate.
- [ ] Validate reports against synthetic test records and later against all formal data.

### Task 4: Preflight and Drive recovery drill

**Files:**
- Create: `src/research/abcd_ladder_v5_preflight.py`
- Create: `src/research/abcd_ladder_v5_snapshot.py`
- Create: `scripts/preflight_abcd_ladder_v5.py`
- Create: `scripts/create_abcd_ladder_v5_snapshot.py`
- Create: `scripts/restore_abcd_ladder_v5_snapshot.py`
- Create: `tests/test_abcd_ladder_v5_snapshot.py`

- [ ] Measure environment CPU, memory and disk plus representative A/B/C/D decisions across rungs.
- [ ] Estimate formal and full-board worst-case wall time from actual V4 history and fresh measurements.
- [ ] Create a dedicated V5 Drive folder; upload/download/hash-check a temporary recovery fixture and restore it into an empty directory.
- [ ] Upload and restore the initial V5 snapshot before official play.

### Task 5: Run, validate and publish V5

**Files:**
- Create: `research/v5/results/*`, `research/v5/figures/*`, `research/v5/REPORT.md`

- [ ] Run all routes to completion; snapshot after each completed group and at least every 24 hours.
- [ ] Replay every completed game and verify exactly two games per played group.
- [ ] Write CSV/JSON summaries, SVG trend figure and full Chinese report.
- [ ] Run verification commands, commit results, push `research/abcd-ladder-v5`, and verify remote head.
