# V4 ABC Search Compute Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, resumable seven-budget search benchmark for A/B/C on fixed 9×9 positions and publish raw measurements, plots, and a Chinese report on an independent V4 branch.

**Architecture:** Keep the existing shared B/C engine and add per-completed-depth telemetry. A research runner loads a frozen matrix, writes one fsynced JSONL result per model/position/budget, and can resume by stable run ID. Separate analysis and snapshot helpers validate score agreement, summarize metrics, plot trends, and verify recovery archives.

**Tech Stack:** Python 3.12, pytest, matplotlib, JSON/JSONL, zipfile/SHA-256, Google Drive connector, git.

**Spec:** `docs/research/abc-search-compute-v4-spec.md`

## Global Constraints

- Use exactly 9×9 positions and budgets 0.05, 0.2, 1, 5, 10, 20, and 50 seconds.
- Freeze at least 12 nonterminal positions spanning opening, midgame, late game, attack, and defense.
- A uses the existing rule selector and does not wait to consume its budget.
- B/C share candidate generation/order, evaluation, and search logic; only C enables alpha-beta pruning.
- Use iterative deepening up to remaining board cells; preserve the previous completed layer on timeout.
- Persist all 252 matrix records with unique stable run IDs and support validated resume.
- Do not run AI match results or claim statistical strength conclusions.
- Do not change main or V1–V3.

## Review Focus

- Timeout during an incomplete layer must preserve the previous complete move, score, and depth; test a short-budget search with a recorded complete iteration.
- A must return promptly under all budget labels; test that no artificial wait occurs.
- A frozen position must reject illegal, alternating-order, or already-terminal move histories; test each invalid condition.
- Resume must reject duplicate or off-matrix run IDs and skip only verified completed IDs; test both behavior and corruption handling.
- B/C score comparisons must include only matching complete depths; test that a deliberately altered trace fails validation.

---

### Task 1: Per-depth B/C telemetry and timeout-safe iterations

**Files:**
- Modify: `src/ai.py`
- Test: `tests/test_ai.py`

**Interfaces:**
- Produces `DepthIteration(depth, root_value, move, nodes, cutoffs, elapsed_seconds)` and `SearchResult.depth_iterations`.
- Existing B/C search entry points remain unchanged; run configurations may set `max_depth` to remaining cells.

- [x] **Step 1:** Add failing tests for completed per-depth rows, matching B/C root scores, and preservation of the last complete depth under a very short deadline.
- [x] **Step 2:** Run those tests and confirm the missing telemetry fails.
- [x] **Step 3:** Add immutable depth rows; count each iteration independently and include partial timed-out work only in total counters.
- [x] **Step 4:** Run targeted AI tests and confirm all telemetry/deadline assertions pass.
- [x] **Step 5:** Commit as `feat: record iterative search depth metrics`.

### Task 2: Frozen positions, matrix runner, and resumable raw records

**Files:**
- Create: `configs/abc_search_compute_v4.json`
- Create: `src/research/abc_search_v4.py`
- Create: `tests/test_abc_search_v4.py`

**Interfaces:**
- Produces `load_config(path)`, `run_matrix(config_path, results_dir, max_new_runs=None)`, and fixed `run_id` generation.
- The config contains exactly 12 or more validated 9×9 histories and seven budget labels.

- [x] **Step 1:** Add failing tests for fixed-position legality, the full 252-ID schedule, fsynced unique appends, and resume skipping.
- [x] **Step 2:** Run the focused tests and verify expected missing-interface failures.
- [x] **Step 3:** Freeze twelve position histories and implement validation, deterministic seeds, per-run timing, checkpointing, and resume.
- [x] **Step 4:** Run focused runner tests, including board restoration and malformed/duplicate records.
- [x] **Step 5:** Commit as `feat: add resumable ABC compute benchmark`.

### Task 3: Preflight, snapshot drill, result validation, plots, and report

**Files:**
- Create: `src/research/abc_search_v4_analysis.py`
- Create: `src/research/abc_search_v4_snapshot.py`
- Create: `tests/test_abc_search_v4_analysis.py`
- Create: `tests/test_abc_search_v4_snapshot.py`

**Interfaces:**
- Produces analysis/consistency validation, metrics tables and charts; produces `create_snapshot(...)` and `restore_snapshot(...)` with SHA-256 verification.
- Report consumes the runner's `search_runs.jsonl` and references only complete verified records.

- [x] **Step 1:** Add failing tests for root-score agreement checks, summary metric definitions, and checksum-verified restore.
- [x] **Step 2:** Run focused tests and confirm failures before implementation.
- [x] **Step 3:** Implement analysis and minimal recovery archive for config, progress, run data, and code version.
- [x] **Step 4:** Run tests and execute a real Drive upload/read/hash/restore/delete drill on a tiny archive.
- [x] **Step 5:** Commit as `feat: analyze and recover ABC benchmark runs`.

### Task 4: Preflight-calibrated full matrix and V4 deliverables

**Files:**
- Create: `research/v4/results/search_runs.jsonl`
- Create: `research/v4/results/progress.json`
- Create: `research/v4/results/preflight.json`
- Create: `research/v4/results/summary.json`
- Create: `research/v4/figures/*.png`
- Create: `research/v4/results/report_zh.md`
- Modify: `README.md` only if needed for reproducibility instructions

**Interfaces:**
- Uses Task 2's stable run schedule and Task 3's validators/plots/snapshot functions.

- [x] **Step 1:** Record visible CPU, memory, cgroup, and disk resources; run representative B/C budget probes and estimate full wall time/storage.
- [x] **Step 2:** Validate the Drive recovery drill, then run all uncompleted IDs sequentially; persist each row immediately.
- [x] **Step 3:** Validate exactly 252 unique records, legal moves, depth traces, and every matching B/C completed-depth root score.
- [x] **Step 4:** Generate seven-budget tables, trends, Chinese report, and rerun the full repository suite.
- [x] **Step 5:** Commit data and report as `research: publish ABC compute expansion V4`.

### Task 5: Final review and independent V4 push

**Files:**
- Review all V4 changes against the V3 merge base.

- [x] **Step 1:** Build the review package and request a fresh whole-branch review.
- [x] **Step 2:** Fix all Important findings test-first; log the deferred Minor timestamp issue.
- [ ] **Step 3:** Run the final full suite and `git status`/history checks.
- [ ] **Step 4:** Push `research/abc-compute-v4` and verify the remote branch resolves to the final commit SHA.
