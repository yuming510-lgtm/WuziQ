# WuziQ G V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether F2-initialized self-play PUCT-64 training produces repeatable neural-network strength growth within a 2 CPU-hour registered budget.

**Architecture:** Keep V1 algorithms and results immutable. Add a V2 schedule, three matched learning-rate candidates, resumable per-candidate training, development-only champion selection, one sealed paired blind schedule, and V2-only reporting/snapshot artifacts. Reuse V1 PUCT, self-play, model, training, evaluation, replay, and safe archive primitives.

**Tech Stack:** Python 3.12, PyTorch CPU, NumPy, pytest, JSON/JSONL, matplotlib, Google Drive connector, git worktree.

**Spec:** `docs/superpowers/specs/2026-11-01-wuziq-g-v2-design.md`

## Global Constraints

- Board is 9×9 free Gomoku; network remains 32 channels, 2 residual blocks; training and evaluation search use exactly 64 PUCT simulations per move.
- G V2 trains only from its own complete self-play game outcomes and visit distributions; teacher labels, F supervised data, and artificial tactical rewards remain zero.
- Candidate LRs are exactly `1e-4`, `3e-5`, and `1e-3`, all initialized from F2 SHA-256 `50164ad9b139e0c47c1b1af7497f670c26e9864a81c10c02de0ff4426b2d0257`.
- Per-candidate state budget is 20,000 unless the preflight freezes a smaller equal budget under the 2 CPU-hour cap.
- Development-only promotion and the V2 blind opening schedule are registered before training; blind results cannot select or tune a model.
- Every completed game and checkpoint is validated and atomically persisted; experiment IDs are unique and resume is idempotent.
- V1 artifacts and `research/g_v1` are read-only inputs; V2 writes only under `research/g_v2` plus new V2 code/tests/docs.
- Drive snapshots are stored in a V2-only folder and follow 1, 2, 4, 6, 10, 16, 26-hour cumulative intervals; a prior snapshot is deleted only after a new archive passes download, SHA-256, and empty-directory restore checks.

## Review Focus

- **Perspective leakage in self-play value labels:** assert winner-relative targets are correct for both colors and final game replay matches outcome.
- **Quota crossing and duplicate samples during resume:** simulate a game that crosses a stage boundary, reload state, and verify exact budget prefix, update index, and unique IDs.
- **Candidate leakage through shared state:** assert each LR gets the same F2 tensor weights and schedule but independent optimizer, RNG, replay, and generated game IDs.
- **Development/ blind schedule overlap:** assert no canonical opening digest or group ID overlaps V1, train, dev, and blind sets.
- **Blind outcomes influencing selection:** assert selection API only consumes development logs and fails closed when called with blind records.

---

### Task 1: V2 opening schedule and frozen configuration

**Files:**
- Create: `src/research/alphazero_g/v2_schedule.py`
- Create: `research/g_v2/config/experiment.json`
- Create: `research/g_v2/config/openings.json`
- Test: `tests/test_alphazero_g_v2_schedule.py`

**Interfaces:**
- Produces `build_v2_schedule(seed: int, train_groups: int, development_groups: int, blind_groups: int, excluded_openings: Iterable[list[dict[str, Any]]]) -> dict[str, Any]` and `validate_v2_schedule(schedule: dict[str, Any], excluded_openings=()) -> None`.
- Produces canonical symmetry-invariant opening SHA-256s, 128 train groups, 8 development groups, 20 blind groups, all four-ply legal/nonterminal and disjoint from V1.

- [ ] **Step 1: Write failing schedule tests** for deterministic output, legal alternation, counts, unique symmetry-canonical digests, V1 exclusion, and checksum rejection.
- [ ] **Step 2: Run** `pytest tests/test_alphazero_g_v2_schedule.py -q`; confirm missing-module failures.
- [ ] **Step 3: Implement** deterministic generation, normalization, and schedule hashing; freeze seed `510701` and all group records.
- [ ] **Step 4: Run** the schedule tests; expected all pass and repeated generation yields identical SHA-256.
- [ ] **Step 5: Save** the checked schedule and initial proposed experiment configuration without registering compute budgets yet.

### Task 2: Preflight and budget registration

**Files:**
- Create: `src/research/alphazero_g/v2_preflight.py`
- Create: `scripts/preflight_g_v2.py`
- Create: `scripts/register_g_v2_budget.py`
- Test: `tests/test_alphazero_g_v2_preflight.py`

**Interfaces:**
- Produces `run_v2_preflight(root: Path, schedule: dict, seed: int) -> dict[str, Any]` measuring 16/32/64 simulation positions, complete 64-simulation self-play games, optimizer throughput at all three LRs, evaluation games, RSS, CPU, disk, and available device.
- Produces `register_v2_budget(preflight: dict, proposed: dict, schedule: dict) -> dict[str, Any]` with optimistic/central/conservative CPU estimates and a finite cap no greater than 7,200 CPU seconds.

- [ ] **Step 1: Write failing tests** for missing budget measurements, absent 64-simulation data, unequal LR quotas, over-cap conservative estimate, and exact milestone retention.
- [ ] **Step 2: Run** `pytest tests/test_alphazero_g_v2_preflight.py -q`; confirm failure before implementation.
- [ ] **Step 3: Implement** end-to-end measurements with `torch.set_num_threads(1)` and fixed seed `510702`; write raw preflight output atomically.
- [ ] **Step 4: Run the real preflight**, including at least one full game and training batches at each LR; estimate 20k/arm and evaluation separately.
- [ ] **Step 5: Register and hash** the equal candidate quotas, reachable milestone nodes, per-state update ratio 1:16, PUCT64, and blind/development schedule before any formal training.

### Task 3: Resumable three-candidate training runner

**Files:**
- Create: `src/research/alphazero_g/v2_runner.py`
- Create: `scripts/run_g_v2.py`
- Test: `tests/test_alphazero_g_v2_runner.py`

**Interfaces:**
- Produces `run_v2_candidate(root: Path, config: dict[str, Any], lr_id: str, stop_after_games: int | None = None) -> dict[str, Any]` for `lr_1e4`, `lr_3e5`, and `lr_1e3`.
- Produces one fsynced JSONL game log per candidate and an atomic exact-resume state containing model, optimizer, scheduler, replay, Python/NumPy/Torch RNGs, opening/game cursor, progress, and update count.
- Candidate quotas and stage nodes derive only from the registered configuration; each stage triggers `floor(total_budgeted_states / 16)` cumulative batch-16 updates.

- [ ] **Step 1: Write failing tests** for independent candidate initialization, exact per-game persistence, state-prefix accounting across stage boundaries, update cadence, interruption/resume equivalence, duplicate IDs, and a 7-ply terminal tail excluded from budget/replay.
- [ ] **Step 2: Run** `pytest tests/test_alphazero_g_v2_runner.py -q`; confirm missing runner failures.
- [ ] **Step 3: Implement** the V2 runner by composing V1 `initialize_arm`, `GTrainer`, replay/selfplay and safe append routines without changing those V1 modules.
- [ ] **Step 4: Run** tests with tiny budgets; compare resumed and uninterrupted candidate model tensors, optimizer count, game IDs, and progress exactly.
- [ ] **Step 5: Run a short three-LR smoke experiment** into a temporary directory and replay every logged game.

### Task 4: Development evaluation and champion selection

**Files:**
- Create: `src/research/alphazero_g/v2_selection.py`
- Create: `scripts/evaluate_g_v2_development.py`
- Test: `tests/test_alphazero_g_v2_selection.py`

**Interfaces:**
- Produces `run_v2_development(root: Path, checkpoint: Path, candidate_id: str, node: int, schedule: dict, seed: int) -> list[dict[str, Any]]` using pure network and PUCT64 versus frozen F2, A, and B (B fixed at 50 ms per move) on identical development opening/color pairs.
- Produces `select_v2_champion(results: list[dict[str, Any]], incumbent_id: str, incumbent_checkpoint: Path) -> dict[str, Any]`, accepting development records only and enforcing the preregistered promotion thresholds and tie-break.

- [ ] **Step 1: Write failing tests** for paired 8-group score, bootstrap CI, color split, all promotion thresholds, incumbent retention, deterministic tie-break, and blind-record rejection.
- [ ] **Step 2: Run** `pytest tests/test_alphazero_g_v2_selection.py -q`; confirm failures.
- [ ] **Step 3: Implement** idempotent development match IDs, reuse `play_evaluation_match`, record all search/timing stats, and atomically persist selection history. A candidate that becomes incumbent later challenges the checkpoint it displaced; other candidates challenge the current incumbent.
- [ ] **Step 4: Run** tests with synthetic results plus real F2-vs-F2 and short candidate matches; validate all games through existing replay validation.

### Task 5: Final paired blind evaluation and statistical summaries

**Files:**
- Create: `src/research/alphazero_g/v2_blind.py`
- Modify: `src/research/alphazero_g/evaluation.py` only if a minimal generic opponent/model adapter is required; retain V1 behavior/tests.
- Test: `tests/test_alphazero_g_v2_blind.py`

**Interfaces:**
- Produces `run_v2_blind(root: Path, config: dict[str, Any], schedule: dict[str, Any], checkpoints: dict[str, Path]) -> list[dict[str, Any]]` for F2 network, F2 PUCT64, selected G network, selected G PUCT64, and A/B comparisons; uses all 20 paired blind openings exactly once per registered comparison (560 games before duplicate elimination).
- Produces `summarize_v2_results(matches: list[dict[str, Any]]) -> dict[str, Any]` with W/D/L, points, paired opening bootstrap 95% intervals, color-specific points, compute, and training-before/after deltas.

- [ ] **Step 1: Write failing tests** for the 560-game schedule, swapped colors, unique IDs, F2/G checkpoint identity, paired bootstrap, color split, incomplete game rejection, and blind immutability.
- [ ] **Step 2: Run** `pytest tests/test_alphazero_g_v2_blind.py -q`; confirm failures.
- [ ] **Step 3: Implement** the fixed comparisons and deterministic game seeds; never pass blind records into the selector.
- [ ] **Step 4: Run** on a temporary mini schedule, verify color pairs and complete game replay, then freeze full blind checksum.

### Task 6: V2 snapshot, restore, report and figures

**Files:**
- Create: `src/research/alphazero_g/v2_snapshot.py`
- Create: `scripts/create_g_v2_snapshot.py`
- Create: `scripts/restore_g_v2_snapshot.py`
- Create: `scripts/report_g_v2.py`
- Test: `tests/test_alphazero_g_v2_snapshot.py`
- Test: `tests/test_alphazero_g_v2_report.py`

**Interfaces:**
- Produces V2-only archive manifest with code commit, frozen F2 SHA, config/schedule hashes, all candidate checkpoint/state/log files, and exact resume progress; validates archive traversal, duplicate paths, hashes, and an empty-directory restore.
- Produces `build_v2_report(root: Path) -> dict[str, Any]` and `render_v2_report(root: Path, summary: dict[str, Any]) -> None` for Chinese `REPORT.md`, CSV/JSON and three registered plots.

- [ ] **Step 1: Write failing tests** for V1/V2 snapshot separation, manifest tampering, empty restore, required-file completeness, score math, and plotted milestone data consistency.
- [ ] **Step 2: Run** the two new test files; confirm failures.
- [ ] **Step 3: Implement** allowlisted V2 recovery only; require upload/download/hash/restore connector checks before deleting an older V2 Drive snapshot.
- [ ] **Step 4: Implement** report generation with actual reached nodes and explicit `未执行` for unbudgeted nodes; produce state/game, CPU, and raw-network-vs-PUCT curves.
- [ ] **Step 5: Validate** report values directly against source JSONL and verify all three plot files exist and have nonzero dimensions.

### Task 7: Registered run, final validation, and GitHub delivery

**Files:**
- Modify: `research/g_v2/config/experiment.json` only through the registration script before formal execution.
- Create: all registered artifacts under `research/g_v2/`.
- Test: all G V2 tests and existing AlphaZero G tests.

- [ ] **Step 1: Verify** current commit remains `6e2bf48aa6b630c0fa4f20284ceb3b5aab295080`, F2 hash matches, Drive upload/download/delete/read permissions work in a V2-only folder, and an initial tiny recovery archive restores in an empty directory.
- [ ] **Step 2: Freeze** budget/schedule/config checksums and output estimate; compare conservative CPU estimate to the 2-hour cap.
- [ ] **Step 3: Execute** all registered candidates, stage evaluations, champion selection, and final blind schedule; persist each complete game and stage before continuing.
- [ ] **Step 4: Run** `pytest tests/test_alphazero_g_v2_*.py tests/test_alphazero_g_*.py -q` and the repository suite with only known branch-name-specific V6.2/V6.3 tests excluded.
- [ ] **Step 5: Audit** every game replay, ID uniqueness, color swap, sample/outcome integrity, equal compute quotas, blind unopened-before-selection metadata, reported totals, Drive latest snapshot hash, and selected model load via the future G loader.
- [ ] **Step 6: Commit and push** the independent `research/alphazero-g-v2` branch; verify remote branch SHA equals local and remote report/checkpoint files are present.
