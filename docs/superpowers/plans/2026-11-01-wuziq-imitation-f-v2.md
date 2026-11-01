# WuziQ F V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Continue F V1 supervised training with targeted tactical and F1-vs-A state coverage, then measure F2 against A on a paired 20-opening blind match set.

**Architecture:** Keep F1 untouched as the starting checkpoint and baseline. Add an F1-vs-A DAgger collector and exact train-only tactical labels, oversample tactical training rows without touching validation/blind states, extend the existing trainer for hash-checked warm starts and candidate snapshots, and run identical paired A match schedules for F1/F2.

**Tech Stack:** Existing Python 3.12, PyTorch CPU/GPU auto-detection, `GomokuBoard`, V1 `PolicyValueNet`, `load_f_checkpoint`, `predict_policy_value`, C/E5 search and A engine.

**Spec:** `docs/superpowers/specs/2026-11-01-wuziq-imitation-f-v2-design.md`

## Global Constraints

- Base commit is exactly `d33e653a29a5f24aecf6c5e25d09b137a68f4c01` on `research/imitation-learning-f-v2`.
- F1 data, report, and checkpoint are immutable controls; A's engine and code remain unchanged.
- F2 inference is neural policy argmax plus empty-cell masking only; no tactical move override.
- New annotations and tactical examples must stay outside V1 validation/blind states; final opening groups are disjoint from training and development.
- Training is capped at two runs, each 30–60 epochs with validation early stopping; no G RL.
- Final test has at least 20 paired opening groups and 40 games per model, with fixed seeds and no post-test tuning.
- Every checkpoint/data file is hashed; every game replays legally; Drive retains one verified latest V2 snapshot.

## Review Focus

- Corrupt or mismatched warm-start resume must fail before optimizer updates; test changed dataset/config/checkpoint hashes.
- A tactical label must point to an actually legal immediate win or unique forced block; test both and a false-threat/dual-threat case.
- Added DAgger rows that match validation or blind canonical states must be rejected or excluded; test the exact boundary.
- Network match actions must reproduce masked policy argmax at every F1/F2 ply; test a board where the network's chosen legal move is not the winning move.
- Pairing and report statistics must preserve the same opening and opposite colors, count draws as half points, and resample whole opening groups for CIs; test hand-computed results.

---

### Task 1: Frozen schedules, V1 loss analysis, and performance preflight

**Files:**
- Create: `src/research/imitation_f/v2_experiment.py`
- Create: `scripts/preflight_f_v2.py`
- Create: `research/f_v2/config/experiment.json`, `research/f_v2/results/loss_analysis.json`, `research/f_v2/results/preflight.json`
- Test: `tests/test_f_v2_experiment.py`

**Interfaces:**
- `build_experiment_plan(seed, train_count, development_count, blind_count, excluded_groups) -> dict` returns deterministic disjoint opening schedules and a schedule hash.
- `analyze_f1_a_losses(records) -> dict` reports missed direct wins and missed unique blocks from the two frozen F1-A losses.
- Preflight measures 100 CNN updates, representative 200ms C/E5 annotations, and paired F1-A game throughput; it prints optimistic/central/conservative task estimates.

- [x] Write and run RED tests for legal/disjoint schedules and exact loss diagnosis.
- [x] Freeze 16 training, 8 development, 20 final blind groups and seeds; exclude V1 split and old evaluation groups.
- [x] Run resource and representative speed preflight, record estimate, then proceed directly.

### Task 2: F1-vs-A DAgger collection and train-only tactical balancing

**Files:**
- Create: `src/research/imitation_f/v2_data.py`
- Create: `scripts/collect_f_v2_data.py`, `scripts/prepare_f_v2_dataset.py`
- Create: `research/f_v2/results/dagger_games.jsonl`, `dagger_annotations.jsonl`
- Create: `research/f_v2/data/train_tactical_30.jsonl`, `train_tactical_40.jsonl`, `validation.jsonl`, `blind_test.jsonl`, `manifest.json`
- Test: `tests/test_f_v2_data.py`

**Interfaces:**
- `collect_dagger_data(f1_checkpoint, openings, output_dir, teacher_budget_seconds, seed) -> dict` plays F1 vs frozen A and labels F turns with C/E5 while atomically persisting each game/annotation.
- `exact_tactical_targets(rows) -> (win_rows, block_rows)` only returns exact immediate wins or exactly-one opponent-win blocks.
- `build_weighted_training_rows(v1_train, dagger_rows, val_keys, blind_keys, tactical_share, seed) -> (rows, manifest)` uses training data only and targets a balanced tactical share.

- [x] Write and run RED tests for no held-out leakage, legal exact labels, deterministic oversampling and ID dedupe.
- [x] Collect 16 paired F1-vs-A training opening groups with C/E5 labels; value labels remain tied to replayed complete outcomes.
- [x] Build 30% and 40% tactical schedules; preserve each V1 validation/blind file byte-for-byte and hash it.
- [x] Verify unique counts, class balance and state intersections before training.

### Task 3: Hash-verified warm-start trainer and bounded model candidates

**Files:**
- Modify: `src/research/imitation_f/training.py`
- Create: `scripts/train_f_v2.py`, `research/f_v2/config/training_*.json`, `research/f_v2/checkpoints/`
- Modify: `tests/test_f_v1_training.py`
- Create: `tests/test_f_v2_training.py`

**Interfaces:**
- Extend `train(config, dataset, checkpoint_dir, resume=True, initial_checkpoint=None)` to validate warm-start architecture/hash and resume fingerprints.
- Save candidates at fixed 10-epoch intervals; all states include optimizer, scheduler, RNG, sampler, accumulated elapsed time, dataset/config/init hashes.

- [x] Add RED tests for exact F1 weight initialization, mismatch rejection, candidate checkpoint loading and minimum 30-epoch behavior.
- [x] Implement warm-start and candidate snapshots without changing the existing F1 checkpoint/API.
- [x] Run at most two configurations (30% then, only if needed, 40% tactical sampling), each max 60 epochs, patience 12 after minimum 30.
- [x] Evaluate candidate checkpoints on the registered 8-group development schedule; select by paired score rate, then validation policy loss as tie-breaker.

### Task 4: Pure-network F1/F2 match runner and paired statistics

**Files:**
- Create: `src/research/imitation_f/v2_matches.py`
- Create: `scripts/play_f_v2_matches.py`, `scripts/evaluate_f_v2.py`
- Create: `research/f_v2/results/development_matches.jsonl`, `blind_matches.jsonl`, `metrics.json`
- Test: `tests/test_f_v2_matches.py`

**Interfaces:**
- `run_paired_matches(checkpoint, version, opponent, schedule, output_path) -> list[dict]` uses only existing neural policy selection and legal masking.
- `verify_network_actions(records, checkpoint) -> dict` replays each F action and checks it equals the checkpoint's masked policy argmax.
- `paired_summary(records, bootstrap_seed, replicates=10000) -> dict` emits W/D/L, points, score rate, black/white splits, cluster-bootstrap 95% CIs and paired F2−F1 interval.

- [x] Add RED tests for pure-network action replay, color pairing, draw points and hand-calculated paired intervals.
- [x] Evaluate/select models only on frozen development schedule; freeze one F2 checkpoint before final test.
- [x] Run F1 and F2 on the same 20 final openings with both colors (40 each), only after selection.
- [x] Replay and hash all 80 games, verify no duplicate IDs, schedule mismatch or rule errors.

### Task 5: Comparisons, recovery, report, tests and push

**Files:**
- Create: `scripts/create_f_v2_snapshot.py`, `scripts/restore_f_v2_snapshot.py`, `scripts/build_f_v2_report.py`
- Create: `research/f_v2/REPORT.md`, curves, tables, integrity and recovery state
- Test: `tests/test_f_v2_snapshot.py`, full applicable suite

- [x] Verify Drive read/write/delete, snapshot round-trip SHA and empty-directory recovery in a dedicated V2 folder.
- [x] Publish preflight estimate and final Chinese report with F1/F2 policy, tactics, value, latency and A match results.
- [x] Run applicable project tests; preserve known legacy V6 branch-guard exclusions and list them in the report.
- [x] Commit/push only V2 branch; verify remote SHA, report and checkpoint paths.


### Execution correction log

- DAgger exact win/block labels were audited against the D4 canonical board. A mismatch was fixed by converting source actions into canonical coordinates before storage; a non-symmetric fixture verifies this conversion.
- The pre-correction candidate run and its 80 blind games are archived under `research/f_v2/results/precorrection/` and excluded from the final score. Since the 200ms C/E5 teacher labels were regenerated and their bytes changed, the final model was retrained and evaluated once on a fresh 20-opening schedule (SHA-256 recorded in `config/experiment.json`).
- Final paired blind score: F1 8-18-14 (42.5%); F2 17-19-4 (66.25%). F2's paired opening-bootstrap 95% interval is 57.5%–75.0%, while its white score is 47.5%; report describes this as a schedule-level result with a remaining color-balance limitation.
