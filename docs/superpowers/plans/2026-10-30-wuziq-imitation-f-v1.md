# WuziQ F V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Train and deliver a small 9×9 policy/value CNN F from validated C/E5 imitation labels, with an initialization interface suitable for a later G stage.

**Architecture:** Extract and replay official historical 9×9 games, deduplicate D4-equivalent positions, and assign whole opening components to train/validation/blind-test. If history alone cannot reach roughly 10k/1k/1k, generate capped mixed A/B/C/E5 games and label positions using frozen C/E5 at a preregistered moderate budget. Train a CPU/GPU auto-device PyTorch residual CNN with atomic resumable state, then evaluate and report without changing historical engines.

**Tech Stack:** Python 3.12, existing `GomokuBoard`, existing C search and `V6Agent("E5")`, PyTorch CPU/GPU, NumPy, pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-30-imitation-learning-f-v1-design.md`

## Global Constraints

- Baseline commit is exactly `ce23c5691237b508ad39aa2765a0682837132765` on `research/imitation-learning-f-v1`.
- Preserve A/B/C/D/E5 and all V1–V6.3 source, data and history.
- 9×9 official rules; C/E5 teacher searches use frozen implementations.
- Policy output is 81 cells and masks occupied cells before normalization or selection.
- Split by whole opening/shared-symmetry group; no canonical state may appear across splits.
- Target about 10,000 train positions and at least 1,000 each validation and blind test; document any adjustment.
- No G self-play RL, external training data, or claim of statistical significance from the small match set.
- Every completed generated game, training checkpoint and recovery snapshot is idempotent and checksummed.

## Review Focus

- Illegal, terminal, malformed or out-of-turn historical records must be rejected before training; test corrupted and valid full-board/draw records.
- Rotations/reflections and identical positions from different source games must never cross splits; test D4 canonicalization and component grouping.
- Duplicate teachers can disagree; test soft-label normalization and preserve label-conflict counts.
- Full boards have no legal policy action; reject gracefully rather than returning an occupied cell.
- Resume must reproduce the next optimizer update and sampling order; test save/restore against uninterrupted deterministic training.

---

### Task 1: Historical data audit, canonical dataset and split

**Files:**
- Create: `src/research/imitation_f/dataset.py`
- Create: `scripts/prepare_f_v1_dataset.py`
- Create: `tests/test_f_v1_dataset.py`
- Create: `research/f_v1/data/manifest.json` and train/validation/blind-test JSONL files

**Interfaces:**
- `validate_game_record(record: dict) -> None` replays 9×9 legality and terminal consistency.
- `canonicalize_position(board, to_move, action=None) -> tuple[str, int | None]` returns D4 canonical state and optional transformed action.
- `prepare_dataset(root: Path, output_dir: Path, seed: int, validation_min: int, test_min: int, annotations_path: Path | None = None) -> dict` returns counts, conflicts, split IDs and hashes; the first call freezes historical splits, the final call merges generated training-only annotations without changing those splits.

- [x] Write failing tests for legal replay, duplicate positions, transformed-action correctness, normalized conflicting targets, whole-group split and no D4 overlap.
- [x] Run `PYTHONPATH=. pytest -q tests/test_f_v1_dataset.py` and confirm feature-related failures.
- [x] Implement source enumeration for 24 registered formal 9×9 JSONL files; exclude pilot/archive/15×15, replay all 2,809 expected games, and deduplicate exact game sequences.
- [x] Build opening components by shared D4-canonical state, split entire components with fixed seed, and emit only C/E5 policy decisions plus verified final-outcome values; preserve the historical split manifest for Task 2.
- [x] Run the focused tests and verify history-only counts against the preflight audit.

### Task 2: Mixed game generation and teacher annotation

**Files:**
- Create: `src/research/imitation_f/generation.py`
- Create: `scripts/generate_f_v1_data.py`
- Create: `tests/test_f_v1_generation.py`
- Create: `research/f_v1/results/generated_games.jsonl`, `generated_annotations.jsonl`, `generation_progress.json`

**Interfaces:**
- `generate_game(spec, seed, budget_seconds) -> dict` uses existing A/B/C/E5 engines and records every legal move.
- `annotate_position(board, to_move, teacher, budget_seconds, seed) -> dict` records teacher action and search metrics.
- `run_generation(config, output_dir, frozen_split_manifest) -> dict` atomically resumes by unique game/position ID, rejects historical/held-out opening collisions, and emits annotations for the train split only.

- [x] Test color swap pairing, fixed seeds, legal replay, generated ID dedupe, annotation ID dedupe and interruption-safe progress.
- [x] Confirm tests fail for missing generation/resume behavior.
- [x] Generate two pilot mixed games and annotate their decision positions with C/E5 at 50ms; measure games/hour, labels/second and unique canonical positions/game.
- [x] Implement the frozen 6-pairing schedule (A/B, A/C, A/E5, B/C, B/E5, C/E5), three openings per pair with color swaps (36 games total); stop after the registered schedule reaches the data threshold.
- [x] Run focused tests and replay pilot games before continuing.

### Task 3: CNN policy/value network and G-compatible API

**Files:**
- Create: `src/research/imitation_f/model.py`
- Create: `tests/test_f_v1_model.py`
- Create: `requirements-f.txt`

**Interfaces:**
- `PolicyValueNet(board_size: int = 9, channels: int = 32, blocks: int = 2)` returns `(policy_logits: Tensor[B,81], value: Tensor[B])`.
- `encode_board(board, to_move) -> Tensor[3,9,9]` encodes current-player/opponent/legal planes.
- `predict_policy_value(model, board, to_move, device) -> tuple[Tensor[81], float]` masks occupied actions before softmax.
- `load_f_checkpoint(path, device="auto") -> PolicyValueNet` validates metadata and loads weights.

- [x] Test shape/range, occupancy masking, full-board rejection, deterministic initialization and checkpoint round-trip.
- [x] Run tests to observe missing API failures.
- [x] Add compact residual CNN, policy/value heads and CPU/GPU device selection; install pinned CPU PyTorch in isolated environment.
- [x] Run tests and benchmark forward passes on representative empty/midgame states.

### Task 4: Resumable trainer and preflight estimate

**Files:**
- Create: `src/research/imitation_f/training.py`
- Create: `scripts/preflight_f_v1.py`, `scripts/train_f_v1.py`
- Create: `tests/test_f_v1_training.py`
- Create: `research/f_v1/config/training.json`, local checkpoint state and `training_history.jsonl`

**Interfaces:**
- `save_training_state(path, state) -> str` atomically writes model/optimizer/scheduler/RNG/sampler/evaluation state and returns SHA-256.
- `load_training_state(path, model, optimizer, scheduler) -> dict` restores the exact next update.
- `train(config, dataset, checkpoint_dir, resume=True) -> dict` records per-epoch losses/metrics and returns best validation checkpoint.

- [x] Test checkpoint resume against an uninterrupted deterministic two-update run and verify sampling cursor/RNG equality.
- [x] Run the new tests and confirm they fail before implementation.
- [x] Implement policy soft-target cross-entropy, legal masking, terminal value MSE, deterministic group-aware batches, mixed-precision only when a supported GPU exists, and frequent atomic checkpoints.
- [x] Benchmark 100 training steps and full data loading, then print optimistic/central/conservative wall-time, CPU, RAM and storage estimates.
- [x] Freeze 10–20 epoch range and early stopping based on validation policy loss; write all seeds and architecture settings.
- [x] Run tests and begin the full training only after publishing the preflight estimate in commentary.

### Task 5: Tactical and match evaluation

**Files:**
- Create: `src/research/imitation_f/evaluation.py`
- Create: `scripts/evaluate_f_v1.py`, `scripts/play_f_v1_matches.py`
- Create: `tests/test_f_v1_evaluation.py`
- Create: `research/f_v1/results/metrics.json`, blind predictions, tactical results and match JSONL

**Interfaces:**
- `evaluate_policy(model, dataset) -> dict` returns top-1 accuracy, target mass, value metrics, legal rate and latency.
- `extract_direct_tactics(records) -> list[dict]` uses exact immediate-five rules for win/block fixtures.
- `play_match(model_f, opponent, opening, color, seed) -> dict` records and replays a full game.

- [x] Test legal masking and direct win/block labels against `GomokuBoard`, plus complete color-paired match replay.
- [x] Verify tests fail before the evaluator exists.
- [x] Compare fixed-seed untrained and best trained models on the same blind set and tactical subset.
- [x] Run six exploratory games on three blind, train-unseen openings: F versus A/B/C, each pairing with both colors; B/C use the same 50ms search cap, A natural time, F one neural inference.
- [x] Validate every game and record mean/median/p95 inference time and actual model decision timing.

### Task 6: Drive recovery, report, integrity checks and delivery

**Files:**
- Create: `src/research/imitation_f/snapshot.py`
- Create: `scripts/create_f_v1_snapshot.py`, `restore_f_v1_snapshot.py`, `build_f_v1_report.py`
- Create: `tests/test_f_v1_snapshot.py`
- Create: `research/f_v1/REPORT.md`, figures, source/data manifests, final checkpoint and recovery metadata

**Interfaces:**
- `create_snapshot(state_root, output_path) -> dict` packages only resume-required files and checksums.
- `restore_snapshot(snapshot, empty_dir) -> dict` validates hashes/code/config and restores into an empty directory.
- `build_report(results_dir) -> Path` emits Chinese report, training curves and data/match tables.

- [x] Test tamper rejection, exact restore hashes and empty-target enforcement.
- [x] Create the F V1 Drive folder and verify upload, download, SHA-256, restore and delete permissions using the official initial snapshot; keep only one official snapshot at a time.
- [x] Schedule Drive snapshots at cumulative effective hours 1, 3, 7, 13, 23, 39, 65, 91…; verify a new snapshot before deleting its predecessor.
- [x] Build report from machine-readable results, verify every displayed count/metric against raw files, and run all applicable project tests (excluding legacy branch-guard tests that necessarily bind V6.2/V6.3 branch names).
- [x] Commit and push only `research/imitation-learning-f-v1`, then verify remote SHA, report, and model artifact URLs.
