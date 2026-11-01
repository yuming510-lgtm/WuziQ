# WuziQ G V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and empirically evaluate two 9×9 AlphaZero-Lite self-play learners, random-start G0 and F2-start G-F2, with frozen historical opponents.

**Architecture:** Keep all new behavior under `src/research/alphazero_g/` and `research/g_v1/`. Reuse the frozen F2 network class, board rules and old opponents; implement a true neural-prior/value PUCT, self-play-only replay, exact-budget trainer, paired evaluator, and resumable snapshots.

**Tech Stack:** Python 3.12, PyTorch CPU (auto-detect GPU), NumPy, existing `GomokuBoard`, `PolicyValueNet`, `SearchConfig`, and `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-01-wuziq-alphazero-g-v1-design.md`

## Global Constraints

- Branch starts at F V2 commit `89fecf2a12c58cc27c985f588d5023ad9d198c80`; preserve all A–F source, checkpoints and results.
- Board is 9×9 freestyle five-in-a-row; both arms use the exact same policy/value architecture.
- G0 starts from registered random weights; G-F2 starts from frozen F2 SHA `50164ad9b139e0c47c1b1af7497f670c26e9864a81c10c02de0ff4426b2d0257`.
- Training rows must come only from that arm's own completed self-play games; no F supervised rows, C/E5 labels, rule bonuses, tactical overrides, or teacher imports.
- PUCT uses legal empty cells, current-player node values and sign-flipping backup; Dirichlet noise is training-root-only; temperature/visit targets are recorded.
- Simulation count, budgeted environment transitions, batch size, optimizer update count and evaluation schedule match between arms; all actual extra terminal-completion plies, simulations and CPU time are reported.
- Development schedules may select a checkpoint; blind schedules are frozen before training and run once after selection.
- Preflight measures 16/32/64 simulation costs, complete self-play throughput and real training throughput before formal budget registration.
- Snapshot intervals are cumulative effective compute hours 1, 3, 7, 13, 23, 39, 65, 91, …; Drive retains one verified current G V1 archive only.

## Review Focus

1. **Value perspective at terminal leaves:** assert a win is +1 for the winner and -1 for the side to move, draws are 0, and each parent backup flips sign.
2. **PUCT action legality and terminal handling:** occupied squares never receive prior/visit mass; terminal leaves never get neural evaluation; search restores the input board.
3. **No teacher/data leakage:** replay rejects non-self-play provenance and experiment manifest proves zero F1/F2 supervised row ingestion.
4. **Paired compute accounting:** exact equal budgeted transitions and optimizer updates; terminal-tail extra plies/search cost are separate fields.
5. **Crash/resume correctness:** game IDs dedupe; optimizer/scheduler/RNG/replay/progress restore from a hash-checked snapshot into an empty directory.

---

### Task 1: Neural PUCT search core

**Files:**
- Create: `src/research/alphazero_g/puct.py`
- Create: `tests/test_alphazero_g_puct.py`

**Interfaces:**
- `PUCTConfig(c_puct=1.5, dirichlet_alpha=0.3, root_noise_fraction=0.25)` validates finite nonnegative settings.
- `PUCTNode(prior, to_move)` tracks visits, value sum, expansion and child nodes.
- `PUCTSearch.search(board, to_move, simulations, *, add_root_noise=False, temperature=0.0, seed=0) -> PUCTResult` returns move, normalized 81-cell visit policy, root value, simulations, expanded nodes, neural evaluations, max tree depth, visit entropy and elapsed time.
- Node Q is stored from that node's side-to-move view; parent PUCT score is `-Q(child) + c_puct*P*sqrt(N_parent)/(1+N_child)`.

- [ ] **Step 1: Write failing tests** for sign-flipping backup, terminal winner/draw values, legal visit mass, root-noise isolation, temperature normalization, terminal no-network behavior, deterministic seed and board restoration.
- [ ] **Step 2: Run RED**: `/workspace/.venv-wuziq-f-v1/bin/python -m pytest -q tests/test_alphazero_g_puct.py`; expected failure because the G PUCT API does not exist.
- [ ] **Step 3: Implement** terminal inspection, legal expansion, neural prior/value leaf evaluation, PUCT selection, root Dirichlet mix, visit-temperature policy, action sampling and alternating-value backup.
- [ ] **Step 4: Run GREEN** on the PUCT file; all listed perspective, terminal and legal-mass tests pass.
- [ ] **Step 5: Commit** `feat(g-v1): add neural PUCT search`.

### Task 2: Complete self-play and isolated replay buffer

**Files:**
- Create: `src/research/alphazero_g/selfplay.py`
- Create: `src/research/alphazero_g/replay.py`
- Create: `tests/test_alphazero_g_selfplay.py`

**Interfaces:**
- `play_selfplay_game(model, opening, *, arm_id, game_id, simulations, seed, ...) -> (game_record, experience_rows)` finishes a legal game and returns one sample per visited state with the MCTS visit policy and final side-to-move outcome.
- Experience rows include `source="self_play"`, `arm_id`, `game_id`, canonical board, visit target, player, and final value; IDs are stable and unique.
- `ReplayBuffer(capacity, seed).add_complete_game(game_id, rows)` atomically deduplicates complete games; `sample(batch_size)` samples games uniformly and positions within game deterministically.
- Exact transition accounting stores exactly the configured prefix positions; a final over-quota episode is completed for a real outcome, with extra terminal-completion plies excluded from budgeted training steps and recorded as overhead.

- [x] **Step 1: Write failing tests** for legal completed games, alternating `+1/-1/0` values, visit-probability labels, game-ID idempotence, provenance rejection, D4-synchronous policy transforms and exact step-prefix/terminal-overrun accounting.
- [x] **Step 2: Run RED** on `tests/test_alphazero_g_selfplay.py`; expected failure because the self-play/replay API is absent.
- [x] **Step 3: Implement** complete game generation, fsynced JSONL persistence, game-level dedupe, bounded replay and deterministic batch selection.
- [x] **Step 4: Run GREEN** on the self-play tests; all labels must arise from the recorded network-guided search and terminal result.
- [x] **Step 5: Commit** `feat(g-v1): record self-play targets and replay`.

### Task 3: Dual-arm optimizer and exact resume checkpoints

**Files:**
- Create: `src/research/alphazero_g/training.py`
- Create: `tests/test_alphazero_g_training.py`
- Create: `research/g_v1/checkpoints/` and training configs (after budget freeze).

**Interfaces:**
- `initialize_arm(arm_id, seed, f2_checkpoint=None) -> PolicyValueNet`; G0 is reproducible random init; G-F2 validates and loads the exact F2 state dict.
- `train_updates(model, optimizer, replay, *, updates, batch_size, seed) -> metrics` uses visit-distribution cross-entropy plus final-outcome value MSE and D4 input/target augmentation.
- `save_g_checkpoint(path, model, metadata)` stores F-compatible policy/value weights; `save_g_training_state` atomically includes optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, replay reference/index and progress for exact resume.
- Both arms take the same `updates`, `batch_size`, replay capacity and self-play data quotas.

- [ ] **Step 1: Write failing tests** for G0 deterministic initialization, G-F2 exact hash/equal weights, distinct initial arms, loss response to visit/value targets, teacher-data rejection, atomic checkpoint load and RNG/optimizer resume equivalence.
- [ ] **Step 2: Run RED** on `tests/test_alphazero_g_training.py`; expected failure because G trainer/checkpoint helpers do not exist.
- [ ] **Step 3: Implement** policy/value optimization and full hash-bound resumable state without editing F1/F2 training files.
- [ ] **Step 4: Run GREEN** on the trainer file; a resumed next update must match an uninterrupted seeded next update.
- [ ] **Step 5: Commit** `feat(g-v1): train independent self-play arms`.

### Task 4: Disjoint schedules, paired frozen-opponent evaluation

**Files:**
- Create: `src/research/alphazero_g/experiment.py`
- Create: `src/research/alphazero_g/evaluation.py`
- Create: `tests/test_alphazero_g_experiment.py`

**Interfaces:**
- `build_g_schedule(seed, train_groups, development_groups, blind_groups, excluded_group_ids) -> dict` creates legal, deterministic, disjoint, 4-ply opening groups and hashes the complete schedule.
- `play_evaluation_match(checkpoint, arm, mode, opponent, opening, color, ...)` supports `mode="network"` and `mode="puct"`; PUCT is deterministic, noise-free, and uses the registered fixed search budget.
- Frozen opponents are A natural rule AI, B pure Minimax with 50ms, and the exact frozen F1/F2 checkpoints. No C in V1 unless the registered resource estimate has ample headroom; default V1 omits C.
- Match records include complete plies, per-move elapsed time, model mode, PUCT visits/sims/nodes/depth/entropy, opponent nodes/depth, outcome, seeds and budgeted/actual compute fields.

- [ ] **Step 1: Write failing tests** for opening legality/disjointness/historical exclusions, deterministic hash, paired color swap, match-ID uniqueness, score accounting, pure-network masked argmax, and same-checkpoint network-vs-PUCT mode separation.
- [ ] **Step 2: Run RED** on `tests/test_alphazero_g_experiment.py`; expected failure because G schedule/evaluator are absent.
- [ ] **Step 3: Implement** frozen opponent adapters and paired game runner; ensure G inference contains only legal masking and policy/MCTS selection.
- [ ] **Step 4: Run GREEN** on the experiment test file; replays confirm all four frozen opponents use unchanged checkpoint/code hashes.
- [ ] **Step 5: Commit** `feat(g-v1): add paired self-play evaluation`.

### Task 5: Integrity-checked snapshots and G Drive recovery path

**Files:**
- Create: `src/research/alphazero_g/snapshot.py`
- Create: `scripts/create_g_v1_snapshot.py`, `scripts/restore_g_v1_snapshot.py`
- Create: `tests/test_alphazero_g_snapshot.py`

**Interfaces:**
- `create_snapshot(root, output, commit, active_seconds, manifest)` packages only `research/g_v1`, G code/scripts/config/tests and frozen F1/F2 checkpoints required to resume/evaluate.
- `restore_snapshot(archive, empty_directory)` rejects unsafe/duplicate paths, verifies size and SHA-256 for every file, and checks branch/base/config identity.
- `next_snapshot_threshold(active_seconds)` returns cumulative thresholds 3600, 10800, 25200, 46800, 82800, 140400, then +26 hours.

- [ ] **Step 1: Write failing tests** for round-trip restoration to an empty directory, tamper/hash failure, duplicate IDs/path rejection, nonempty destination rejection and every interval boundary.
- [ ] **Step 2: Run RED** on `tests/test_alphazero_g_snapshot.py`; expected failure because G snapshots are absent.
- [ ] **Step 3: Implement** safe manifest-based atomic archive/restore and active-compute threshold tracking.
- [ ] **Step 4: Run GREEN**; all snapshot tests pass and a restored G-F2 checkpoint loads.
- [ ] **Step 5: Commit** `feat(g-v1): add verified recovery snapshots`.

### Task 6: Full preflight and frozen, affordable budget registration

**Files:**
- Create: `scripts/preflight_g_v1.py`, `scripts/register_g_v1_budget.py`
- Create: `research/g_v1/results/preflight.json`, `research/g_v1/config/experiment.json`
- Test: `tests/test_alphazero_g_experiment.py`

- [ ] **Step 1: Write failing tests** for budget builder requiring measured simulation/self-play/update rates, matching both arms, unique seeds, finite hard cost cap and immutable blind groups.
- [ ] **Step 2: Run RED** on the new budget-builder tests; expected failure until measured registration exists.
- [ ] **Step 3: Measure preflight** on the real 32×2 CNN: initial/middle/tactical board states at 16/32/64 simulations; two complete self-play games per arm; real optimizer update throughput on those replay states.
- [ ] **Step 4: Record estimates** (optimistic/central/conservative wall-clock, CPU, memory, disk) and register exact matched simulations, environment-step quota, updates, batch size, stage checkpoints, development groups, final blind groups, seeds, and evaluation games before formal training.
- [ ] **Step 5: Verify Drive** by creating/listing a dedicated G V1 folder, uploading a small probe snapshot, downloading it, checking SHA-256, restoring to an empty directory, and deleting only the probe after success.
- [ ] **Step 6: Run GREEN** on budget tests; verify G0 and G-F2 budgets/config are byte-identical except initial checkpoint hash/seed.
- [ ] **Step 7: Commit and push registration** before the formal blind evaluation; no budget or blind schedule changes afterward.

### Task 7: Blocking, resumable matched self-play training

**Files:**
- Create: `src/research/alphazero_g/runner.py`
- Create: `scripts/run_g_v1.py`
- Create: `research/g_v1/results/selfplay/`, `research/g_v1/results/training/`, staged checkpoints.
- Test: `tests/test_alphazero_g_training.py`

- [ ] **Step 1: Write failing tests** for per-game atomic persistence, kill/resume with unique-ID dedupe, exact arm environment-step/update counts, replay-only provenance and stage-complete resume.
- [ ] **Step 2: Run RED** on runner tests; expected failure because no resumable experiment runner exists.
- [ ] **Step 3: Implement** matched stage runner that self-plays both arms with the same registered sims, gathers exactly equal budgeted state counts, performs exact equal updates and freezes every registered checkpoint.
- [ ] **Step 4: Run GREEN** on tiny configurations; interrupted/resumed and uninterrupted tiny runs must end with the same progress/target accounting and match results.
- [ ] **Step 5: Commit** `feat(g-v1): run matched self-play training`.
- [ ] **Step 6: Execute** the pre-registered formal budget to completion in blocking mode, persisting every complete game/batch; at every crossed Drive threshold upload, download/hash-check, restore-test the new snapshot, then delete the preceding G V1 snapshot only.

### Task 8: Frozen evaluation, report, verification and branch delivery

**Files:**
- Create: `scripts/evaluate_g_v1.py`, `scripts/report_g_v1.py`
- Create: `research/g_v1/REPORT.md`, `research/g_v1/figures/`, final result summaries and integrity manifest.
- Tests: all `tests/test_alphazero_g_*.py` plus applicable repository tests.

- [ ] **Step 1: Write failing tests** for W/D/L and half-point scores, color splits, paired-opening bootstrap intervals, fresh blind schedule use, report/raw-data consistency and “search gain vs learning gain” decomposition.
- [ ] **Step 2: Run RED** on report/evaluation tests; expected failure because registered evaluation/report outputs are absent.
- [ ] **Step 3: Run development evaluations** at registered checkpoints against A, B, F1, F2 for both pure policy and identical-budget PUCT; choose best checkpoint per arm only with the frozen development score rule.
- [ ] **Step 4: Run the final paired blind schedule once** for selected G0/G-F2 checkpoints against A, B, F1, F2 in both inference modes; do not tune after it.
- [ ] **Step 5: Generate report and plots** for learning curves, entropy, outcomes, color, CPU cost, simulations, search-vs-network contributions and limitations; save full match/replay/config data and checkpoint hashes.
- [ ] **Step 6: Validate** every self-play/evaluation game by legal replay; verify no training/validation/blind leakage, exact arm budgets, all result IDs unique, snapshot restore, F2/G checkpoint loading and report values against raw records.
- [ ] **Step 7: Run full applicable tests**, commit all G V1 code/data/models/report, push `research/alphazero-g-v1`, and verify remote SHA and artifact paths.

## Execution Ruling

- `Ruling`: keep full terminal labels while matching each arm's *budgeted training environment steps* exactly. If the last full game crosses the quota, finish it but exclude its over-quota tail states from the equal training sample budget; report actual tail plies/simulations/CPU separately. This follows the user's equal-step requirement without inventing terminal outcomes; cost if wrong is that a reader may prefer raw executed plies, which will be shown alongside the exact training quota.
