# AlphaZero G V6.2 Implementation Plan

> **For agentic workers:** Implement in the current isolated V6.2 worktree. Keep every task testable and preserve the frozen V6 algorithm.

**Goal:** Continue the complete V6 training state from 1,012 to 2,000 games, then advance to 3,000 only if the registered PUCT-400 paired gate passes.

**Architecture:** Add an isolated V6.2 orchestration namespace around the existing reference trainer and evaluator. The only trainer change is an optional exact cumulative-game stop boundary plus passive loss-component logging; model, optimizer, replay, RNG, search, objective, and hyperparameters stay frozen. Stage evaluation and checkpoints live under `research/g_v6_2/`.

**Tech Stack:** Python, PyTorch CPU, existing vendored AlphaZero_Gomoku trainer, pytest, Git, Google Drive connector.

**Spec:** User request “WuziQ｜AlphaZero G V6.2：9×9 长期自我对弈训练”; frozen parameters in `research/g_v6/config/registration.json` and `research/g_v5/config/experiment.json`.

## Global Constraints

- 9×9 free Gomoku, five in a row, 400 PUCT simulations per self-play move, c_puct 5.0.
- Resume the exact V6 full state: 1,012 games, 20,001 states, 1,011 updates, same model, Adam slots, replay, and RNG.
- No teacher data, external games, handcrafted rewards, tactical rules, or inference patches in training.
- Stop at cumulative game milestones 1,250, 1,500, 1,750, 2,000; 2,500/3,000 only after the registered gate.
- Per-run CPU cap 10 hours; total new CPU cap 30 hours.
- Persist each completed game and make recovery idempotent.

## Review Focus

- Full-state restore and model identity: verify tensors, optimizer, replay, and RNG before first move.
- Milestone boundary correctness: exact game count and no duplicate IDs after resume.
- Paired evaluation: fixed opening reused across both colors, separate PUCT-64/400 accounting.
- Blind-set isolation: no blind opening is read by model selection.
- CPU accounting: distinguish resumed V6 cumulative time from new V6.2 CPU consumption.

---

### Task 1: Freeze V6.2 protocol and recovery baseline

**Files:** `research/g_v6_2/config/registration.json`, `opening_schedule.json`, `tactics/`, `checkpoints/`, `scripts/prepare_g_v6_2.py`.

- [ ] Validate V6 baseline checkpoint SHA and exact 1,012/20,001/1,011 counters.
- [ ] Copy the full checkpoint atomically to V6.2 while retaining the V6 branch files unchanged.
- [ ] Register budgets, openings, gate, and CPU limits before training.
- [ ] Verify the copied checkpoint by exact restore.

### Task 2: Add exact game-boundary continuation

**Files:** `src/research/alphazero_g/v5_training.py`, `src/research/alphazero_g/v5_reference.py`, `src/research/alphazero_g/v6_2_training.py`, `tests/test_alphazero_g_v6_2_training.py`.

- [ ] Add and fail a test for exact cumulative game stopping.
- [ ] Add optional game stopping without changing existing behavior.
- [ ] Record policy/value loss components from already-computed tensors without changing optimization.
- [ ] Test exact stopping and complete-state restoration.

### Task 3: Evaluate milestones and enforce the 2,000-game gate

**Files:** `src/research/alphazero_g/v6_2_evaluation.py`, `scripts/run_g_v6_2.py`, `research/g_v6_2/results/`.

- [ ] Generate and hash distinct development/blind opening pools.
- [ ] Persist full paired games and summaries under a V6.2-only namespace.
- [ ] Evaluate baseline, A, B, PUCT-64, PUCT-400, and tactics at each completed checkpoint.
- [ ] Apply the registered PUCT-400/80-pair gate before 2,500/3,000.
- [ ] Freeze the final candidate from development data, then run separate blind evaluation.

### Task 4: Audit, report, backup, and publish

**Files:** `src/research/alphazero_g/v6_2_audit.py`, `scripts/audit_g_v6_2.py`, `scripts/report_g_v6_2.py`, `research/g_v6_2/REPORT.md`, `research/g_v6_2/charts/`.

- [ ] Replay every completed game and audit IDs, paired colors, checkpoint counters, and CPU ledger.
- [ ] Generate Chinese analysis and plots from checked-in raw results.
- [ ] Verify latest Drive snapshot upload, readback, SHA-256, and empty-directory restore before replacing it.
- [ ] Run tests, commit, push the V6.2 branch, and confirm remote SHA.
