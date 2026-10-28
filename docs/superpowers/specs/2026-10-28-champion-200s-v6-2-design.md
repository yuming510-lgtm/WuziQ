# V6.2 200-second champion tournament design

## Intent

Run a new, fully reproducible exploratory round robin among the frozen V6 C, D, and E5 players. Use a 9×9 board, a shared legal opening not used in V6 results, a 200-second maximum for every searched move, and two color-swapped games for each of C–D, C–E5, and D–E5. Preserve the original V6 branch, code, and results.

## Goals and boundaries

- Record and replay-check all six newly played games, including move times, completed search depth, nodes, cutoffs, MCTS simulations, and timeouts.
- Persist each completed game atomically and resume by unique game ID. Restart an interrupted game from its opening because internal search state is not persisted.
- Pre-register the one-point-per-win, half-point-per-draw scoring rule. Highest total points is champion; an exact tie produces joint champions, with no secondary tie-break.
- Publish raw data, frozen configuration and schedule, runtime/resource estimate, tests, charts, Chinese report, a restore tool, and one verified V6.2 Drive snapshot.
- Keep `src/ai.py`, `src/gomoku.py`, `src/research/mcts_d2.py`, the V6 E5 implementation, and all V6 configurations unchanged.
- No training, decision tuning, algorithm changes, historical-game substitution, or score inference.

## Approaches considered

1. **Reuse the V6 per-game engine and validator, with a separate V6.2 schedule and runner (selected).** This preserves the tested C/D/E5 decision path and replay validation while confining all new records and control logic to `src/research/v6_2`.
2. **Copy the V6 move loop into a V6.2 implementation.** This would duplicate algorithm dispatch and risk different timing or metric semantics, so it is not selected.
3. **Modify the V6 tournament configuration and runner.** This would make the six-game challenge harder to distinguish from V6 and risks changing historical experiment artifacts, so it is not selected.

## Frozen opening and schedule

The registered four-ply opening is `731201`: B `(4,4)`, W `(3,5)`, B `(5,3)`, W `(2,6)`. Validate it against the existing 9×9 rules and assert that its exact four-ply sequence is absent from V6's complete opening catalog and actual game records. All three matchups use this same sequence. For each matchup, the second game swaps the model-to-color assignment. Derive each game's seed deterministically from the V6.2 base seed and unique game ID.

## Architecture

`src/research/v6_2/config.py` pins the V6.2 settings and hashes of the decision-engine files. `schedule.py` produces exactly three pairs and six unique game specifications. `runner.py` invokes V6 `play_game` with the V6 engine settings, appends each validated complete record through the existing fsync checkpoint helper, and atomically updates progress. `validation.py` checks schedule coverage, board replay, terminal correctness, color swaps, move budgets, exact-once IDs, and recomputed totals. `snapshot.py` writes and restores a minimal checksummed archive containing the frozen config, schedule, progress, results, and V6.2 recovery code identity. Scripts provide preflight, execution/resume, validation, report generation, and snapshot restore.

## Persistence and recovery

Create a new Drive folder named `WuziQ V6.2 Champion 200s`, leaving the V6 folder untouched. Before official play, upload a small probe, fetch and download it, verify SHA-256, restore it into an empty directory, and delete only that probe. Upload an initial V6.2 snapshot before the games. Every fully completed game is appended and fsynced before the next starts. Incomplete games are never counted and restart from the beginning. If the work takes 24 hours or more, replace the Drive snapshot only after its replacement has been downloaded and hash-checked and restored successfully in an empty directory.

## Report and analysis

Report total points, pairwise W–D–L, color split, per-model elapsed time and compute totals, C/E5 search depth and nodes, D depth and simulations, timeout counts, replay validation, actual runtime, and the six-game sample limitation. Use a compact SVG visualization of pairwise points and compute cost. Label every conclusion exploratory; do not claim statistical significance.

## Anti-goals

- Do not count any V6/V5.1 historical game as a V6.2 result.
- Do not modify or retune C, D, or E5.
- Do not stop early due to an apparent winner.
- Do not describe a near-zero-time or interrupted move as if it consumed 200 seconds.
- Do not claim Drive upload, restore, GitHub push, or model identity unless independently verified.
