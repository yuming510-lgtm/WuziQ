# V5 ABCD Ladder Challenge Design

## Purpose and constraints

V5 measures where the four existing Gomoku algorithm families stop advancing as compute budgets increase. It runs only on 9x9, leaves A/B/C/D decision implementations unchanged, preserves all V1–V4C artifacts, and keeps all V5 records separate. A uses its natural rule-selector time. B/C/D use the frozen ladder `50ms, 200ms, 1s, 5s, 10s, 20s, 50s`.

## Frozen pairing and promotion rules

- A is the challenger in A–B, A–C and A–D. At rung `r`, A plays its natural rule against the other model at budget `r`.
- For each B/C/D pair X–Y, the `50ms vs 50ms` group is shared by both directed routes. At each later rung, X at the current budget challenges Y fixed at 50ms, while Y at the current budget separately challenges X fixed at 50ms.
- Each group has two games from one frozen opening, with model colors swapped. A challenger advances on a two-game score of at least 1.0. A failed route stays failed. For shared B/C/D baseline results, both routes score the same games from their own perspective and advance independently.
- This yields at most 21 A groups plus 39 B/C/D groups = 60 groups / 120 games. Historical games are comparison material only.
- A frozen 12-position legal, nonterminal opening pool is distributed by a deterministic pair/rung/direction formula. Game and per-move random seeds derive from a fixed base seed and stable IDs.

## Implementation design

1. A frozen JSON config pins the schedule, engine source hashes, timing policy, opening pool and seed. It contains no neural network or learned weights.
2. `src/research/abcd_ladder_v5.py` owns config validation, schedule expansion, game execution, atomic per-game persistence, resume, unique-ID checks and route-state reconstruction. It reuses the unchanged `choose_move`, `SearchConfig`, `MCTSAgent` and `GomokuBoard` implementations.
3. `src/research/abcd_ladder_v5_analysis.py` replays every game, validates the route graph, calculates group/model/side metrics and writes CSV/JSON.
4. `src/research/abcd_ladder_v5_report.py` writes the Chinese report and an SVG budget-gradient chart from validated records.
5. Snapshot scripts package only frozen config, source commit/hash manifest, progress and completed games. The Drive handoff uploads a new V5 snapshot, downloads and verifies its SHA-256, restores it to an empty directory and validates replay before deleting the previous V5 snapshot.

## Validation focus

- Every opening is legal and nonterminal; every challenge group has exactly two color-swapped games.
- The schedule has 60 unique group IDs at maximum, with only one shared baseline group per internal family pair and no duplicate model-budget pairing.
- Routes advance independently after shared baselines and stop permanently after a failure.
- A is never delayed and has no artificial timeout. B/C use identical V4 search settings except each current budget; D uses the V4C MCTS agent and tree reuse with only the budget changed.
- Completed games replay to their recorded result; a half-complete pair does not update promotion state and resumes with only its missing game.
