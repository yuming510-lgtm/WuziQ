# V5.1 ABCD independent-version ladder design

## Purpose

V5 treated a family as one challenge route. A loss by a low-budget version could therefore prevent an untested high-budget version from playing. V5.1 separates the fixed-budget challenger version from the opponent-budget rung and computes advancement per route. It retains the V1–V4C decision engines and V5 paired-game conventions.

## Frozen competition model

- A is one natural-time rules model. B, C, and D each have seven fixed-budget versions: 50ms, 200ms, 1s, 5s, 10s, 20s, and 50s.
- A challenges each other family through its seven budget versions, in increasing order. Each B/C/D version also challenges A once; those exact games are shared with A's route.
- A fixed B/C/D version challenges each of the other two families through all seven opponent versions. The opposite fixed-version route reads the same two games from its own perspective.
- There are 66 independent routes: three A-to-family ladders, 21 fixed-version-to-A routes, and 42 fixed-version-to-family ladders. The maximum unique schedule is 168 paired groups / 336 games, with no same-family pairings.
- A paired group is one exact matchup between model versions and one opening. It has two games with colors swapped. It may advance two routes, but each route earns its own score and stops independently. A later shared result remains visible as a real game but is marked not counted for any route that already failed.
- A route passes a level at 1.0 or more points from the two games. Route score is `wins + draws / 2`; route score rate divides by two games. Missing levels are never inferred as wins or losses.

## Schedule and seed design

The config freezes the V5 opening catalog, base seed, algorithm parameters, engine hashes, and family ordering. The schedule is materialized and hash-pinned before official execution. A-family opponent ladders and B/C/D pairwise budget grids use deterministic order. In a pairwise grid, the first model's budget index is the outer loop and the second model's index is the inner loop; this is a topological order for both directions' ascending opponent ladders. Every schedule group has two route references, each with its own rung index. Opening allocation is deterministic, spans the full 12-position legal nonterminal pool, and aligns eligible V5 axis groups to their original openings for exact reuse.

## V5 reuse boundary

Only a V5 group whose pair, per-model budget, opening coordinates/seed, color assignment, per-game random seed, algorithm configuration, and pinned engine hashes match the V5.1 schedule can be reused. Source IDs and source commit are retained. The official V5 records are replay-validated first. Reuse is prepared and stored before new results are run; the importer must never select games by winner or route outcome. Nonmatching groups are new games.

## Execution and recovery

Each game is appended once under a V5.1 unique ID after completion and replay validation. After every paired group, progress and independent route states are atomically rewritten. A process restart replays durable records, verifies the frozen config/schedule hashes, finishes a partial two-game group, then continues the next group with at least one active route. A group with no active route reference is skipped without counting its result. The V5.1 Drive snapshot contains only config, frozen schedule/reuse manifest, code identity, durable game records, and progress. A new snapshot replaces the previous V5.1 snapshot only after upload, download, SHA-256 validation, and empty-directory restoration all succeed.

## Statistical outputs and V5 score correction

All reported cumulative points are derived from counts as `wins + draws / 2`; all rates are points divided by actual games. Group, pairwise, version, route, color, and family summaries must reconcile to the raw games. Descriptive 95% intervals use paired group results and must be labeled exploratory. The V5 report's matrix used a pairwise field that halved the first-listed model's points while leaving its rate unchanged; V5.1 recomputes both sides from win/draw counts and includes an explicit audited correction note. The V5 branch remains untouched.

## Verification contract

Before formal execution: validate the complete schedule, route graph, unique IDs, lack of same-family games and cycles; verify all eligible V5 reuse entries; measure CPU, memory, disk, and per-rung decisions; estimate full schedule and strict budget ceiling; verify Drive round-trip recovery. Before delivery: replay every official V5.1 game, prove exact-once IDs and complete groups, recompute all points and rates directly from game outcomes, validate report/CSV/SVG against the analysis, run the full test suite, and push only `research/abcd-ladder-v5-1`.
