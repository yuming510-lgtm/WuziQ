# V4B ABCD Final Match Design

## Goal

Run a small, paired and color-swapped 9×9 final match for models A, B, C, and D, where D is the V3 optimized pure-MCTS D3 implementation. Preserve all V1–V4 artifacts and publish V4B on `research/abcd-final-v4b`.

## Frozen conditions

- Board: 9×9; six matchups: A–B, A–C, B–C, A–D, B–D, C–D.
- Each matchup uses the registered V3 opening seeds 731091, 731092, and 731093, with both color assignments for each opening: six games per matchup.
- A uses the unchanged rule selector and returns as soon as it chooses a move. Its measured time is reported; the 50ms entry is only a ceiling, not a sleep.
- B and C each use a 200ms deadline and iterative deepening through the remaining board plies (`81 - occupied`), with identical V4 candidate generation, ordering, evaluation, and search settings. Only C enables alpha-beta.
- D uses the V3 D3 `MCTSAgent` implementation, one persistent agent for the game, a 20-second per-move deadline, and the unchanged V3 D2/D3 UCT configuration.
- All searches use only the public board. No neural network, reinforcement learning, or minimax substitution is allowed in D.
- A–D's six verified V3 games are reused; the other 30 games are newly played. If source, opening, budget, seed pairing, or replay verification fails, the six A–D games are rerun and marked new.

## Data and recovery

- New game IDs are deterministic and unique across the V4B schedule; each finished game is validated, appended, flushed, and fsynced before progress is updated.
- Results retain complete move-level timing, completed depth/tree depth, nodes, cutoffs, MCTS simulations, rollout plies, node reuse, timeouts, and terminal status.
- Reused V3 raw records are retained verbatim in a separate source JSONL and linked by source game ID and SHA-256. Normalized V4B rows identify historical reuse and map D3 to V4B model D.
- Recovery archives include only V4B config, progress, completed canonical games, and code fingerprint. Drive replacement follows upload → download → hash verification → restore verification → delete prior V4B snapshot. V1–V4 files and Drive artifacts are out of scope for deletion.

## Analysis and limits

- Report six-game per-matchup W/D/L, win rate, score rate (win=1/draw=0.5), paired-opening block 95% intervals, color split, and per-model compute/search costs.
- Overall rank is exploratory and is shown alongside the match matrix, not presented as a stable strength ordering. Three paired opening blocks do not support strong inferential claims.
- Compare B/C 200ms search telemetry, D's 20-second simulations and time use, and V1–V4 historical results with the historical conditions clearly separated.
- Report actual run time, reused/new game counts, environment limits, Drive drill result, and model/runtime provenance. The runtime may not expose the requested Codex model tier; state that limitation if so.

## Acceptance

1. Baseline tests pass; new tests cover frozen schedule/openings, V3 reuse gates, game persistence/resume/deduplication, B/C shared settings, persistent D3 routing, replay validation, and snapshot restore.
2. Exactly 36 canonical games exist: 30 new plus six validated V3 reuses (or 36 new if reuse cannot be verified), with no duplicate IDs and a legal complete replay for every game.
3. Each of six matchups has three fixed opening blocks and one game per color assignment.
4. Every requested report table, chart, raw game data, config, and reproducible run/analysis commands are committed.
5. V4B branch is pushed without changing `main` or V1–V4 branches.
