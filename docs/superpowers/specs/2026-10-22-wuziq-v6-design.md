# WuziQ V6 advanced search design

## Goal

Add five non-learning search players E1–E5 on top of the frozen V5.1 rule and A/B/C/D engines. Validate tactical proof behavior before running paired budget ladders, preserve exact V5.1 games when configuration and replay agree, and publish all V6 artifacts on `research/advanced-search-v6`.

## Frozen boundaries

- Board size is 9×9, five-in-a-row rules and the existing `GomokuBoard` API remain authoritative.
- `src/ai.py`, `src/gomoku.py`, and `src/research/mcts_d2.py` stay byte-for-byte unchanged. A/B/C/D therefore retain their tested decisions and timing behavior.
- Every E move receives one wall-clock budget covering tactical search, fallback selection, and result assembly. The first legal fallback is retained before search; a timeout returns it or the last fully proven tactical move.
- No learned weights, neural networks, reinforcement learning, or opponent-specific logic.
- Proof claims are asymmetric: a returned proof must be verifiable from a complete OR/AND certificate. Search exhaustion, timeout, omitted attack candidates, or an incomplete defense list means `unknown`, never `no forced win` or `proven win`.

## Algorithm architecture

- E1 VCF searches attacker moves that create immediate winning continuations. A one-square threat branches on its blocking move; a multi-square threat is forced only if no immediate counter-win exists. A certificate records every defense that can remove the current win set.
- E2 VCT extends the threat vocabulary to live-three and multi-threat moves. It searches OR attack nodes and AND defense nodes. All legal opponent moves are covered at AND nodes unless a separately checked local theorem proves a smaller complete critical-defense set. The default certificate verifier requires complete coverage.
- E3 builds threat records (`gain`, `cost`, `rest`) and a dependency/conflict graph. The graph orders and narrows attack candidates only; it cannot remove legal opponent defenses from a proof certificate. Final proof validation checks every defense branch.
- E4 is proof-number search over an attacker-OR / defender-AND tree. Attacker expansions use legal threat candidates, defender expansions use all legal moves, and terminal proof/disproof numbers are assigned only from the existing win checker or a fully exhausted node.
- E1–E4 share one fallback: the existing deterministic-seed rule selector A. Fallback time and tactical-search time share the same deadline.
- E5 is a separate iterative-deepening negamax alpha-beta implementation with PVS, transposition table, deterministic Zobrist keys, tactical move ordering, and bounded threat extensions. Flags support one-component-at-a-time ablations. E5 never calls E1–E4 as a replacement search; its tactical features are move ordering and bounded extensions only.

## Tactical acceptance set

Freeze deterministic board fixtures for: immediate win, mandatory immediate block, open-four VCF, a live-three/fork VCT pattern, a counter-winning refutation, a false tactical claim, an edge threat, a no-proof position, timeout, deterministic repeat, and board/history restoration. Each positive fixture has an independently replayable proof certificate. Record proof-found, no-proof-within-bound, or unknown separately.

## Tournament model

- Families: A, B, C, D, E1, E2, E3, E4, E5. A is natural-time; each other family has versions at 50ms, 200ms, 1s, 5s, 10s, 20s, and 50s.
- No same-family matches. Every paired group has one frozen legal opening and exactly two games with colors swapped.
- A route is one fixed challenger family/version against another family's opponent budgets in ascending order. A failed level stops only that route. Shared exact games are evaluated independently by both routes. Under the user's monotonic-budget rule, skipped higher opponent rungs may be marked inferred, but they receive no game, score, or empirical CI.
- Same model-version pair and opening is scheduled once. Only full, replay-valid V5.1 paired games with unchanged A/B/C/D engine hashes, budgets, opening, colors, and seed are eligible for reuse. V5.1 inferred groups and the unpaired game are not standings results.
- Full theoretical maximum is 1,428 paired groups / 2,856 games / 456 routes. Preflight and early-stop rules determine the actual run; no result is fabricated for unrun combinations.
- Each completed game is append-only JSONL, fsynced, and unique by game ID. Progress is atomically written after each group. A partial pair is finished on resume before advancing.

## Environment and recovery

- Record CPU count, memory, disk, software version, source commit, and benchmark timings before official games.
- A V6-only Drive folder is used. A probe must upload, download, match SHA-256, restore into an empty directory, and delete successfully before official execution. Rotating snapshots contain only frozen config/schedule, code commit, completed games, route progress, and resume identity. New snapshot verification precedes deletion of the previous V6 snapshot.
- Report actual Drive and GitHub outcomes. If external access changes, stop claiming backup/push success and record the observed limitation.

## Reporting

Publish Chinese report, tactical validation record, raw games, inferred/unrun route states, family/version score matrices, paired color tables, E5 ablation data, compute profiles, SVG charts, and a source/config hash manifest. Counts, points (`wins + draws/2`), and rates must recompute from raw paired games. Tables label empirical, reused, inferred, and unrun separately. Two games per matchup are exploratory only.

## Review focus

- No proof can arise from incomplete opponent coverage or deadline interruption.
- Every board mutation is undone even on timeout/exception.
- E1–E4 fallback and proof search share one deadline; returned move is always legal when the board is nonterminal.
- E5 TT keys include side-to-move and board size; TT bounds are used only at sufficient depth; PVS re-searches fail-high windows correctly.
- Tournament resume cannot double-count reused or newly completed games; partial source data is excluded.
- Inference and empirical outcome rows are never merged in score denominators.
