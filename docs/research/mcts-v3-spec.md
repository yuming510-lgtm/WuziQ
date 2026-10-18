# V3 MCTS High-Compute Challenge

## Research question

Measure whether the V2 optimized pure UCT implementation can challenge the existing A (rule AI), B (Minimax), and C (Minimax with Alpha-Beta) when D3 receives 10, 20, or 50 seconds per move, while A/B/C remain at 50 ms. This is an intentionally unequal-compute, exploratory challenge.

## Algorithm boundary

D3 is the same `MCTSAgent` implementation and search settings as V2 D2. The only planned experimental change is D3's time budget. It uses UCT selection, MCTS expansion, terminal tactical rollouts, and backpropagation; it must not call Minimax/Alpha-Beta or any trained model. A/B/C implementations and their 50 ms settings stay unchanged. If measurements identify a necessary D3-only implementation fix, it must be separately tested, documented, and distinguished from the compute increase; otherwise there is no algorithm-policy change.

## Registered game schedule

- Board: 9×9.
- D3 budgets: 10 s, 20 s, and 50 s per move.
- Opponent budget: 0.05 s per move for A, B, and C.
- Conditions: each D3 budget versus each of A, B, and C; 6 games per condition; 3 opening pairs with model colors exchanged, for 54 official games total.
- Three opening seeds and their generated four-ply positions are frozen before official games and reused across all nine conditions. Each game has a stable ID and a deterministic game/search seed derived from the registered base seed and ID.
- A completed game is appended and fsynced before progress advances. Resume skips completed IDs and rejects duplicates or records outside the registered schedule.
- A legal move returned after its time limit is counted as a timeout and remains on the board; an illegal move forfeits. Both are recorded.

## Measurements and interpretation

Preserve every move and game result. Per move record elapsed time, actual completed simulations, MCTS tree depth, and timeout status. Report W/D/L, pure win rate, score rate (draw=0.5), paired 95% intervals, simulation totals/rates, depth, elapsed time, and timeout/illegal counts for each condition and budget trend. Include the full 3×3 opponent/budget matrix and inspect first-player outcomes. Compare D3 against the matching V2 D2 conditions; attribute differences between D2 and D3 solely to the changed time allowance unless the implementation changed. Six games per condition are exploratory and cannot support a stable-superiority claim.

## Recovery and delivery

Before formal games, measure representative positions at all three budgets, inspect CPU/memory/storage, estimate wall time and result size, and complete an actual Drive upload/read/hash/restore drill. Recovery archives contain only the registered config, code commit, progress and complete per-game records. If the run crosses 24 hours, upload and verify a new snapshot before deleting the previous V3 snapshot. Deliver the code, tests, preflight measurements, immutable raw game JSONL, summary, trend chart, and Chinese report on `research/mcts-potential-v3`, leaving `main` and V1/V2 history intact.

## Limits

No neural network or reinforcement-learning training is involved. Results describe algorithm development and exploratory play evaluation only. The active Codex model/preset cannot be independently verified from the runtime unless the platform exposes it; the report must say so rather than infer it from the request.
