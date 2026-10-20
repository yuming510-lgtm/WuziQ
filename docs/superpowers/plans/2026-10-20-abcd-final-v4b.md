# V4B ABCD Final Match Execution Plan

## Workflow

1. Inspect V3 reuse provenance, V4 search limits, cloud resources, and baseline tests.
2. Freeze the V4B config and create a separate, resumable tournament runner, validator, snapshot support, and tests. Write tests before implementation changes.
3. Validate the exact six V3 D20s-vs-A records and run a Drive upload/download/hash/restore drill.
4. Measure representative A/B/C/D decisions, then persist a wall-time/resource estimate before the formal run.
5. Run the 30 uncompleted scheduled games sequentially; persist and fsync each complete game, update progress, and skip completed unique IDs on resume.
6. Replay and validate all 36 canonical games; analyze paired score rates, intervals, color effects, and compute telemetry; generate tables, plots, and the Chinese report.
7. Run the full test suite and analysis validation, request a whole-branch review, address Important/Critical issues, finalize the report, commit, push, and verify the remote branch.

## Implementation tasks

- **Configuration and schedule:** freeze the six pairings, three registered V3 openings, 200ms B/C settings, D3 engine alias and 20s budget; create deterministic IDs and seeds.
- **V3 reuse:** validate source commit and algorithm file hash, exact board/opening/matching budgets/color pairings, and legal move replay. Keep source JSONL immutable.
- **Runner and persistence:** dispatch A/B/C through V4 `choose_move`; maintain one `MCTSAgent` per D player for V3-compatible subtree reuse; enforce per-model budgets and record full metrics; append one game atomically and update resume state.
- **Validation and snapshots:** check quotas, exact opening catalog, colors, legal replay, terminal result, metrics, IDs, hashes, snapshot restore, and Drive replacement rules.
- **Analysis/report:** calculate match tables and paired-opening intervals, compute search/cost summaries and color effects, generate reproducible figures and Chinese findings.

## Self-review questions

- Is D exactly V3 D3, including persistent tree reuse and all V3 settings?
- Do B and C share the same V4 search implementation and differ only by alpha-beta?
- Are historical and newly played games labeled and compared without conflation?
- Does the runner recover after a crash without duplicate games or invalid partial records?
- Are timing, node/simulation metrics, and tiny-sample uncertainty described accurately?
