# V6.3 C–E5 Champion Tiebreak Plan

## Scope

Run a paired, 200-second-per-move C versus E5 tiebreak on the frozen V6.2 engine. Preregister two fresh four-ply openings and one conditional tiebreak opening. Play four opening/color-swap games; play the last two only when the first four points are tied. Keep V6.2 data and all existing Drive snapshots untouched.

## Files

- `configs/v6_3/championship.json`: baseline, immutable engine hashes, seeds, openings, score rule, and Drive folder ID.
- `src/research/v6_3/`: config validation, deterministic schedule, exact-once game runner, replay/score validation, snapshot/restore, and analysis.
- `scripts/run_v6_3_championship.py`, `preflight_v6_3.py`, `validate_v6_3_results.py`, `build_v6_3_report.py`, `create_v6_3_snapshot.py`, `restore_v6_3_snapshot.py`: execution and recovery entry points.
- `tests/test_v6_3_*.py`: opening freshness, paired schedule, conditional tiebreak, idempotent resume, replay validation, and empty-directory snapshot restoration.
- `research/v6_3/`: preregistration, progress, raw JSONL, derived tables/figures, Drive snapshot audit, and Chinese report.

## Steps and gates

1. Write schedule, conditional-route, restore, and resume tests; observe the expected failures.
2. Implement V6.3 using the existing V6 game engine and V6.2 checkpoint/replay primitives; do not modify C/E5 implementation files.
3. Run unit tests, validate three fixed openings against V6/V6.2 history, inspect engine/config hashes, and freeze the four-plus-two schedule.
4. Verify the dedicated Drive folder by uploading, downloading, hashing, restoring into an empty directory, and deleting only the probe file. Upload and restore the initial V6.3 recovery snapshot before the first game.
5. Run a brief C/E5 budget preflight and update optimistic/central/conservative wall-time estimates from V6.2 observations; then execute the four scheduled games and conditional two only if required.
6. Replay every completed game, reconcile scores and V6.2 direct results, produce tables, visualizations, and the Chinese report, then run the full test suite and data-integrity checks.
7. Commit and push only to `research/champion-tiebreak-v6-3`; verify remote SHA and report/raw data URLs.

## Expected outcome

The score, any champion, and the V6.2 combined head-to-head record are computed only from validated games. With four to six new games this remains an exploratory result; no statistical-significance claim is made.
