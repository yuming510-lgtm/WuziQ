# WuziQ V6.2 run progress

- Baseline: `0c7e0e5881b8e0bcd2bdd8c07c247912ad86783d`; independent branch `research/champion-200s-v6-2`.
- Frozen schedule: C–D, C–E5, D–E5; two games each; common new opening 731201; six unique IDs; each configured with 200 seconds per model decision.
- Preflight completed before formal play in 407.05 seconds. Estimates were 3.42 h optimistic, 9.59 h central, 25.67 h conservative. See `preflight/preflight.json`.
- Formal results: 6/6 games, 3/3 complete paired matchups, 211 plies replayed, 6 unique IDs. C scored 2.5, D 0.5, E5 3.0; E5 is the exploratory champion under the preregistered score rule.
- Formal wall-clock: 9 h 2 min 21 sec (2026-10-28 05:10:10Z to 14:12:31Z). Total model thinking time: 8 h 57 min 26 sec.
- Timing audit: 13 of 187 model decisions measured above 200 seconds, all flagged timed out. The largest measured overrun was 71.756 ms. Full per-model counts and amounts are in `REPORT.md` and `results/summary.json`.
- A pair-draw accounting error was caught after the second game; it was corrected in commit `dd1f32c`. The two completed C–D records were preserved, validated and resumed without replay. No decision-engine or frozen configuration files changed; see `RECOVERY_LOG.md`.
- Google Drive read/write/delete and restore were tested before play. A five-game snapshot was the previous latest valid snapshot; final six-game snapshot is uploaded, SHA-256 verified and successfully restored from an empty directory; only this V6.2 snapshot remains in its folder (`drive_recovery.json`). V6's original snapshot remains preserved.
- All six completed game records are in `results/9x9_games.jsonl`; no game is reused. `results/validation.json` contains the full replay and score validation.

Resume command: `python scripts/run_v6_2_championship.py`
Validate command: `python scripts/validate_v6_2_results.py`
Restore command: `python scripts/restore_v6_2_snapshot.py --snapshot <downloaded-zip> --destination research/v6_2/recovered-results`
