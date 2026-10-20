# V4B Run and Recovery

## Frozen run

`configs/abcd_final_v4b.json` contains the registered six-matchup matrix, fixed V3 openings, model budgets, and reuse provenance. Before formal games, run:

```bash
python scripts/preflight_abcd_final_v4b.py
```

This runs one representative move per model, reads the existing V3 D20 calibration games for B/C, captures CPU/memory/disk limits, and saves the runtime estimate at `research/v4b/results/preflight.json`. The preflight move is not an official game.

Start or resume the official schedule with:

```bash
python scripts/run_abcd_final_v4b.py --config configs/abcd_final_v4b.json --output-dir research/v4b/results
```

`9x9_games.jsonl` is the durable canonical ledger. Every completed game is flushed and fsynced before progress is updated. The runner checks game IDs and replays each game before appending; reruns skip completed IDs. V3 reuse failure is recorded and causes A–D's six games to be played as new V4B games.

Validate all records and write tables, figures, and the Chinese report with:

```bash
python scripts/summarize_abcd_final_v4b.py --results research/v4b/results
```

## Recovery archive contents

Create a local minimal snapshot with:

```bash
python scripts/create_abcd_final_v4b_snapshot.py --results research/v4b/results --output /tmp/wuziq-v4b-recovery.zip
```

It includes only config, progress, canonical games, reuse audit/source copy, code version, and SHA-256 manifest. Verify/restore it into an empty directory with:

```bash
python scripts/restore_abcd_final_v4b_snapshot.py /tmp/wuziq-v4b-recovery.zip /tmp/wuziq-v4b-restored
```

Drive backup procedure: upload the ZIP as `WuziQ V4B recovery latest.zip`; fetch it back as a streamed file, download the materialized file, compare its SHA-256 with the local archive, then run the restore command and compare completed IDs. Only after the new archive passes both checks may the previous **V4B-named** recovery archive be deleted. Do not delete V1–V4 snapshots or unrelated Drive files. When a run lasts over 24 hours, repeat this replacement process every 24 hours. If local state disappears, download the latest V4B ZIP, restore, check the `progress.json` IDs against `9x9_games.jsonl`, then rerun the schedule; IDs already present are skipped.

No neural network or reinforcement-learning training is part of V4B.
