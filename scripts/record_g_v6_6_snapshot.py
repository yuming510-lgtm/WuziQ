#!/usr/bin/env python3
"""Record a Drive-verified snapshot and print snapshot files safe to prune."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.research.alphazero_g.v6_6_evaluation import atomic_json, ARTIFACT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    artifact = ROOT / ARTIFACT
    path = artifact / "results/drive_backup_status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "experiment_id": "wuziq-alphazero-g-v6-6-acceptance", "verified_snapshots": [],
        "active_drive_directory": "https://drive.google.com/drive/folders/1uOc9NRe8-GKHeHyoqdesNfyVUZYqj5_R",
    }
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if record.get("readback_sha256") != record.get("archive_sha256") or not record.get("empty_restore_verified"):
        raise SystemExit("refusing to register a snapshot without successful Drive readback and restore")
    existing = [row for row in status.get("verified_snapshots", []) if row.get("file_id") != record.get("file_id")]
    all_rows = existing + [record]
    keep_count = 3 if args.final else 2
    keep = all_rows[-keep_count:]
    remove = all_rows[:-keep_count]
    status.update({
        "verified_snapshots": keep,
        "latest_snapshot_id": record["snapshot_id"],
        "latest_snapshot_sha256": record["archive_sha256"],
        "latest_snapshot_drive_url": record["drive_url"],
        "last_verified_at_utc": record["verified_at_utc"],
        "final_snapshot_id": record["snapshot_id"] if args.final else status.get("final_snapshot_id"),
        "empty_directory_restore_verified": True,
        "sha256_readback_verified": True,
    })
    atomic_json(path, status)
    print(json.dumps({"keep": keep, "delete_file_ids": [row["file_id"] for row in remove]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
