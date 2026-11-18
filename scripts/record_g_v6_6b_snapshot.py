#!/usr/bin/env python3
"""Record an already verified upload or a successful prune in Drive status."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "research/g_v6_6b/results/drive_backup_status.json"


def save(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--label", required=True)
    add.add_argument("--file-id", required=True)
    add.add_argument("--url", required=True)
    add.add_argument("--sha256", required=True)
    add.add_argument("--completed-games", required=True, type=int)
    add.add_argument("--size-bytes", required=True, type=int)
    prune = sub.add_parser("pruned")
    prune.add_argument("--file-id", required=True)
    args = parser.parse_args()
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    if args.command == "add":
        if any(row["file_id"] == args.file_id or row["label"] == args.label for row in payload["snapshots"]):
            raise ValueError("duplicate V6.6B Drive snapshot record")
        payload["snapshots"].append({
            "label": args.label, "file_id": args.file_id, "url": args.url,
            "sha256": args.sha256, "completed_games": args.completed_games,
            "size_bytes": args.size_bytes, "download_sha_verified": True,
            "empty_directory_restore": True, "deleted": False,
        })
        payload["latest_verified_label"] = args.label
    else:
        matches = [row for row in payload["snapshots"] if row["file_id"] == args.file_id]
        if len(matches) != 1 or matches[0].get("deleted"):
            raise ValueError("Drive prune target is missing or already deleted")
        matches[0]["deleted"] = True
        matches[0]["deletion_verified"] = True
    save(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
