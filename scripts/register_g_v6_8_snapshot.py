from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.research.alphazero_g.v6_8_tournament import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a fully verified V6.8 Drive recovery snapshot")
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=("initial", "rolling", "final"), required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--download-sha256", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--restored-files", type=int, required=True)
    args = parser.parse_args()
    if args.sha256 != args.download_sha256:
        raise SystemExit("refusing to register a Drive snapshot whose downloaded SHA differs")
    path = Path("research/g_v6_8/results/drive_backup_status.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = {"label": args.label, "kind": args.kind, "games_completed": args.games,
             "drive_file_id": args.file_id, "drive_url": args.url,
             "local_sha256": args.sha256, "download_sha256": args.download_sha256,
             "bytes": args.bytes, "empty_directory_restore": True,
             "restored_files_verified": args.restored_files,
             "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    old = next((row for row in data.get("snapshots", []) if row.get("label") == args.label), None)
    if old and (old.get("drive_file_id") != args.file_id or old.get("local_sha256") != args.sha256):
        raise SystemExit(f"snapshot label {args.label} already exists with different identity")
    if not old:
        data.setdefault("snapshots", []).append(entry)
    data["latest_verified_snapshot"] = args.label
    data["latest_games"] = args.games
    data["last_verified_at_utc"] = entry["verified_at_utc"]
    data["write_read_sha_restore_verified"] = True
    atomic_json(path, data)
    print(json.dumps(entry, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
