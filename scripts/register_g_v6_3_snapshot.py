from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "research/g_v6_3"


def register(*, sequence: int, games: int, training_cpu_seconds: float,
             file_name: str, drive_file_id: str, drive_url: str, sha256: str,
             size_bytes: int, code_commit: str, package_cpu_seconds: float,
             download_sha256: str, empty_restore_verified: bool,
             drive_parts: list[dict] | None = None) -> dict:
    if not empty_restore_verified or sha256 != download_sha256:
        raise ValueError("a Drive snapshot can only be registered after SHA and restore verification")
    if drive_parts is not None:
        required = {"file_name", "drive_file_id", "drive_url", "sha256", "size_bytes"}
        if not drive_parts or any(not required.issubset(part) for part in drive_parts):
            raise ValueError("multipart snapshot metadata is incomplete")
        if sum(int(part["size_bytes"]) for part in drive_parts) != int(size_bytes):
            raise ValueError("multipart sizes do not match the archived snapshot size")
    path = ARTIFACT / "results/snapshot_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    records = list(status.get("verified_snapshots", []))
    expected_sequence = max((int(row["sequence"]) for row in records), default=0) + 1
    if int(sequence) != expected_sequence:
        raise ValueError(f"snapshot sequence must be {expected_sequence}")
    row = {
        "sequence": int(sequence), "games_completed": int(games),
        "cumulative_new_training_cpu_seconds": float(training_cpu_seconds),
        "file_name": str(file_name), "drive_file_id": str(drive_file_id),
        "drive_url": str(drive_url), "sha256": str(sha256),
        "download_sha256": str(download_sha256), "size_bytes": int(size_bytes),
        "code_commit": str(code_commit), "package_cpu_seconds": float(package_cpu_seconds),
        "readback_sha256_verified": True, "empty_directory_restore_verified": True,
    }
    if drive_parts is not None:
        row["multipart"] = True
        row["parts"] = drive_parts
    records.append(row)
    status["verified_snapshots"] = records[-2:]
    status["latest_verified_sequence"] = int(sequence)
    status["latest_verified_snapshot"] = row
    status["snapshot_count_total"] = int(sequence)
    status["total_packaging_cpu_seconds"] = float(status.get("total_packaging_cpu_seconds", 0.0)) + float(package_cpu_seconds)
    path.write_text(json.dumps(status, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--training-cpu-seconds", type=float, required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--drive-file-id", required=True)
    parser.add_argument("--drive-url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size-bytes", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--package-cpu-seconds", type=float, default=0.0)
    parser.add_argument("--download-sha256", required=True)
    parser.add_argument("--empty-restore-verified", action="store_true")
    parser.add_argument("--parts-json")
    args = parser.parse_args()
    values = vars(args)
    parts_json = values.pop("parts_json")
    drive_parts = json.loads(Path(parts_json).read_text(encoding="utf-8")) if parts_json else None
    result = register(**values, drive_parts=drive_parts)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
