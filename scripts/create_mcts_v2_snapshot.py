"""Create a self-consistent V2 recovery archive during a live tournament."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.checkpoint import create_snapshot, load_game_records


def create_consistent_snapshot(results_dir: Path, output_path: Path) -> dict:
    for attempt in range(500):
        with tempfile.TemporaryDirectory(prefix="mcts-v2-snapshot-") as name:
            temporary = Path(name)
            before = json.loads((results_dir / "progress.json").read_text(encoding="utf-8"))
            config_copy = temporary / "config.json"
            progress_copy = temporary / "progress.json"
            shutil.copyfile(results_dir / "config.json", config_copy)
            shutil.copyfile(results_dir / "progress.json", progress_copy)
            game_copies = []
            for source in sorted(results_dir.glob("*_games.jsonl")):
                destination = temporary / source.name
                shutil.copyfile(source, destination)
                game_copies.append(destination)
            after = json.loads((results_dir / "progress.json").read_text(encoding="utf-8"))
            try:
                identifiers = [
                    str(row["game_id"])
                    for path in game_copies
                    for row in load_game_records(path)
                ]
            except (OSError, ValueError, json.JSONDecodeError):
                identifiers = []
            coherent = (
                before == after
                and len(identifiers) == len(set(identifiers))
                and len(identifiers) == int(after["games_completed"])
            )
            if coherent:
                manifest = create_snapshot(config_copy, progress_copy, game_copies, output_path)
                digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
                return {
                    "snapshot_path": str(output_path),
                    "games": len(identifiers),
                    "code_commit": after["code_commit"],
                    "bytes": output_path.stat().st_size,
                    "sha256": digest,
                    "manifest": manifest,
                    "attempts": attempt + 1,
                }
        time.sleep(0.05)
    raise TimeoutError("Could not capture a stable progress/JSONL snapshot within 25 seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "research/v2/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_consistent_snapshot(args.results_dir, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
