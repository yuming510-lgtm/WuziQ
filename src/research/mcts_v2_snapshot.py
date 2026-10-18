"""Minimal local recovery drill used before a V2 Drive upload."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from typing import Any

from .checkpoint import create_snapshot, restore_snapshot


def run_local_restore_drill(directory: str | Path) -> dict[str, Any]:
    """Create, restore, and corrupt-check a minimal snapshot in a temp folder."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mcts-v2-restore-", dir=root) as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        config = source / "config.json"
        progress = source / "progress.json"
        games = source / "9x9_games.jsonl"
        config.write_text(
            json.dumps({"experiment_id": "mcts-v2-drive-drill", "seed": 731}) + "\n",
            encoding="utf-8",
        )
        progress.write_text(
            json.dumps({"completed": 1, "last_game_id": "drill-game-001"}) + "\n",
            encoding="utf-8",
        )
        games.write_text(
            json.dumps({"game_id": "drill-game-001", "outcome": "draw"}) + "\n",
            encoding="utf-8",
        )
        archive_path = work / "recovery.zip"
        manifest = create_snapshot(config, progress, [games], archive_path)

        destination = work / "restored"
        destination.mkdir()
        unrelated = destination / "keep.txt"
        unrelated.write_text("preserve", encoding="utf-8")
        game_ids = restore_snapshot(archive_path, destination)
        restore_verified = (
            sorted(game_ids) == ["drill-game-001"]
            and (destination / "config.json").read_bytes() == config.read_bytes()
            and (destination / "progress.json").read_bytes() == progress.read_bytes()
            and (destination / "9x9_games.jsonl").read_bytes() == games.read_bytes()
            and unrelated.read_text(encoding="utf-8") == "preserve"
        )

        corrupted = work / "corrupted.zip"
        with zipfile.ZipFile(archive_path, "r") as original:
            entries = {name: original.read(name) for name in original.namelist()}
        entries["9x9_games.jsonl"] = entries["9x9_games.jsonl"].replace(
            b"drill-game-001", b"drill-game-999"
        )
        with zipfile.ZipFile(corrupted, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for name, data in entries.items():
                output.writestr(name, data)
        checksum_corruption_rejected = False
        try:
            restore_snapshot(corrupted, work / "bad-restore")
        except ValueError as exc:
            checksum_corruption_rejected = "checksum mismatch" in str(exc)

        if not restore_verified:
            raise RuntimeError("Local snapshot restore did not reproduce source files")
        if not checksum_corruption_rejected:
            raise RuntimeError("Local restore accepted a checksum-corrupted snapshot")
        return {
            "restore_verified": restore_verified,
            "checksum_corruption_rejected": checksum_corruption_rejected,
            "game_ids": sorted(game_ids),
            "snapshot_bytes": archive_path.stat().st_size,
            "snapshot_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "manifest": manifest,
        }
