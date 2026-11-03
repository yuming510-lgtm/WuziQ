"""Atomic result and progress files for resumable V4.2 work."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def game_result_path(root: str | Path, namespace: str, game_id: str) -> Path:
    key = hashlib.sha256(game_id.encode("utf-8")).hexdigest()
    return Path(root) / "research/g_v4_2/results/games" / namespace / f"{key}.json"


def persist_unique_game(root: str | Path, namespace: str, game: dict[str, Any]) -> bool:
    game_id = game.get("game_id")
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("a V4.2 game must have a stable unique game_id")
    path = game_result_path(root, namespace, game_id)
    if path.exists():
        existing = load_json(path)
        if existing.get("game_id") != game_id or canonical_json(existing) != canonical_json(game):
            raise ValueError(f"conflicting duplicate V4.2 game id: {game_id}")
        return False
    atomic_write_json(path, game)
    return True


def iter_games(root: str | Path, namespace: str) -> list[dict[str, Any]]:
    directory = Path(root) / "research/g_v4_2/results/games" / namespace
    if not directory.exists():
        return []
    output = []
    seen: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        item = load_json(path)
        game_id = item.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError(f"result without game_id: {path}")
        encoded = canonical_json(item)
        if game_id in seen:
            if seen[game_id] != encoded:
                raise ValueError(f"duplicate conflicting game_id in {namespace}: {game_id}")
            raise ValueError(f"duplicate V4.2 game_id in {namespace}: {game_id}")
        seen[game_id] = encoded
        output.append(item)
    return output
