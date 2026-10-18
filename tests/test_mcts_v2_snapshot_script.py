from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_mcts_v2_snapshot import create_consistent_snapshot
from src.research.checkpoint import restore_snapshot


def test_live_snapshot_contains_one_consistent_unique_game_set(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "config.json").write_text(
        json.dumps({"code_commit": "test-commit"}) + "\n", encoding="utf-8"
    )
    (results / "progress.json").write_text(
        json.dumps({"games_completed": 2, "code_commit": "test-commit"}) + "\n",
        encoding="utf-8",
    )
    (results / "9x9_test_games.jsonl").write_text(
        '{"game_id":"game-1"}\n{"game_id":"game-2"}\n', encoding="utf-8"
    )
    archive = tmp_path / "recovery.zip"

    manifest = create_consistent_snapshot(results, archive)

    restored = tmp_path / "restored"
    restored.mkdir()
    identifiers = restore_snapshot(archive, restored)
    assert sorted(identifiers) == ["game-1", "game-2"]
    assert manifest["games"] == 2
    assert json.loads((restored / "progress.json").read_text())[
        "games_completed"
    ] == 2
