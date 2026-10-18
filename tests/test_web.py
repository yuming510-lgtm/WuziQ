from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("flask")

from src.webapp import create_app


def build_client():
    app = create_app()
    app.config.update({"TESTING": True})
    return app.test_client()


def test_state_endpoint_returns_board():
    client = build_client()
    response = client.get("/state")
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["size"] == 15
    assert len(data["board"]) == data["size"]
    assert data["ai_model"] == "A"


def test_move_valid_and_invalid():
    client = build_client()
    ok_response = client.post("/move", json={"row": 0, "col": 0})
    assert ok_response.status_code == 200
    ok_data = ok_response.get_json()
    assert ok_data is not None
    assert ok_data["board"][0][0] == "B"
    assert ok_data["current_player"] == "W"

    invalid = client.post("/move", json={"row": 0, "col": 0})
    assert invalid.status_code == 400


def test_undo_restores_previous_state():
    client = build_client()
    client.post("/move", json={"row": 0, "col": 0})
    response = client.post("/undo")
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["board"][0][0] is None
    assert data["current_player"] == "B"


def test_config_triggers_ai_when_applicable():
    client = build_client()
    response = client.post("/config", json={"ai_side": "black"})
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["ai_side"] == "black"
    centre = data["size"] // 2
    assert data["board"][centre][centre] == "B"


@pytest.mark.parametrize("model", ["A", "B", "C", "D"])
def test_config_selects_each_ai_model(model):
    client = build_client()
    response = client.post("/config", json={"ai_model": model})
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["ai_model"] == model


def test_config_rejects_unknown_ai_model():
    client = build_client()
    response = client.post("/config", json={"ai_model": "E"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid ai_model value."


def test_selected_model_plays_black_first_move():
    client = build_client()
    response = client.post("/reset", json={"size": 9, "ai_side": "black", "ai_model": "C"})
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    centre = data["size"] // 2
    assert data["ai_model"] == "C"
    assert data["board"][centre][centre] == "B"


def test_reset_preserves_selected_model_when_omitted():
    client = build_client()
    client.post("/config", json={"ai_model": "D"})
    response = client.post("/reset", json={"size": 9})
    assert response.status_code == 200
    assert response.get_json()["ai_model"] == "D"
