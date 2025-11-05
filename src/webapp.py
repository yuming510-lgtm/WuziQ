"""Flask application providing a minimalist web UI for WuziQ."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

from .gomoku import Coordinate, GomokuBoard, Player, select_ai_move


_VALID_AI_MODES = {"off", "black", "white"}


def _opponent(player: Player) -> Player:
    return "W" if player == "B" else "B"


@dataclass
class GameState:
    """Container storing the mutable state for the web session."""

    board: GomokuBoard
    current_player: Player = "B"
    ai_mode: str = "off"
    winner: Optional[str] = None
    last_move: Optional[Coordinate] = None

    def reset(self, size: int, ai_mode: str) -> None:
        self.board = GomokuBoard(size)
        self.current_player = "B"
        self.ai_mode = ai_mode
        self.winner = None
        self.last_move = None

    @property
    def ai_player(self) -> Optional[Player]:
        if self.ai_mode == "black":
            return "B"
        if self.ai_mode == "white":
            return "W"
        return None


def create_app() -> Flask:
    """Instantiate and return the configured Flask application."""

    base_path = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_path / "templates"),
        static_folder=str(base_path / "static"),
    )

    state = GameState(GomokuBoard())

    def build_payload() -> Dict[str, Any]:
        board_snapshot = [[cell for cell in row] for row in state.board.grid]
        last_move = (
            {"row": state.last_move[0], "col": state.last_move[1]}
            if state.last_move
            else None
        )
        payload: Dict[str, Any] = {
            "size": state.board.size,
            "board": board_snapshot,
            "current_player": state.current_player,
            "ai_side": state.ai_mode,
            "winner": state.winner,
            "is_full": state.board.is_full(),
            "can_undo": bool(state.board.history),
            "last_move": last_move,
        }
        return payload

    def apply_ai_moves_if_needed() -> None:
        while True:
            if state.winner:
                return
            ai_player = state.ai_player
            if not ai_player or state.current_player != ai_player:
                return
            opponent = _opponent(ai_player)
            try:
                coord = select_ai_move(state.board, ai_player, opponent)
            except ValueError:
                # No moves possible; treat as a draw if the board is full.
                state.winner = state.winner or "draw"
                return
            state.board.place_stone(ai_player, coord)
            state.last_move = coord
            if state.board.check_victory(coord, ai_player):
                state.winner = ai_player
                return
            if state.board.is_full():
                state.winner = "draw"
                return
            state.current_player = opponent

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/state")
    def get_state():  # type: ignore[override]
        return jsonify(build_payload())

    @app.post("/move")
    def post_move():  # type: ignore[override]
        if state.winner:
            return jsonify({"error": "Game is over."}), 400
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Missing move payload."}), 400
        try:
            row = int(payload.get("row"))
            col = int(payload.get("col"))
        except (TypeError, ValueError):
            return jsonify({"error": "Row and column must be integers."}), 400

        if not state.board.in_bounds(row, col):
            return jsonify({"error": "Move is out of bounds."}), 400
        if state.board.get((row, col)) is not None:
            return jsonify({"error": "Cell is already occupied."}), 400

        coord = (row, col)
        state.board.place_stone(state.current_player, coord)
        state.last_move = coord

        if state.board.check_victory(coord, state.current_player):
            state.winner = state.current_player
        elif state.board.is_full():
            state.winner = "draw"
        else:
            state.current_player = _opponent(state.current_player)
            apply_ai_moves_if_needed()

        return jsonify(build_payload())

    @app.post("/undo")
    def post_undo():  # type: ignore[override]
        undone = state.board.undo()
        if undone is None:
            return jsonify({"error": "No moves to undo."}), 400
        state.winner = None
        undone_player, _ = undone
        state.current_player = undone_player

        ai_player = state.ai_player
        if ai_player and undone_player == ai_player and state.board.history:
            second = state.board.undo()
            if second:
                state.current_player = second[0]

        state.last_move = state.board.history[-1][1] if state.board.history else None
        apply_ai_moves_if_needed()
        return jsonify(build_payload())

    @app.post("/reset")
    def post_reset():  # type: ignore[override]
        payload = request.get_json(silent=True) or {}
        size = state.board.size
        ai_side = state.ai_mode
        if "size" in payload:
            try:
                size = int(payload["size"])
            except (TypeError, ValueError):
                return jsonify({"error": "Size must be an integer."}), 400
        if "ai_side" in payload:
            ai_side = str(payload["ai_side"])
        if ai_side not in _VALID_AI_MODES:
            return jsonify({"error": "Invalid ai_side value."}), 400
        try:
            state.reset(size, ai_side)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        apply_ai_moves_if_needed()
        return jsonify(build_payload())

    @app.post("/config")
    def post_config():  # type: ignore[override]
        payload = request.get_json(silent=True)
        if not payload or "ai_side" not in payload:
            return jsonify({"error": "Missing ai_side value."}), 400
        ai_side = str(payload["ai_side"])
        if ai_side not in _VALID_AI_MODES:
            return jsonify({"error": "Invalid ai_side value."}), 400
        state.ai_mode = ai_side
        apply_ai_moves_if_needed()
        return jsonify(build_payload())

    return app


def main() -> None:
    """Run the Flask development server."""

    app = create_app()
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
