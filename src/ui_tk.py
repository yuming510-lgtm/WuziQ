"""Tkinter-based desktop interface for the WuziQ Gomoku game."""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from src.gomoku import GomokuBoard, select_ai_move

Player = str
Coordinate = Tuple[int, int]
AI_MODES = ("off", "black", "white")


def configure_high_dpi() -> None:
    """Apply a best-effort DPI awareness hint on Windows platforms."""

    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            # The call fails on older Windows versions or if the flag is unsupported.
            pass


@dataclass
class GameState:
    """Bundle together the dynamic state of the Gomoku match."""

    board: GomokuBoard
    next_player: Player = "B"
    ai_mode: str = "off"
    last_move: Optional[Coordinate] = None
    game_over: bool = False


class GomokuApp:
    """Tkinter application presenting a playable Gomoku board."""

    def __init__(self, size: int, ai_mode: str) -> None:
        if size < 5:
            raise ValueError("Board size must be at least 5.")

        if ai_mode not in AI_MODES:
            raise ValueError(f"Unsupported AI mode: {ai_mode}")

        configure_high_dpi()

        self.root = tk.Tk()
        self.root.title("WuziQ Gomoku")

        self.state = GameState(board=GomokuBoard(size), ai_mode=ai_mode)
        self.ai_player = self._ai_player_from_mode(ai_mode)

        self.cell_size = max(28, min(52, int(620 / size)))
        self.margin = self.cell_size
        board_pixels = self.cell_size * (size - 1)
        canvas_size = board_pixels + self.margin * 2

        self.turn_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.ai_var = tk.StringVar()

        outer = tk.Frame(self.root, padx=12, pady=12)
        outer.pack(fill=tk.BOTH, expand=True)

        top_bar = tk.Frame(outer)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        self.turn_label = tk.Label(top_bar, textvariable=self.turn_var, font=("Segoe UI", 12, "bold"))
        self.turn_label.pack(side=tk.LEFT)

        self.status_label = tk.Label(top_bar, textvariable=self.status_var, font=("Segoe UI", 11))
        self.status_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            outer,
            width=canvas_size,
            height=canvas_size,
            background="#f2d5a9",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        bottom_bar = tk.Frame(outer)
        bottom_bar.pack(fill=tk.X, pady=(8, 0))

        undo_button = tk.Button(bottom_bar, text="Undo", command=self._on_undo, width=8)
        undo_button.pack(side=tk.LEFT, padx=(0, 6))

        reset_button = tk.Button(bottom_bar, text="Reset", command=self._on_reset, width=8)
        reset_button.pack(side=tk.LEFT, padx=(0, 6))

        self.ai_button = tk.Button(bottom_bar, text="", command=self._toggle_ai, width=12)
        self.ai_button.pack(side=tk.LEFT)

        self.ai_status_label = tk.Label(bottom_bar, textvariable=self.ai_var)
        self.ai_status_label.pack(side=tk.RIGHT)

        self._update_status("Click a grid intersection to place a stone.")
        self._draw_board()
        self._maybe_trigger_ai()

    def run(self) -> None:
        """Start the Tkinter main loop."""

        self.root.mainloop()

    def _update_status(self, message: Optional[str] = None) -> None:
        """Refresh the informational labels with the current state."""

        if message is not None:
            self.status_var.set(message)

        if self.state.game_over:
            self.turn_var.set("Game over")
        else:
            player_name = "Black" if self.state.next_player == "B" else "White"
            self.turn_var.set(f"Turn: {player_name} ({self.state.next_player})")

        ai_readable = self.state.ai_mode.capitalize()
        self.ai_var.set(f"AI side: {ai_readable}")
        self.ai_button.config(text=f"AI: {ai_readable}")

    def _draw_board(self) -> None:
        """Render the board grid and any placed stones onto the canvas."""

        self.canvas.delete("all")
        size = self.state.board.size
        board_pixels = self.cell_size * (size - 1)
        start = self.margin
        end = start + board_pixels

        # Draw grid lines.
        for idx in range(size):
            offset = start + idx * self.cell_size
            self.canvas.create_line(start, offset, end, offset, fill="#8c6239")
            self.canvas.create_line(offset, start, offset, end, fill="#8c6239")

        radius = self.cell_size * 0.38
        for row, row_cells in enumerate(self.state.board.grid):
            for col, cell in enumerate(row_cells):
                if cell is None:
                    continue
                center_x = start + col * self.cell_size
                center_y = start + row * self.cell_size
                x0 = center_x - radius
                y0 = center_y - radius
                x1 = center_x + radius
                y1 = center_y + radius
                fill = "#111111" if cell == "B" else "#f8f8f8"
                outline = "#000000" if cell == "B" else "#b4b4b4"
                self.canvas.create_oval(x0, y0, x1, y1, fill=fill, outline=outline, width=2)

        if self.state.last_move is not None:
            row, col = self.state.last_move
            center_x = start + col * self.cell_size
            center_y = start + row * self.cell_size
            highlight_radius = radius + 4
            self.canvas.create_oval(
                center_x - highlight_radius,
                center_y - highlight_radius,
                center_x + highlight_radius,
                center_y + highlight_radius,
                outline="#d63b3b",
                width=2,
            )

    def _on_canvas_click(self, event: tk.Event[tk.Canvas]) -> None:
        """Handle mouse clicks by mapping pixels to board coordinates."""

        if self.state.game_over:
            return

        if self._ai_controls(self.state.next_player):
            return

        row, col = self._pixel_to_coord(event.x, event.y)
        if row is None or col is None:
            return

        self._handle_move((row, col))

    def _handle_move(self, coord: Coordinate) -> None:
        """Attempt to place a stone for the current player at ``coord``."""

        current_player = self.state.next_player
        try:
            self.state.board.place_stone(current_player, coord)
        except ValueError as exc:
            self._update_status(str(exc))
            return

        self.state.last_move = coord
        if self.state.board.check_victory(coord, current_player):
            winner = "Black" if current_player == "B" else "White"
            self.state.game_over = True
            self._update_status(f"{winner} wins!")
        elif self.state.board.is_full():
            self.state.game_over = True
            self._update_status("The board is full. It's a draw.")
        else:
            self.state.next_player = "W" if current_player == "B" else "B"
            self._update_status()

        self._draw_board()

        if not self.state.game_over:
            self._maybe_trigger_ai()

    def _pixel_to_coord(self, x: int, y: int) -> Tuple[Optional[int], Optional[int]]:
        """Convert a pixel coordinate to a board coordinate."""

        start = self.margin
        end = start + self.cell_size * (self.state.board.size - 1)
        if x < start - self.cell_size / 2 or y < start - self.cell_size / 2:
            return None, None
        if x > end + self.cell_size / 2 or y > end + self.cell_size / 2:
            return None, None

        col = round((x - start) / self.cell_size)
        row = round((y - start) / self.cell_size)
        if not self.state.board.in_bounds(row, col):
            return None, None
        return row, col

    def _on_undo(self) -> None:
        """Undo the latest move (AI and human if necessary)."""

        undone = self.state.board.undo()
        if undone is None:
            self._update_status("No moves to undo.")
            return

        undone_player, _ = undone
        self.state.next_player = undone_player

        if self._ai_controls(self.state.next_player) and self.state.board.history:
            second = self.state.board.undo()
            if second is not None:
                self.state.next_player = second[0]

        self.state.last_move = self.state.board.history[-1][1] if self.state.board.history else None
        self.state.game_over = False
        self._update_status("Moves undone.")
        self._draw_board()
        self._maybe_trigger_ai()

    def _on_reset(self) -> None:
        """Reset the board to a fresh game state."""

        self.state = GameState(board=GomokuBoard(self.state.board.size), ai_mode=self.state.ai_mode)
        self.ai_player = self._ai_player_from_mode(self.state.ai_mode)
        self._update_status("New game started.")
        self._draw_board()
        self._maybe_trigger_ai()

    def _toggle_ai(self) -> None:
        """Cycle through available AI modes."""

        current_index = AI_MODES.index(self.state.ai_mode)
        new_mode = AI_MODES[(current_index + 1) % len(AI_MODES)]
        self.state.ai_mode = new_mode
        self.ai_player = self._ai_player_from_mode(new_mode)
        if self.state.game_over:
            self._update_status("AI mode updated. Reset to play again.")
            return

        self._update_status()
        self._maybe_trigger_ai()

    def _maybe_trigger_ai(self) -> None:
        """Schedule an AI move if the current side is automated."""

        if self.state.game_over:
            return

        if not self._ai_controls(self.state.next_player):
            return

        self.root.after(150, self._perform_ai_move)

    def _perform_ai_move(self) -> None:
        """Execute an AI move when it is the AI's turn."""

        if self.state.game_over:
            return

        if not self._ai_controls(self.state.next_player):
            return

        ai_player = self.state.next_player
        opponent = "W" if ai_player == "B" else "B"
        try:
            coord = select_ai_move(self.state.board, ai_player, opponent)
        except ValueError as exc:
            self._update_status(str(exc))
            return

        self._handle_move(coord)

    def _ai_player_from_mode(self, mode: str) -> Optional[Player]:
        """Return the player identifier controlled by the AI for ``mode``."""

        if mode == "black":
            return "B"
        if mode == "white":
            return "W"
        return None

    def _ai_controls(self, player: Player) -> bool:
        """Return ``True`` if ``player`` is currently AI-controlled."""

        return self.ai_player == player


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Tkinter UI."""

    parser = argparse.ArgumentParser(description="Launch the WuziQ desktop UI.")
    parser.add_argument("--size", type=int, default=15, help="Board dimension (default: 15).")
    parser.add_argument(
        "--ai",
        choices=AI_MODES,
        default="off",
        help="Enable the lightweight AI for the specified colour.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point for running the Tkinter application."""

    args = parse_args(argv)
    try:
        app = GomokuApp(args.size, args.ai)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    app.run()


if __name__ == "__main__":
    main()
