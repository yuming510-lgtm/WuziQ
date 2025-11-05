# Minimalist Gomoku (WuziQ)

A command-line Gomoku implementation written in pure Python. The game defaults to a
15x15 board and supports simple two-player matches with coordinate-based input.

## Installation & Requirements

The project uses only the Python standard library. Ensure Python 3.10+ is
available (for type annotations used in the code).

## Running the Game

You can start a match using either of the following commands:

```bash
python -m src.cli
```

```bash
python src/cli.py
```

Optional flags:

- `--size N` — change the board dimension (minimum 5). For example, to play on a
  10x10 board:

  ```bash
  python -m src.cli --size 10
  ```

During play, enter moves such as `H8` (column letter, row number) or
space-separated numeric coordinates like `8 7`. Enter `quit` or `exit` to end
the session early.

## Example Gameplay

```
Minimalist Gomoku
Board size: 15x15

    A B C D E F G H I J K L M N O
 1  . . . . . . . . . . . . . . .
 2  . . . . . . . . . . . . . . .
 ...
Player B, enter your move: H8
```

## Running Tests

```bash
pytest -q
```

## Design Notes

- **Board Representation:** the board is a `GomokuBoard` data class wrapping a
  2D list of optional player identifiers (`"B"` or `"W"`).
- **Move Parsing:** user input is interpreted by `parse_move`, accepting both
  chess-style (`H8`) and numeric (`8 7`) coordinates.
- **Win Detection:** after each move, `check_victory` counts contiguous stones in
  four directions (horizontal, vertical, and the two diagonals). A run of five or
  more stones signals victory.
- **Command Line Loop:** the CLI alternates between players, printing the board
  after each valid move and exiting when either a player wins or the board fills
  up.
