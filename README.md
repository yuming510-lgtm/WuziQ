# Minimalist Gomoku (WuziQ)

A command-line Gomoku implementation written in pure Python. The game defaults to a
15x15 board and supports simple two-player matches with coordinate-based input.

## Installation & Requirements

The core game logic depends only on the Python standard library. To use the web
interface install the additional dependency listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Web UI

Launch the lightweight Flask server and open the exposed address (default
`http://127.0.0.1:8000`) in a browser:

```bash
python -m src.webapp
```

The server binds to `0.0.0.0:8000`, making it suitable for Codespaces or other
remote development environments—simply forward port 8000 and open it from your
local browser. The page lets you click cells to place stones, undo moves, reset
the board, or change the AI side.

## Running the Game (CLI)

You can start a terminal match using either of the following commands:

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

- `--ai {off,black,white}` — enable a simple AI opponent. `black` lets the AI
  play first, `white` makes the AI respond to the human's black stones. The
  default is `off` (two human players).

During play you can enter moves such as `H8` (column letter, row number) or
space-separated numeric coordinates like `8 7`. Game commands are prefixed with a
colon:

- `:undo` — undo the previous move (in AI games this rewinds the AI and human
  moves together).
- `:save filename.json` — store the current game state to disk.
- `:load filename.json` — resume a saved game.
- `:quit` — abort the match.

## Example Gameplay

```
Minimalist Gomoku
Board size: 10x10

    A B C D E F G H I J
 1  . . . . . . . . . .
 2  . . . . . . . . . .
 ...
Player B, enter your move: H8

AI (W) plays H9.

Player B, enter your move: :save demo.json
Game saved to demo.json.

Player B, enter your move: :undo
Undid the latest AI and player moves to maintain turn order.
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
