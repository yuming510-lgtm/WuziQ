import numpy as np
import torch
import json

from third_party.AlphaZero_Gomoku.game import Board, Game
from third_party.AlphaZero_Gomoku.mcts_alphaZero import MCTS
from src.research.alphazero_g.v5_evaluation import generate_opening_pool, play_match, summarize_paired_games
from src.research.alphazero_g.v5_audit import audit_evaluation_game
from src.research.alphazero_g.v5_reference import ReferencePolicyValueNet, instrumented_player, make_board
from src.research.alphazero_g.v5_training import augment_reference_data


def _oracle_winner(grid, n):
    size = len(grid)
    for row in range(size):
        for col in range(size):
            player = grid[row][col]
            if player == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r, end_c = row + (n - 1) * dr, col + (n - 1) * dc
                if not (0 <= end_r < size and 0 <= end_c < size):
                    continue
                if all(grid[row + i * dr][col + i * dc] == player for i in range(n)):
                    return int(player)
    return None


def test_reference_board_matches_plain_array_oracle_for_6x6_and_9x9():
    for size, n_in_row, seed in ((6, 4, 611), (9, 5, 919)):
        rng = np.random.default_rng(seed)
        for game_index in range(4):
            board = Board(width=size, height=size, n_in_row=n_in_row)
            board.init_board(start_player=game_index % 2)
            grid = [[0 for _ in range(size)] for _ in range(size)]
            expected_player = 1 if game_index % 2 == 0 else 2
            for _ in range(size * size):
                legal = list(board.availables)
                action = int(rng.choice(legal))
                assert action in board.availables
                board.do_move(action)
                row, col = divmod(action, size)
                grid[row][col] = expected_player
                oracle = _oracle_winner(grid, n_in_row)
                ended, winner = board.game_end()
                assert bool(ended) == (oracle is not None or not board.availables)
                if oracle is not None:
                    assert winner == oracle == expected_player
                    break
                if not board.availables:
                    assert winner == -1
                    break
                expected_player = 3 - expected_player


def test_reference_current_state_encodes_player_relative_stones_and_turn():
    board = Board(width=6, height=6, n_in_row=4)
    board.init_board(start_player=0)
    board.do_move(3 * 6 + 2)  # black
    state_white = board.current_state()
    assert state_white[1, 2, 2] == 1.0  # black is the opponent to white
    assert state_white[2, 2, 2] == 1.0  # last action
    assert state_white[3].sum() == 0.0  # player two to move
    board.do_move(4 * 6 + 1)  # white
    state = board.current_state()
    assert state.shape == (4, 6, 6)
    # Returned feature maps are vertically flipped to the upstream CNN view.
    assert state[0, 2, 2] == 1.0  # black is now to move
    assert state[1, 1, 1] == 1.0  # white is the opponent
    assert state[2, 1, 1] == 1.0  # last action
    assert state[3].sum() == 36.0  # player one to move


def test_reference_puct_finds_exact_immediate_win_and_backs_up_winner_view():
    board = Board(width=6, height=6, n_in_row=4)
    board.init_board()
    for col in range(3):
        board.do_move(2 * 6 + col)
        board.do_move(5 * 6 + col)
    winning_action = 2 * 6 + 3

    def policy(state):
        priors = [(move, 1.0 if move == winning_action else 1e-4) for move in state.availables]
        total = sum(value for _, value in priors)
        return [(move, value / total) for move, value in priors], 0.0

    search = MCTS(policy, c_puct=5, n_playout=2)
    actions, probabilities = search.get_move_probs(board, temp=1e-3)
    assert actions[int(np.argmax(probabilities))] == winning_action
    child = search._root._children[winning_action]
    assert child._Q == 1.0
    assert child._n_visits == 1
    assert sum(node._n_visits for node in search._root._children.values()) == 1


def test_reference_train_targets_are_current_player_relative_and_modern_torch_compatible():
    class ScriptedPlayer:
        def __init__(self):
            self.actions = [0, 30, 1, 31, 2, 32, 3]

        def set_player_ind(self, player):
            self.player = player

        def reset_player(self):
            pass

        def get_action(self, board, temp=1e-3, return_prob=0):
            action = self.actions.pop(0)
            probs = np.zeros(board.width * board.height)
            probs[action] = 1.0
            return (action, probs) if return_prob else action

    board = Board(width=6, height=6, n_in_row=4)
    game = Game(board)
    winner, data = game.start_self_play(ScriptedPlayer(), is_shown=0, temp=1.0)
    rows = list(data)
    assert winner == 1
    assert [float(row[2]) for row in rows] == [1, -1, 1, -1, 1, -1, 1]
    assert all(np.isclose(row[1].sum(), 1.0) for row in rows)

    torch.set_num_threads(1)
    model = ReferencePolicyValueNet(6, seed=88)
    states = np.zeros((8, 4, 6, 6), dtype=np.float32)
    targets = np.full((8, 36), 1 / 36, dtype=np.float32)
    values = np.zeros(8, dtype=np.float32)
    loss, entropy = model.train_step(states, targets, values, 0.002)
    assert np.isfinite(loss) and np.isfinite(entropy)


def test_reference_augmentation_preserves_normalized_visit_distributions():
    state = np.zeros((4, 6, 6), dtype=np.float32)
    state[0, 1, 1] = 1
    policy = np.zeros(36, dtype=np.float32)
    policy[[2, 12, 27]] = [0.2, 0.3, 0.5]
    transformed = augment_reference_data([(state, policy, 1.0)])
    assert len(transformed) == 8
    assert all(np.isclose(row[1].sum(), 1.0) for row in transformed)
    assert all(row[2] == 1.0 for row in transformed)


def test_opening_pool_is_legal_nonterminal_d4_unique_and_deterministic():
    first = generate_opening_pool(size=6, n_in_row=4, groups=20, plies=4, seed=811)
    second = generate_opening_pool(size=6, n_in_row=4, groups=20, plies=4, seed=811)
    assert first == second
    assert len({repr(row["moves"]) for row in first}) == 20
    board = make_board(6, 4)
    for item in first:
        board = make_board(6, 4)
        for move in item["moves"]:
            assert move["player"] == ("B" if board.current_player == 1 else "W")
            action = move["row"] * 6 + move["col"]
            assert action in board.availables
            board.do_move(action)
            assert not board.game_end()[0]


def test_paired_score_summary_requires_color_swap_and_uses_pair_bootstrap():
    games = [
        {"game_id": "g1b", "opening_group_id": "o1", "candidate_color": "B", "points": 1.0,
         "move_count": 15, "candidate_cpu_seconds": 1.0, "opponent_cpu_seconds": 0.4},
        {"game_id": "g1w", "opening_group_id": "o1", "candidate_color": "W", "points": 0.0,
         "move_count": 16, "candidate_cpu_seconds": 1.1, "opponent_cpu_seconds": 0.5},
        {"game_id": "g2b", "opening_group_id": "o2", "candidate_color": "B", "points": 0.5,
         "move_count": 18, "candidate_cpu_seconds": 1.2, "opponent_cpu_seconds": 0.5},
        {"game_id": "g2w", "opening_group_id": "o2", "candidate_color": "W", "points": 0.5,
         "move_count": 18, "candidate_cpu_seconds": 1.0, "opponent_cpu_seconds": 0.5},
    ]
    summary = summarize_paired_games(games, seed=91)
    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 2, 1)
    assert summary["points"] == 2.0
    assert summary["score_rate"] == 0.5
    assert summary["paired_openings"] == 2


def test_complete_evaluation_game_replays_against_independent_oracle(tmp_path):
    torch.set_num_threads(1)
    opening = generate_opening_pool(size=6, n_in_row=4, groups=1, plies=4, seed=88)[0]
    model = ReferencePolicyValueNet(6, seed=99)
    game = play_match(
        game_id="v5-audit-smoke", candidate_name="ref-candidate",
        opponent_name="ref-opponent", opening=opening, candidate_color="B",
        seed=100, candidate_reference_model=model,
        opponent_reference_model=model, simulations=2, board_size=6, n_in_row=4,
    )
    path = tmp_path / "v5-audit-smoke.json"
    path.write_text(json.dumps(game), encoding="utf-8")
    assert audit_evaluation_game(path)["game_id"] == "v5-audit-smoke"
