"""Deterministic exact tactical positions and inference-only V6.1 scoring."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .v5_evaluation import _canonical_position, make_board
from .v5_reference import ReferencePolicyValueNet, instrumented_player


def _other(color: str) -> str:
    return "W" if color == "B" else "B"


class _Grid:
    def __init__(self, moves: list[dict[str, Any]] | None = None) -> None:
        self.size = 9
        self.cells: list[list[str | None]] = [[None] * 9 for _ in range(9)]
        self.history: list[dict[str, Any]] = []
        for item in moves or []:
            self.place(str(item["player"]), (int(item["row"]), int(item["col"])))

    def place(self, color: str, pos: tuple[int, int]) -> None:
        r, c = pos
        if color not in {"B", "W"} or not (0 <= r < 9 and 0 <= c < 9) or self.cells[r][c] is not None:
            raise ValueError("illegal tactical history move")
        self.cells[r][c] = color
        self.history.append({"player": color, "row": r, "col": c})

    def undo(self) -> None:
        item = self.history.pop()
        self.cells[item["row"]][item["col"]] = None

    def won(self, pos: tuple[int, int], color: str) -> bool:
        r, c = pos
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < 9 and 0 <= cc < 9 and self.cells[rr][cc] == color:
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= 5:
                return True
        return False

    def wins(self, color: str) -> list[int]:
        result = []
        for r in range(9):
            for c in range(9):
                if self.cells[r][c] is not None:
                    continue
                self.cells[r][c] = color
                won = self.won((r, c), color)
                self.cells[r][c] = None
                if won:
                    result.append(r * 9 + c)
        return result

    def has_any_win(self) -> bool:
        for r in range(9):
            for c in range(9):
                color = self.cells[r][c]
                if color and self.won((r, c), color):
                    return True
        return False

    def board(self):
        result = make_board(9, 5, 0)
        for item in self.history:
            result.do_move(item["row"] * 9 + item["col"])
        return result

    def key(self, to_move: str) -> str:
        board = self.board()
        expected = "B" if board.current_player == 1 else "W"
        if expected != to_move:
            raise ValueError("tactical side-to-move mismatch")
        return _canonical_position(board, board.current_player)


def _closed_four(grid: _Grid, color: str) -> bool:
    # A 4-cell uninterrupted run with no free endpoint is a plausible-looking
    # but non-winning closed four. It is not itself scored as a forced win.
    for r in range(9):
        for c in range(9):
            if grid.cells[r][c] != color:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                points = [(r + i * dr, c + i * dc) for i in range(4)]
                if any(not (0 <= rr < 9 and 0 <= cc < 9) for rr, cc in points):
                    continue
                if all(grid.cells[rr][cc] == color for rr, cc in points):
                    ends = ((r - dr, c - dc), (r + 4 * dr, c + 4 * dc))
                    if all(not (0 <= rr < 9 and 0 <= cc < 9) or grid.cells[rr][cc] == _other(color)
                           for rr, cc in ends):
                        return True
    return False


def _forced_threat_actions(grid: _Grid, attacker: str) -> list[int]:
    """Prove a <=3-ply win against every legal defense, with exact rule checks."""
    if grid.wins(attacker) or grid.wins(_other(attacker)):
        return []
    forced: list[int] = []
    for r in range(9):
        for c in range(9):
            if grid.cells[r][c] is not None:
                continue
            move = (r, c)
            grid.place(attacker, move)
            # A two-end immediate threat leaves at least one winning point after
            # any single defensive stone. Still verify every legal response.
            root_wins = grid.wins(attacker)
            candidate_proved = len(root_wins) >= 2 and not grid.wins(_other(attacker))
            if candidate_proved:
                defender = _other(attacker)
                for rr in range(9):
                    if not candidate_proved:
                        break
                    for cc in range(9):
                        if grid.cells[rr][cc] is not None:
                            continue
                        reply = (rr, cc)
                        grid.place(defender, reply)
                        defender_won = grid.won(reply, defender)
                        still_winning = bool(grid.wins(attacker))
                        grid.undo()
                        if defender_won or not still_winning:
                            candidate_proved = False
                            break
            grid.undo()
            if candidate_proved:
                forced.append(r * 9 + c)
    return forced


def _position_from_template(category: str, rng: np.random.Generator) -> tuple[_Grid, str, list[int]]:
    """Build a legal alternating history, then derive its labels by the oracle."""
    grid = _Grid()
    # The motif position is randomized; D4 canonical de-duplication is applied
    # after construction. Extreme rows are sampled deliberately for edge cases.
    row = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]))
    col = int(rng.integers(0, 5))
    if category in {"direct_win", "forced_threat_3ply"}:
        if category == "direct_win":
            row = int(rng.choice([0, 1, 2, 4, 7, 8]))
            start = int(rng.integers(0, 5))
            bline = [(row, start + i) for i in range(4)]
            # One half of the direct-win set has one winning endpoint; the rest
            # keep both endpoints open and test multiple optimal actions.
            one_target = bool(rng.integers(0, 2))
            wline = []
            for c in range(9):
                if (row, c) in bline:
                    continue
                if c in (start - 1, start + 4) and 0 <= c < 9:
                    if one_target and not wline:
                        wline.append((row, c))
            # For one target, fill an endpoint blocker if the edge does not do so.
            if one_target and len(wline) == 0:
                if start - 1 >= 0:
                    wline = [(row, start - 1)]
                elif start + 4 < 9:
                    wline = [(row, start + 4)]
            bfill = [pos for pos in bline]
            # Four alternating B motif stones and four W fillers.
            fillers = [(rr, cc) for rr in range(9) for cc in range(9)
                       if (rr, cc) not in bline and (rr, cc) not in wline
                       and rr != row and abs(rr - row) >= 2]
            rng.shuffle(fillers)
            wfill = (wline + [pos for pos in fillers if pos not in wline])[:4]
            if len(wfill) < 4:
                raise RuntimeError("could not fill direct-win template")
            for index in range(4):
                grid.place("B", bline[index])
                if grid.won(bline[index], "B"):
                    raise RuntimeError("terminal prefix in direct-win template")
                grid.place("W", wfill[index])
                if grid.won(wfill[index], "W"):
                    raise RuntimeError("terminal prefix in direct-win template")
            return grid, "B", grid.wins("B")

        # Three stones plus a legal extension to an open four create two
        # immediate winning endpoints. The exact verifier checks every reply.
        start = int(rng.integers(1, 5))
        bline = [(row, start + i) for i in range(3)]
        fillers = [(rr, cc) for rr in range(9) for cc in range(9)
                   if rr != row or abs(cc - start) > 4]
        rng.shuffle(fillers)
        wfill = fillers[:3]
        for index in range(3):
            grid.place("B", bline[index])
            grid.place("W", wfill[index])
        return grid, "B", _forced_threat_actions(grid, "B")

    if category == "unique_block":
        row = int(rng.choice([1, 2, 3, 4, 5, 6, 7]))
        start = int(rng.integers(0, 5))
        wline = [(row, start + i) for i in range(4)]
        block_cell = (row, start - 1) if start > 0 else (row, start + 4)
        if not (0 <= block_cell[1] < 9):
            raise RuntimeError("invalid unique-block edge")
        filler_pool = [(rr, cc) for rr in range(9) for cc in range(9)
                       if rr != row and abs(rr - row) > 1]
        rng.shuffle(filler_pool)
        bfillers = [block_cell] + [p for p in filler_pool if p != block_cell][:3]
        wfillers = [p for p in ((0, 0), (0, 8), (8, 0), (8, 8), (4, 0), (4, 8), (0, 4), (8, 4))
                    if p not in wline and p != block_cell]
        rng.shuffle(wfillers)
        wi = 0
        bi = 0
        for ply in range(8):
            color = "B" if ply % 2 == 0 else "W"
            if color == "B":
                pos = bfillers[bi]
                bi += 1
            else:
                pos = wline[wi] if wi < 4 else wfillers[wi - 4]
                wi += 1
            grid.place(color, pos)
            if grid.won(pos, color):
                raise RuntimeError("terminal prefix in unique-block template")
        return grid, "B", grid.wins("W")

    if category == "false_threat":
        # B has a closed three-stone development threat. W has a four with one
        # endpoint blocked by B, so W still has one immediate win. The unique
        # exact defense is to stop W; pursuing B's tempting line loses now.
        row_b = int(rng.choice([0, 1, 2, 6, 7, 8]))
        start_b = int(rng.integers(0, 6))
        bline = [(row_b, start_b + i) for i in range(3)]
        allowed_w_rows = [rr for rr in range(9) if rr != row_b and min((rr-row_b) % 9, (row_b-rr) % 9) >= 2]
        row_w = int(rng.choice(allowed_w_rows))
        start_w = int(rng.integers(1, 5))
        wline = [(row_w, start_w + i) for i in range(4)]
        block_cell = (row_w, start_w - 1) if start_w > 0 else (row_w, start_w + 4)
        if not (0 <= block_cell[1] < 9):
            raise RuntimeError("invalid false-threat endpoint")
        filler = [(rr, cc) for rr in range(9) for cc in range(9)
                  if rr not in (row_b, row_w) and (rr, cc) != block_cell]
        rng.shuffle(filler)
        sequence = [
            ("B", bline[0]), ("W", wline[0]), ("B", bline[1]), ("W", wline[1]),
            ("B", bline[2]), ("W", wline[2]), ("B", block_cell), ("W", wline[3]),
        ]
        for color, pos in sequence:
            grid.place(color, pos)
            if grid.won(pos, color):
                raise RuntimeError("terminal prefix in false-threat template")
        return grid, "B", grid.wins("W")
    raise ValueError(category)


def generate_tactical_suite(*, seed: int = 610612, per_category: int = 32,
                            historical_keys: set[str] | None = None) -> list[dict[str, Any]]:
    """Create at least 128 distinct exact positions; no teacher or training data."""
    rng = np.random.default_rng(int(seed))
    quotas = {"direct_win": per_category, "unique_block": per_category,
              "forced_threat_3ply": per_category, "false_threat": per_category}
    found = {name: [] for name in quotas}
    seen = set(historical_keys or set())
    attempts = 0
    maximum = per_category * 500
    while any(len(found[key]) < quotas[key] for key in quotas):
        attempts += 1
        if attempts > maximum:
            counts = {name: len(rows) for name, rows in found.items()}
            raise RuntimeError(f"tactical generator exhausted exact legal templates: {counts}")
        category = next(name for name in quotas if len(found[name]) < quotas[name])
        try:
            grid, to_move, targets = _position_from_template(category, rng)
        except RuntimeError:
            continue
        if grid.has_any_win() or len(grid.history) < 1 or not targets:
            continue
        # Verify action labels independently from the template construction.
        own_wins = grid.wins(to_move)
        other_wins = grid.wins(_other(to_move))
        if category == "direct_win" and set(targets) != set(own_wins):
            continue
        if category in {"unique_block", "false_threat"}:
            if own_wins or len(other_wins) != 1 or set(targets) != set(other_wins):
                continue
        if category == "forced_threat_3ply":
            exact = _forced_threat_actions(grid, to_move)
            if set(targets) != set(exact) or not exact:
                continue
        key = grid.key(to_move)
        if key in seen:
            continue
        seen.add(key)
        category_index = len(found[category])
        edge_tag = any((action // 9 in {0, 1, 7, 8} or action % 9 in {0, 1, 7, 8})
                       for action in targets)
        tags = []
        if edge_tag:
            tags.append("edge_corner")
        if len(targets) > 1:
            tags.append("multiple_exact_winning_actions")
        if category == "forced_threat_3ply":
            tags.extend(["continuous_threat", "all_legal_defenses_verified"])
        if category == "false_threat":
            tags.append("false_attack_must_be_ignored_for_unique_block")
        value_label = 1 if category in {"direct_win", "forced_threat_3ply"} else None
        puzzle = {
            "puzzle_id": f"v61-9x9-{category}-{category_index:03d}",
            "category": category,
            "board_size": 9,
            "n_in_row": 5,
            "to_move": to_move,
            "moves": grid.history,
            "target_actions": sorted(set(map(int, targets))),
            "unique_solution": len(set(targets)) == 1,
            "solved_value_to_move": value_label,
            "tags": tags,
            "position_key": key,
            "label_source": "independent exhaustive legal-action five-in-a-row oracle",
            "proof_scope": ("all legal opponent replies checked; forced win within three plies"
                            if category == "forced_threat_3ply" else
                            "all legal immediate winning/blocking actions enumerated"),
        }
        found[category].append(puzzle)
    puzzles = [puzzle for category in quotas for puzzle in found[category]]
    # IDs and positions must be globally unique even after symmetry canonicalization.
    if len(puzzles) != len({puzzle["position_key"] for puzzle in puzzles}):
        raise AssertionError("tactical positions are not D4-distinct")
    return puzzles


def validate_tactical_suite(puzzles: list[dict[str, Any]], *, expected_count: int = 128) -> dict[str, Any]:
    if len(puzzles) < expected_count:
        raise ValueError("V6.1 new tactical suite is smaller than registered minimum")
    ids, keys, counts = set(), set(), {}
    edge, multiple, proof_checks = 0, 0, 0
    for puzzle in puzzles:
        if puzzle["puzzle_id"] in ids or puzzle["position_key"] in keys:
            raise ValueError("duplicate tactical ID or D4-equivalent position")
        ids.add(puzzle["puzzle_id"])
        keys.add(puzzle["position_key"])
        grid = _Grid(puzzle["moves"])
        if grid.has_any_win():
            raise ValueError("tactical history includes a terminal position")
        to_move = str(puzzle["to_move"])
        targets = set(map(int, puzzle["target_actions"]))
        if not targets:
            raise ValueError("tactical target set is empty")
        category = puzzle["category"]
        if category == "direct_win":
            if targets != set(grid.wins(to_move)):
                raise ValueError("direct-win targets differ from exhaustive exact oracle")
        elif category in {"unique_block", "false_threat"}:
            if grid.wins(to_move) or targets != set(grid.wins(_other(to_move))) or len(targets) != 1:
                raise ValueError("unique-block labels differ from exact immediate-win enumeration")
            if category == "false_threat" and not any(
                sum(grid.cells[r][c] == to_move for c in range(9)) >= 3 for r in range(9)
            ):
                raise ValueError("false-threat puzzle lacks the registered attack motif")
        elif category == "forced_threat_3ply":
            proof_checks += 1
            if targets != set(_forced_threat_actions(grid, to_move)):
                raise ValueError("forced-threat target failed all-defense proof replay")
        else:
            raise ValueError(f"unknown tactical category {category}")
        counts[category] = counts.get(category, 0) + 1
        edge += int("edge_corner" in puzzle.get("tags", []))
        multiple += int(len(targets) > 1)
    if sum(counts.values()) < expected_count:
        raise ValueError("tactical quota was not met")
    return {"puzzles": len(puzzles), "unique_positions": len(keys), "by_category": counts,
            "edge_corner_positions": edge, "multi_solution_positions": multiple,
            "forced_threats_all_defenses_replayed": proof_checks,
            "labels_verified_by_independent_rule_oracle": True}


def _normalize_historical(puzzle: dict[str, Any], index: int) -> dict[str, Any]:
    size = int(puzzle.get("board_size", 9))
    moves = puzzle.get("moves", puzzle.get("position", []))
    targets = puzzle.get("target_actions", puzzle.get("optimal_actions", []))
    result = {
        **puzzle,
        "puzzle_id": str(puzzle.get("puzzle_id", f"v6-old-{index:03d}")),
        "category": str(puzzle.get("category", "historical")),
        "board_size": size,
        "n_in_row": int(puzzle.get("n_in_row", 5)),
        "moves": moves,
        "target_actions": list(map(int, targets)),
        "solved_value_to_move": puzzle.get("solved_value_to_move"),
    }
    return result


def evaluate_tactics(model: ReferencePolicyValueNet, puzzles: list[dict[str, Any]], *,
                     simulations: int, seed: int, output_path: str | Path | None = None,
                     cpu_deadline: float | None = None) -> dict[str, Any]:
    if simulations not in {64, 400}:
        raise ValueError("registered V6.1 tactical budgets are 64 and 400")
    output = Path(output_path) if output_path is not None else None
    rows: list[dict[str, Any]] = []
    if output is not None and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if int(previous.get("simulations", -1)) != simulations:
            raise ValueError("partial tactic results use a different search budget")
        rows = list(previous.get("rows", []))
    completed = {row["puzzle_id"] for row in rows}
    for index, raw in enumerate(puzzles):
        if raw["puzzle_id"] in completed:
            continue
        if cpu_deadline is not None and time.process_time() >= cpu_deadline:
            break
        puzzle = _normalize_historical(raw, index)
        board = make_board(9, 5, 0)
        for item in puzzle["moves"]:
            board.do_move(int(item["row"]) * 9 + int(item["col"]))
        side = "B" if board.current_player == 1 else "W"
        if side != puzzle["to_move"] or board.game_end()[0]:
            raise ValueError(f"invalid tactical position: {puzzle['puzzle_id']}")
        targets = set(puzzle["target_actions"])
        if not targets:
            raise ValueError("unlabeled historical puzzle cannot be scored")
        priors, net_value = model.policy_value_fn(board)
        legal_probs = {int(action): float(prob) for action, prob in priors}
        if set(legal_probs) - set(board.availables):
            raise ValueError("policy assigned probabilities to illegal moves")
        legal_mass = float(sum(legal_probs.values()))
        mass = float(sum(legal_probs.get(action, 0.0) for action in targets))
        net_action = max(legal_probs, key=lambda action: (legal_probs[action], -action))
        player = instrumented_player(model.policy_value_fn, c_puct=5.0,
                                     n_playout=simulations + 1, is_selfplay=False)
        # temp=1 returns visits normalized exactly as N_a / sum(N_a), retaining
        # the full root action distribution instead of a sampled move.
        acts, visit_probs = player.mcts.get_move_probs(board, temp=1.0)
        visit_dist = {int(action): float(prob) for action, prob in zip(acts, visit_probs)}
        puct_action = max(visit_dist, key=lambda action: (visit_dist[action], -action))
        target_mass = float(sum(visit_dist.get(action, 0.0) for action in targets))
        exact_value = puzzle.get("solved_value_to_move")
        rows.append({
            "puzzle_id": puzzle["puzzle_id"],
            "category": puzzle["category"],
            "targets": sorted(targets),
            "unique_solution": len(targets) == 1,
            "network_action": int(net_action),
            "network_correct": net_action in targets,
            "network_optimal_probability_mass": mass,
            "network_legal_probability_mass": legal_mass,
            "network_optimal_probability_mass_conditioned_on_legal": mass / legal_mass if legal_mass > 0 else 0.0,
            "network_value_to_move": float(net_value),
            "puct_action": int(puct_action),
            "puct_correct": puct_action in targets,
            "puct_optimal_visit_mass": target_mass,
            "puct_root_visits": int(player.last_search_stats.get("root_visits", 0)),
            "puct_root_children": int(player.last_search_stats.get("root_children", 0)),
            "search_depth": int(player.last_search_stats.get("tree_depth", 0)),
            "tree_nodes": int(player.last_search_stats.get("tree_nodes", 0)),
            "search_cpu_seconds": float(player.last_search_stats.get("search_cpu_seconds", 0.0)),
            "search_root_value": player.last_search_stats.get("root_network_value"),
            "exact_value_to_move": exact_value,
            "value_absolute_error": (abs(float(net_value) - float(exact_value))
                                     if exact_value is not None else None),
        })
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix(output.suffix + ".tmp")
            with temp.open("w", encoding="utf-8") as handle:
                json.dump({"simulations": simulations, "rows": rows}, handle,
                          ensure_ascii=False, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, output)
    categories = sorted({row["category"] for row in rows})
    solved = [row for row in rows if row["exact_value_to_move"] is not None]
    result = {
        "simulations": simulations,
        "puzzles": len(rows),
        "complete": len(rows) == len(puzzles),
        "network_top1_accuracy": float(np.mean([row["network_correct"] for row in rows])) if rows else None,
        "network_optimal_action_probability_mass": float(np.mean([row["network_optimal_probability_mass"] for row in rows])) if rows else None,
        "network_legal_probability_mass": float(np.mean([row["network_legal_probability_mass"] for row in rows])) if rows else None,
        "network_optimal_mass_conditioned_on_legal": float(np.mean([row["network_optimal_probability_mass_conditioned_on_legal"] for row in rows])) if rows else None,
        "puct_accuracy": float(np.mean([row["puct_correct"] for row in rows])) if rows else None,
        "puct_optimal_visit_mass": float(np.mean([row["puct_optimal_visit_mass"] for row in rows])) if rows else None,
        "root_visit_count_min": min((row["puct_root_visits"] for row in rows), default=0),
        "root_visit_count_max": max((row["puct_root_visits"] for row in rows), default=0),
        "solved_value_positions": len(solved),
        "value_mae_on_proven_values": float(np.mean([row["value_absolute_error"] for row in solved])) if solved else None,
        "value_mean_on_proven_wins": float(np.mean([row["network_value_to_move"] for row in solved])) if solved else None,
        "by_category": {
            category: {
                "puzzles": sum(row["category"] == category for row in rows),
                "network_top1_accuracy": float(np.mean([row["network_correct"] for row in rows if row["category"] == category])),
                "network_target_probability_mass": float(np.mean([row["network_optimal_probability_mass"] for row in rows if row["category"] == category])),
                "puct_accuracy": float(np.mean([row["puct_correct"] for row in rows if row["category"] == category])),
                "puct_target_visit_mass": float(np.mean([row["puct_optimal_visit_mass"] for row in rows if row["category"] == category])),
            } for category in categories
        },
        "rows": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, output)
    return result
