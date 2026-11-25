from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import torch

from src.research.alphazero_g.v5_training import train_reference_arm
from src.research.alphazero_g.v6_9_training import (
    ARTIFACT, ARM, BASE_GAMES, ROOT, SEED, SIMULATIONS, artifact_path,
    preflight_resources, state_path,
)


def main() -> None:
    torch.set_num_threads(1)
    config_path = artifact_path(ROOT) / "config/reference_training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resources = preflight_resources(ROOT)
    source_state = state_path(ROOT)
    before = torch.load(source_state, map_location="cpu", weights_only=False)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="wuziq_v6_9_preflight_") as scratch_text:
        scratch = Path(scratch_text)
        target = scratch / ARTIFACT / "checkpoints" / ARM / "training_state.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_state, target)
        scratch_config = scratch / ARTIFACT / "config/reference_training_config.json"
        scratch_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, scratch_config)
        pilot = train_reference_arm(
            scratch, arm=ARM, board_size=9, n_in_row=5, simulations=SIMULATIONS,
            target_states=10**12, target_games=BASE_GAMES + 1, milestones=[], seed=SEED,
            config=config, artifact_dir=ARTIFACT,
            progress_experiment_id="wuziq-alphazero-g-v6-9-preflight-excluded",
        )
        raw_path = scratch / ARTIFACT / "raw/selfplay" / ARM / f"{ARM}-game-{BASE_GAMES:06d}.json"
        game = json.loads(raw_path.read_text(encoding="utf-8"))
        result = {
            "pilot_excluded_from_formal_training": True,
            "pilot_game_id": game["game_id"], "pilot_states": game["training_state_count"],
            "pilot_moves": game["move_count"], "pilot_runner_cpu_seconds": game["runner_cpu_seconds"],
            "pilot_complete_training_step_cpu_seconds": pilot["cumulative_cpu_seconds"] - float(before["cumulative_cpu_seconds"]),
            "pilot_wall_seconds": time.perf_counter() - started,
            "resources": resources,
        }
    # Scale the actual pilot by the historical mean game length.  A single
    # game can be much longer/shorter than average, so use the larger of the
    # same-machine estimate and the completed V6.7 historical mean.
    historical_audit_path = ROOT / "research/g_v6_7/results/training_audit.json"
    historical_mean_moves = float(json.loads(historical_audit_path.read_text())["mean_selfplay_moves"])
    pilot_cpu_per_move = game["runner_cpu_seconds"] / max(1, int(game["move_count"]))
    pilot_game_estimate = pilot_cpu_per_move * historical_mean_moves
    historical_game_estimate = resources["historical_cpu_seconds_per_game"]
    normal_cpu_per_game = max(pilot_game_estimate, historical_game_estimate)
    estimate = normal_cpu_per_game * resources["remaining_games"] / 3600
    result["completion_estimates_cpu_hours"] = {
        "optimistic": estimate * 0.8,
        "normal": estimate,
        "conservative": estimate * 1.3,
    }
    result["cost_projection"] = {
        "historical_mean_moves": historical_mean_moves,
        "pilot_cpu_seconds_per_move": pilot_cpu_per_move,
        "pilot_scaled_cpu_seconds_per_game": pilot_game_estimate,
        "historical_cpu_seconds_per_game": historical_game_estimate,
        "selected_normal_cpu_seconds_per_game": normal_cpu_per_game,
        "assumed_training_parallelism": 1,
        "estimated_wall_hours_single_threaded": result["completion_estimates_cpu_hours"],
    }
    output = artifact_path(ROOT) / "results/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
