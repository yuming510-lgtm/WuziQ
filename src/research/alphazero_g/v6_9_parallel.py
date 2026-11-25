"""V6.9 game-level parallel self-play throughput experiment.

The training process owns the replay buffer and Adam optimizer. Each worker
plays one complete game with an immutable policy snapshot; workers receive the
newest weights when they start their next game. Each completed game triggers
exactly one call to the V6.7/V5 reference update function.
"""
from __future__ import annotations

import os

# Set these before importing NumPy or PyTorch so each process uses one BLAS/OpenMP thread.
for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import re
import threading
import time
from typing import Any

import numpy as np
import psutil
import torch

from . import v5_training, v6_2_training, v6_5_audit, v6_5_training, v6_7_training
from .v5_reference import ReferencePolicyValueNet


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = Path("research/g_v6_9")
ARM = "reference400_empty"
BASE_GAMES = 20_000
BASE_STATES = 348_216
BASE_UPDATES = 19_999
SIMULATIONS = 400
TRAINING_SEED = 590_502
BASE_STATE_SHA256 = "f932ec34eaf4da6ac63bc77b6636e0517235a8986bbcf981139afa4cae7b7719"
BASE_ARCHIVE_SHA256 = "5d41512351907c2653a7018569e2200aa2a8a4e22c1d439baca03794f8be43f1"
BASE_COMMIT = "911242c10c8a374740db072f71a6b5f450a837e6"
GAME_SEED_STRIDE = 10_007

_WORKER_MODEL: ReferencePolicyValueNet | None = None


def _atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _policy_snapshot(model: ReferencePolicyValueNet) -> dict[str, np.ndarray]:
    # Plain NumPy payloads use multiprocessing's ordinary pickle transport. PyTorch
    # tensor reducers require a Unix resource-sharer socket that Work disallows.
    return {name: tensor.detach().cpu().numpy().copy() for name, tensor in model.get_policy_param().items()}


def _worker_initializer() -> None:
    global _WORKER_MODEL
    torch.set_num_threads(1)
    # Spawned workers need to set this explicitly. Forked workers inherit the
    # trainer's one-thread setting and PyTorch rejects setting it a second time.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    _WORKER_MODEL = ReferencePolicyValueNet(9, seed=TRAINING_SEED)


def _worker_play(job: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        raise RuntimeError("self-play worker was not initialized")
    cpu_started = time.process_time()
    worker_weights = {
        name: torch.from_numpy(np.ascontiguousarray(array))
        for name, array in job["model_state"].items()
    }
    _WORKER_MODEL.load_policy_param(worker_weights)
    record, states, policies, values = v5_training._run_game(
        size=9,
        n_in_row=5,
        simulations=SIMULATIONS,
        model=_WORKER_MODEL,
        game_id=str(job["game_id"]),
        seed=int(job["seed"]),
        opening=[],
    )
    return {
        "ordinal": int(job["ordinal"]),
        "game_id": str(job["game_id"]),
        "seed": int(job["seed"]),
        "model_version_used": int(job["model_version"]),
        "worker_pid": os.getpid(),
        "worker_task_cpu_seconds": time.process_time() - cpu_started,
        "worker_finished_monotonic": time.perf_counter(),
        "record": record,
        "states": states,
        "policies": policies,
        "values": values,
    }


class _ProcessTreeMemorySampler:
    """Sample the trainer plus its worker tree; the peak covers measured training only."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = float(interval_seconds)
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._lock = threading.Lock()
        self._peak_bytes = 0
        self._thread = threading.Thread(target=self._sample, name="rss-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def enable(self) -> None:
        with self._lock:
            self._peak_bytes = 0
        self._enabled.set()

    def disable(self) -> int:
        self._enabled.clear()
        with self._lock:
            return self._peak_bytes

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _sample(self) -> None:
        while not self._stop.is_set():
            if self._enabled.is_set():
                total = 0
                processes = [self._process]
                try:
                    processes.extend(self._process.children(recursive=True))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                for process in processes:
                    try:
                        total += int(process.memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                with self._lock:
                    self._peak_bytes = max(self._peak_bytes, total)
            self._stop.wait(self.interval_seconds)


def _adam_step(model: ReferencePolicyValueNet) -> int:
    state = model.optimizer.state_dict().get("state", {})
    if not state:
        return 0
    first = next(iter(state.values()))
    step = first.get("step", 0)
    return int(step.item()) if isinstance(step, torch.Tensor) else int(step)


def _load_model_and_state(
    checkpoint_path: Path,
    config: dict[str, Any],
) -> tuple[ReferencePolicyValueNet, tuple[int, int, int, float, deque, int, float]]:
    model = ReferencePolicyValueNet(9, seed=TRAINING_SEED)
    loaded = v5_training.load_training_checkpoint(
        checkpoint_path,
        model=model,
        arm=ARM,
        config_hash=str(config["config_sha256"]),
        buffer_size=int(config["training"]["buffer_size"]),
    )
    return model, loaded


def _write_progress(path: Path, *, run_id: str, status: str, values: dict[str, Any]) -> None:
    _atomic_json(path, {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-9-parallel-efficiency",
        "run_id": run_id,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    })


def _validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
        raise ValueError("run_id may contain only ASCII letters, digits, '_' and '-' (maximum 48 characters)")
    return value


def run_one(
    *,
    run_id: str,
    workers: int,
    games: int = 100,
    baseline_checkpoint: str | Path,
    output_root: str | Path | None = None,
    root: str | Path = ROOT,
    start_method: str = "spawn",
) -> dict[str, Any]:
    """Run one independently restored condition, with one warm-up update then measured games."""
    run_id = _validate_run_id(run_id)
    if workers not in (1, 2, 4, 6, 8):
        raise ValueError("workers must be one of 1, 2, 4, 6 or 8")
    if start_method not in mp.get_all_start_methods():
        raise ValueError(f"unsupported multiprocessing start method: {start_method}")
    if games <= 0:
        raise ValueError("games must be positive")
    root = Path(root).resolve()
    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    output_root = Path(output_root) if output_root is not None else root / ARTIFACT / "runs"
    if not output_root.is_absolute():
        output_root = root / output_root
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoints" / "training_state.pt"
    progress_path = run_dir / "results" / "progress.json"
    metrics_path = run_dir / "results" / "per_game.jsonl"
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(baseline_checkpoint)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config_path = root / "research/g_v6_7/config/reference_training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_config = config["training"]
    state_validation = v6_7_training.validate_training_state(
        baseline_checkpoint, root=root, expected_games=BASE_GAMES,
    )
    if state_validation["sha256"] != BASE_STATE_SHA256:
        raise ValueError("G20000 checkpoint SHA-256 differs from the registered V6.7 full state")
    if (state_validation["environment_states"], state_validation["optimizer_updates"],
            state_validation["replay_rows"]) != (BASE_STATES, BASE_UPDATES, int(training_config["buffer_size"])):
        raise ValueError("G20000 counters or replay capacity differ from the registered V6.7 checkpoint")

    prior_progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    if checkpoint_path.is_file():
        if prior_progress.get("status") == "complete":
            raise FileExistsError(f"run {run_id} is already complete; use a new run_id for a repeat")
        current_validation = v6_5_training.validate_training_state(checkpoint_path, root=root)
        game_count = int(current_validation["game_count"])
        if not BASE_GAMES <= game_count <= BASE_GAMES + games:
            raise ValueError("existing V6.9 checkpoint is outside this run's resume boundary")
        if int(current_validation["optimizer_updates"]) != BASE_UPDATES + game_count - BASE_GAMES:
            raise ValueError("resume checkpoint optimizer updates do not match completed games")
        initial_status = "resuming"
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        game_count = BASE_GAMES
        initial_status = "warming_up"
        provenance_path = run_dir / "results" / "baseline_provenance.json"
        _atomic_json(provenance_path, {
            "baseline_commit": BASE_COMMIT,
            "baseline_checkpoint_sha256": state_validation["sha256"],
            "baseline_archive_sha256": BASE_ARCHIVE_SHA256,
            "games_completed": BASE_GAMES,
            "environment_states": BASE_STATES,
            "optimizer_updates": BASE_UPDATES,
            "replay_rows": state_validation["replay_rows"],
            "replay_sha256": state_validation["replay_sha256"],
            "config_sha256": config["config_sha256"],
            "torch_version": torch.__version__,
        })

    completed_before = game_count - BASE_GAMES
    if completed_before > games:
        raise ValueError("run contains more completed games than requested")
    if metrics_path.is_file():
        metric_rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        metric_rows = []
    if len(metric_rows) == completed_before + 1:
        # Metrics are fsynced immediately before the matching full checkpoint.
        # If interrupted in that narrow window, the checkpoint remains authoritative.
        metric_rows = metric_rows[:completed_before]
        _atomic_jsonl(metrics_path, metric_rows)
    if len(metric_rows) != completed_before:
        raise ValueError("per-game timing rows do not reconcile with the resumable checkpoint")
    committed_ordinals = [int(row["ordinal"]) for row in metric_rows]
    if len(set(committed_ordinals)) != len(committed_ordinals):
        raise ValueError("resumable per-game timing rows contain duplicate game ordinals")
    committed_ordinal_set = set(committed_ordinals)

    state_to_load = checkpoint_path if checkpoint_path.is_file() else baseline_checkpoint
    model, loaded = _load_model_and_state(state_to_load, config)
    game_count, state_count, update_count, lr_multiplier, replay, saved_seed, baseline_cpu = loaded
    if saved_seed != TRAINING_SEED:
        raise ValueError("loaded state counters or training seed failed the V6.9 identity check")
    if completed_before == 0 and (game_count, state_count, update_count) != (BASE_GAMES, BASE_STATES, BASE_UPDATES):
        raise ValueError("initial measured run did not load exact G20000 counters")
    if completed_before > 0 and (game_count, update_count) != (BASE_GAMES + completed_before, BASE_UPDATES + completed_before):
        raise ValueError("resume state counters do not match completed V6.9 games")

    sampler = _ProcessTreeMemorySampler()
    context = mp.get_context(start_method)
    preheat_started_wall = time.perf_counter()
    preheat_started_cpu = time.process_time()
    preheat_task_cpu = 0.0
    preheat_parent_cpu = 0.0
    preheat_wall_seconds = 0.0
    prior_elapsed_wall = float(prior_progress.get("elapsed_wall_seconds", 0.0)) if completed_before else 0.0
    prior_aggregate_cpu = float(prior_progress.get("aggregate_cpu_seconds", 0.0)) if completed_before else 0.0
    prior_worker_cpu = float(sum(float(row.get("worker_task_cpu_seconds", 0.0)) for row in metric_rows))
    total_states_added = int(sum(int(row["training_state_count"]) for row in metric_rows))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_initializer) as executor:
        initial_weights = _policy_snapshot(model)
        warmup_futures = [executor.submit(_worker_play, {
            "ordinal": -index - 1,
            "game_id": f"warmup-{run_id}-{index:02d}",
            "seed": TRAINING_SEED + (BASE_GAMES + 100_000 + index) * GAME_SEED_STRIDE,
            "model_version": update_count,
            "model_state": initial_weights,
        }) for index in range(workers)]
        if warmup_futures:
            warmup_results = [future.result() for future in warmup_futures]
            preheat_task_cpu = sum(float(item["worker_task_cpu_seconds"]) for item in warmup_results)
            warmup = warmup_results[0]
            replay.extend(v5_training.augment_reference_data([
                (warmup["states"][i], warmup["policies"][i], float(warmup["values"][i]))
                for i in range(len(warmup["values"]))
            ]))
            if len(replay) <= int(training_config["batch_size"]):
                raise ValueError("warm-up did not populate enough replay rows for the original update flow")
            v5_training._policy_update(
                model, replay,
                batch_size=int(training_config["batch_size"]),
                epochs=int(training_config["epochs"]),
                learning_rate=float(training_config["learning_rate"]),
                kl_target=float(training_config["kl_target"]),
                lr_multiplier=float(lr_multiplier),
            )
            del warmup_results, warmup
            # Throw away warm-up data and optimizer changes; reload the exact start/resume point.
            model, loaded = _load_model_and_state(state_to_load, config)
            game_count, state_count, update_count, lr_multiplier, replay, saved_seed, baseline_cpu = loaded
            initial_status = "measuring"
        else:
            initial_weights = _policy_snapshot(model)

        preheat_wall_seconds = time.perf_counter() - preheat_started_wall
        preheat_parent_cpu = time.process_time() - preheat_started_cpu

        # For fork, create all workers before the auxiliary sampler thread exists.
        sampler.start()
        sampler.enable()
        wall_started = time.perf_counter()
        parent_cpu_started = time.process_time()
        worker_cpu_seconds = prior_worker_cpu
        submitted_context: dict[Any, dict[str, Any]] = {}
        pending_ordinals = iter(ordinal for ordinal in range(games) if ordinal not in committed_ordinal_set)

        def submit_next(ordinal: int) -> None:
            version = int(update_count)
            game_index = BASE_GAMES + ordinal
            game_id = f"{run_id}-game-{game_index:06d}"
            seed = TRAINING_SEED + game_index * GAME_SEED_STRIDE
            submit_perf = time.perf_counter()
            snapshot = _policy_snapshot(model)
            future = executor.submit(_worker_play, {
                "ordinal": ordinal,
                "game_id": game_id,
                "seed": seed,
                "model_version": version,
                "model_state": snapshot,
            })
            submitted_context[future] = {
                "ordinal": ordinal,
                "game_id": game_id,
                "seed": seed,
                "model_version": version,
                "submitted_monotonic": submit_perf,
            }

        while len(submitted_context) < workers:
            ordinal = next(pending_ordinals, None)
            if ordinal is None:
                break
            submit_next(ordinal)

        while submitted_context:
            done, _ = wait(tuple(submitted_context), return_when=FIRST_COMPLETED)
            finished = []
            for future in done:
                context_row = submitted_context.pop(future)
                result = future.result()
                finished.append((float(result["worker_finished_monotonic"]), result, context_row))
            for _, result, context_row in sorted(finished, key=lambda item: item[0]):
                ordinal = int(context_row["ordinal"])
                if ordinal in {int(row["ordinal"]) for row in metric_rows}:
                    raise ValueError("duplicate completed game ordinal")
                record = result["record"]
                if int(record.get("simulations_configured_per_move", -1)) != SIMULATIONS:
                    raise ValueError("worker returned a game with a reduced PUCT budget")
                if int(record.get("training_state_count", -1)) != len(result["values"]):
                    raise ValueError("worker training samples do not match recorded decision count")

                updates_before = int(update_count)
                weight_lag = updates_before - int(result["model_version_used"])
                if weight_lag < 0:
                    raise ValueError("worker used a model version newer than the trainer")
                record["parallel_training"] = {
                    "worker_count": int(workers),
                    "worker_pid": int(result["worker_pid"]),
                    "model_version_used": int(result["model_version_used"]),
                    "optimizer_updates_before_game_update": updates_before,
                    "weight_lag_updates": weight_lag,
                    "optimizer_updates_after_game_update": updates_before + 1,
                }
                save_started = time.perf_counter()
                v5_training._save_game(
                    root, ARM, record, result["states"], result["policies"], result["values"],
                    artifact_dir=run_dir,
                )
                save_wall = time.perf_counter() - save_started

                augment_started = time.perf_counter()
                play_data = [
                    (result["states"][i], result["policies"][i], float(result["values"][i]))
                    for i in range(len(result["values"]))
                ]
                augmented = v5_training.augment_reference_data(play_data)
                replay.extend(augmented)
                augment_wall = time.perf_counter() - augment_started
                if len(replay) <= int(training_config["batch_size"]):
                    raise ValueError("replay buffer cannot supply the registered training batch")

                adam_step_before = _adam_step(model)
                update_wall_started = time.perf_counter()
                update_cpu_started = time.process_time()
                loss, entropy, lr_multiplier = v5_training._policy_update(
                    model, replay,
                    batch_size=int(training_config["batch_size"]),
                    epochs=int(training_config["epochs"]),
                    learning_rate=float(training_config["learning_rate"]),
                    kl_target=float(training_config["kl_target"]),
                    lr_multiplier=float(lr_multiplier),
                )
                update_cpu = time.process_time() - update_cpu_started
                update_wall = time.perf_counter() - update_wall_started
                update_count += 1
                game_count += 1
                state_count += int(record["training_state_count"])
                total_states_added += int(record["training_state_count"])
                if int(update_count) != int(game_count) - 1:
                    raise ValueError("one optimizer update was not applied for this completed game")

                row = {
                    "run_id": run_id,
                    "ordinal": ordinal,
                    "game_id": str(record["game_id"]),
                    "seed": int(result["seed"]),
                    "winner_color": record["winner_color"],
                    "termination": record["termination"],
                    "move_count": int(record["move_count"]),
                    "training_state_count": int(record["training_state_count"]),
                    "simulations_configured_per_move": SIMULATIONS,
                    "model_version_used": int(result["model_version_used"]),
                    "updates_before": updates_before,
                    "updates_after": int(update_count),
                    "weight_lag_updates": weight_lag,
                    "worker_pid": int(result["worker_pid"]),
                    "worker_task_cpu_seconds": float(result["worker_task_cpu_seconds"]),
                    "selfplay_cpu_seconds": float(record["runner_cpu_seconds"]),
                    "selfplay_wall_seconds": float(record["runner_wall_seconds"]),
                "game_finished_elapsed_seconds": float(prior_elapsed_wall + result["worker_finished_monotonic"] - wall_started),
                "game_submitted_elapsed_seconds": float(prior_elapsed_wall + context_row["submitted_monotonic"] - wall_started),
                    "record_sample_save_wall_seconds": float(save_wall),
                    "augmentation_wall_seconds": float(augment_wall),
                    "optimizer_update_wall_seconds": float(update_wall),
                    "optimizer_update_cpu_seconds": float(update_cpu),
                    "adam_step_before": adam_step_before,
                    "adam_step_after": _adam_step(model),
                    "training_update_calls": 1,
                    "loss": float(loss),
                    "policy_loss": float(model.last_policy_loss),
                    "value_loss": float(model.last_value_loss),
                    "entropy": float(entropy),
                    "lr_multiplier": float(lr_multiplier),
                    "replay_rows": len(replay),
                    "replay_rows_added_before_capacity": len(augmented),
                    "completed_at_elapsed_seconds": float(prior_elapsed_wall + time.perf_counter() - wall_started),
                }
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                metric_rows.append(row)
                worker_cpu_seconds += float(result["worker_task_cpu_seconds"])

                parent_cpu_elapsed = time.process_time() - parent_cpu_started
                current_worker_cpu = worker_cpu_seconds - prior_worker_cpu
                checkpoint_payload = v5_training._checkpoint_payload(
                    model,
                    arm=ARM,
                    config_hash=str(config["config_sha256"]),
                    game_count=game_count,
                    state_count=state_count,
                    update_count=update_count,
                    lr_multiplier=lr_multiplier,
                    replay=replay,
                    base_seed=TRAINING_SEED,
                    cumulative_cpu_seconds=float(baseline_cpu + parent_cpu_elapsed + current_worker_cpu),
                )
                _atomic_torch_save(checkpoint_path, checkpoint_payload)

                measured_elapsed = prior_elapsed_wall + time.perf_counter() - wall_started
                measured_cpu = prior_aggregate_cpu + time.process_time() - parent_cpu_started + current_worker_cpu
                _write_progress(progress_path, run_id=run_id, status="running", values={
                    "workers": int(workers),
                    "games_requested": int(games),
                    "games_completed": len(metric_rows),
                    "game_count_cumulative": int(game_count),
                    "training_states_cumulative": int(state_count),
                    "new_training_states": int(total_states_added),
                    "optimizer_updates_cumulative": int(update_count),
                    "optimizer_update_calls": len(metric_rows),
                    "elapsed_wall_seconds": measured_elapsed,
                    "aggregate_cpu_seconds": measured_cpu,
                    "average_effective_cpu_cores": measured_cpu / measured_elapsed if measured_elapsed else None,
                    "max_weight_lag_updates": max(int(r["weight_lag_updates"]) for r in metric_rows),
                    "max_process_tree_rss_bytes": sampler._peak_bytes,
                    "checkpoint_sha256": _sha_file(checkpoint_path),
                })

                next_ordinal = next(pending_ordinals, None)
                if next_ordinal is not None:
                    submit_next(next_ordinal)
                completed_before += 1

        # No worker remains in a game. Stop timing after the final full update and checkpoint.
        final_wall_seconds = prior_elapsed_wall + time.perf_counter() - wall_started
        final_parent_cpu = time.process_time() - parent_cpu_started
        final_worker_cpu = worker_cpu_seconds
        final_cpu_seconds = prior_aggregate_cpu + final_parent_cpu + (final_worker_cpu - prior_worker_cpu)
        peak_rss = sampler.disable()

    sampler.close()

    node_path = run_dir / "checkpoints" / f"node_G{game_count}.pt"
    node_payload = {
        "format_version": 1,
        "arm_id": ARM,
        "board_size": 9,
        "n_in_row": 5,
        "simulations_per_move": SIMULATIONS,
        "environment_states": int(state_count),
        "game_count": int(game_count),
        "games_completed": int(game_count),
        "optimizer_updates": int(update_count),
        "update_count": int(update_count),
        "model_state": {key: value.detach().cpu() for key, value in model.get_policy_param().items()},
        "config_sha256": str(config["config_sha256"]),
        "source_training_state_sha256": _sha_file(checkpoint_path),
    }
    _atomic_torch_save(node_path, node_payload)

    # Full-state round trip uses the same V6.5 validator as the V6.7 baseline.
    final_validation = v6_5_training.validate_training_state(checkpoint_path, root=root)
    final_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not v6_5_training._state_equal(final_state["model_state"], node_payload["model_state"]):
        raise ValueError("saved node weights differ from the full training checkpoint")
    baseline_payload = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    if v6_5_training._state_equal(final_state["model_state"], baseline_payload["model_state"]):
        raise ValueError("model weights did not change during the measured updates")
    if v6_2_training._replay_digest(final_state["replay_buffer"]) == v6_2_training._replay_digest(baseline_payload["replay_buffer"]):
        raise ValueError("replay buffer did not change during the measured self-play games")
    if (int(final_validation["game_count"]) != BASE_GAMES + games
            or int(final_validation["optimizer_updates"]) != BASE_UPDATES + games
            or int(final_validation["replay_rows"]) != int(training_config["buffer_size"])):
        raise ValueError("final game/update/replay counters differ from the measured workload")

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != games or len({row["game_id"] for row in rows}) != games:
        raise ValueError("official game timing rows have missing or duplicate IDs")
    if any(int(row["training_update_calls"]) != 1 or int(row["updates_after"]) != int(row["updates_before"]) + 1 for row in rows):
        raise ValueError("at least one official game did not receive exactly one update call")

    total_moves = sum(int(row["move_count"]) for row in rows)
    total_states = sum(int(row["training_state_count"]) for row in rows)
    if total_states != int(final_state["environment_states"]) - BASE_STATES:
        raise ValueError("per-game state counts do not reconcile with the final training checkpoint")
    speed = games * 3600.0 / final_wall_seconds
    summary = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-9-parallel-efficiency",
        "run_id": run_id,
        "workers": int(workers),
        "games_measured": int(games),
        "base": {
            "commit": BASE_COMMIT,
            "game_count": BASE_GAMES,
            "environment_states": BASE_STATES,
            "optimizer_updates": BASE_UPDATES,
            "training_state_sha256": state_validation["sha256"],
            "archive_sha256": BASE_ARCHIVE_SHA256,
        },
        "protocol": {
            "process_start_method": start_method,
            "board_size": 9,
            "n_in_row": 5,
            "simulations_per_move": SIMULATIONS,
            "batch_size": int(training_config["batch_size"]),
            "epochs": int(training_config["epochs"]),
            "learning_rate": float(training_config["learning_rate"]),
            "kl_target": float(training_config["kl_target"]),
            "replay_capacity": int(training_config["buffer_size"]),
            "training_update_calls_per_game": 1,
            "worker_torch_interop_threads": 1,
            "worker_torch_intraop_threads": 1,
            "blas_threads": 1,
            "seed_plan": "590502 + cumulative_game_index * 10007; same ordinal seeds for every condition",
        },
        "preheat": {
            "completed_warmup_games": int(workers),
            "warmup_wall_seconds": float(preheat_wall_seconds),
            "warmup_parent_cpu_seconds": float(preheat_parent_cpu),
            "warmup_worker_task_cpu_seconds": float(preheat_task_cpu),
            "warmup_update_discarded_and_baseline_reloaded": True,
        },
        "measurement": {
            "wall_seconds": float(final_wall_seconds),
            "parent_cpu_seconds": float(prior_aggregate_cpu - prior_worker_cpu + final_parent_cpu),
            "worker_task_cpu_seconds": float(final_worker_cpu),
            "aggregate_cpu_seconds": float(final_cpu_seconds),
            "average_effective_cpu_cores": float(final_cpu_seconds / final_wall_seconds),
            "games_per_hour": float(speed),
            "training_states": int(total_states),
            "training_states_per_second": float(total_states / final_wall_seconds),
            "mean_selfplay_moves_per_game": float(total_moves / games),
            "cpu_seconds_per_game": float(final_cpu_seconds / games),
            "max_process_tree_rss_bytes": int(peak_rss),
            "worker_cpu_accounting": "sum of each worker task's process_time; trainer CPU uses process_time; worker startup and IPC after task return excluded",
        },
        "weight_sync": {
            "mean_lag_updates": float(np.mean([row["weight_lag_updates"] for row in rows])),
            "median_lag_updates": float(np.median([row["weight_lag_updates"] for row in rows])),
            "p95_lag_updates": float(np.percentile([row["weight_lag_updates"] for row in rows], 95)),
            "max_lag_updates": int(max(row["weight_lag_updates"] for row in rows)),
            "mean_optimizer_update_wall_seconds": float(np.mean([row["optimizer_update_wall_seconds"] for row in rows])),
            "adam_step_delta": int(final_validation["optimizer_updates"] - BASE_UPDATES),
        },
        "correctness": {
            "full_checkpoint_valid": bool(final_validation["valid"]),
            "model_updated": True,
            "adam_restored": bool(final_validation["adam_restored"]),
            "replay_restored": bool(final_validation["replay_restored"]),
            "python_rng_restored": bool(final_validation["python_rng_restored"]),
            "numpy_rng_restored": bool(final_validation["numpy_rng_restored"]),
            "torch_rng_restored": bool(final_validation["torch_rng_restored"]),
            "unique_game_ids": len({row["game_id"] for row in rows}),
            "unique_seeds": len({int(row["seed"]) for row in rows}),
            "game_replay_audit_pending": True,
        },
        "checkpoint": {
            "training_state_path": str(checkpoint_path.relative_to(root)),
            "training_state_sha256": final_validation["sha256"],
            "training_state_game_count": int(final_validation["game_count"]),
            "training_state_environment_states": int(final_validation["environment_states"]),
            "training_state_optimizer_updates": int(final_validation["optimizer_updates"]),
            "training_state_replay_rows": int(final_validation["replay_rows"]),
            "node_path": str(node_path.relative_to(root)),
            "node_sha256": _sha_file(node_path),
        },
        "torch_version": torch.__version__,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = run_dir / "results" / "run_summary.json"
    _atomic_json(summary_path, summary)
    _write_progress(progress_path, run_id=run_id, status="complete", values={
        "workers": int(workers),
        "games_requested": int(games),
        "games_completed": int(games),
        "game_count_cumulative": int(game_count),
        "training_states_cumulative": int(state_count),
        "new_training_states": int(total_states_added),
        "optimizer_updates_cumulative": int(update_count),
        "optimizer_update_calls": int(games),
        "elapsed_wall_seconds": float(final_wall_seconds),
        "aggregate_cpu_seconds": float(final_cpu_seconds),
        "average_effective_cpu_cores": float(final_cpu_seconds / final_wall_seconds),
        "max_weight_lag_updates": int(summary["weight_sync"]["max_lag_updates"]),
        "max_process_tree_rss_bytes": int(peak_rss),
        "checkpoint_sha256": final_validation["sha256"],
    })
    return summary


def audit_run(
    run_dir: str | Path,
    *,
    baseline_checkpoint: str | Path | None = None,
    root: str | Path = ROOT,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Independently replay every recorded game and reconcile samples, counters and updates."""
    root = Path(root).resolve()
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "results" / "run_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "training_state.pt"
    metrics_path = run_dir / "results" / "per_game.jsonl"
    if not summary_path.is_file() or not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("run summary, full checkpoint, or game metrics are missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_games = int(summary["games_measured"])
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_games:
        raise ValueError("per-game metrics count differs from the registered run length")
    rows_by_id = {str(row["game_id"]): row for row in rows}
    if len(rows_by_id) != expected_games:
        raise ValueError("per-game metric IDs are not unique")
    raw_dir = run_dir / "raw" / "selfplay" / ARM
    raw_records: dict[str, dict[str, Any]] = {}
    replayed_states = 0
    for path in sorted(raw_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(record.get("game_id", ""))
        if game_id != path.stem or game_id in raw_records or game_id not in rows_by_id:
            raise ValueError(f"duplicate, orphaned or mismatched game record: {path.name}")
        result = v6_5_audit.audit_game_record(record, path.with_suffix(".npz"))
        if int(record["simulations_configured_per_move"]) != SIMULATIONS:
            raise ValueError("a recorded game does not retain PUCT-400")
        metric = rows_by_id[game_id]
        if int(record["seed"]) != int(metric["seed"]):
            raise ValueError("game record seed differs from its timing row")
        if int(metric["updates_after"]) != int(metric["updates_before"]) + 1:
            raise ValueError("training update count skipped or duplicated a game")
        if int(metric["training_update_calls"]) != 1:
            raise ValueError("a game did not receive exactly one V6.7 update flow")
        parallel = record.get("parallel_training", {})
        if int(parallel.get("weight_lag_updates", -1)) != int(metric["weight_lag_updates"]):
            raise ValueError("recorded weight lag differs between game and trainer metric")
        replayed_states += int(result["move_count"])
        raw_records[game_id] = record
    if len(raw_records) != expected_games or len(list(raw_dir.glob("*.npz"))) != expected_games:
        raise ValueError("raw self-play game and sample files do not match the run length")

    checkpoint_validation = v6_5_training.validate_training_state(checkpoint_path, root=root)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    baseline_checkpoint = Path(baseline_checkpoint) if baseline_checkpoint is not None else (
        root / "research/g_v6_9/source/G20000_training_state.pt"
    )
    base_state = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    updates_added = int(state["update_count"]) - int(base_state["update_count"])
    if updates_added != expected_games or replayed_states != int(state["environment_states"]) - BASE_STATES:
        raise ValueError("full checkpoint counts do not reconcile with game replays or updates")
    if int(checkpoint_validation["game_count"]) != BASE_GAMES + expected_games:
        raise ValueError("full checkpoint cumulative game count is incorrect")
    result = {
        "run_id": summary["run_id"],
        "games": expected_games,
        "unique_game_ids": len(raw_records),
        "legal_terminal_games": len(raw_records),
        "puct400_games": sum(int(record["simulations_configured_per_move"]) == SIMULATIONS for record in raw_records.values()),
        "selfplay_training_states": replayed_states,
        "optimizer_update_calls": len(rows),
        "optimizer_updates_added": updates_added,
        "weight_lag_max": max(int(row["weight_lag_updates"]) for row in rows),
        "checkpoint": checkpoint_validation,
        "valid": True,
    }
    _atomic_json(run_dir / "results" / "audit.json", result)
    summary["correctness"]["game_replay_audit_pending"] = False
    summary["correctness"]["legal_terminal_games"] = len(raw_records)
    summary["correctness"]["puct400_games"] = result["puct400_games"]
    summary["correctness"]["full_checkpoint_valid"] = bool(checkpoint_validation["valid"])
    _atomic_json(summary_path, summary)
    return result
