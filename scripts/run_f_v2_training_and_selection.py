#!/usr/bin/env python3
"""Run capped F V2 continuation rounds and select only on frozen development games."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.model import load_f_checkpoint
from src.research.imitation_f.training import train
from src.research.imitation_f.v2_evaluation import paired_opening_statistics, run_opening_matches


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _run_training(root: Path, config_path: Path, experiment: dict) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["initial_checkpoint"] = str(root / config["initial_checkpoint"])
    share = int(round(float(config["tactical_share"]) * 100))
    dataset = {
        "train": root / f"research/f_v2/data/train_tactical_{share:02d}.jsonl",
        "validation": root / "research/f_v2/data/validation.jsonl",
    }
    output = root / "research/f_v2/checkpoints" / config["run_name"]
    result = train(config, dataset, output, resume=True)
    summary = {
        "run_name": config["run_name"], "config": config,
        "dataset_sha256": hashlib.sha256(Path(dataset["train"]).read_bytes()).hexdigest(),
        "initial_checkpoint_sha256": result["initial_checkpoint_sha256"],
        "expected_initial_checkpoint_sha256": experiment["f1_checkpoint_sha256"],
        "updates": result["updates"], "best_epoch": result["best_epoch"],
        "elapsed_seconds": result["elapsed_seconds"], "device": result["device"],
        "best_checkpoint": str(result["best_checkpoint"]),
        "best_checkpoint_sha256": hashlib.sha256(Path(result["best_checkpoint"]).read_bytes()).hexdigest(),
        "state_checkpoint": str(result["state_checkpoint"]),
        "candidate_checkpoints": sorted(path.name for path in output.glob("epoch-*.pt")),
        "history": result["history"],
    }
    if summary["initial_checkpoint_sha256"] != experiment["f1_checkpoint_sha256"]:
        raise ValueError("continued run did not initialize from the frozen F1 checkpoint")
    _atomic_json(output / "training_summary.json", summary)
    return summary


def _run_candidate(root: Path, experiment: dict, candidate_id: str, checkpoint: Path) -> dict:
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_name = "F1" if candidate_id == "F1" else "F2"
    model = load_f_checkpoint(checkpoint, device="cpu")
    output = root / "research/f_v2/results/development" / candidate_id / "matches.jsonl"
    summary = run_opening_matches(
        model, model_name, candidate_id, "A", experiment["development_openings"], output,
        "development", digest, int(experiment["seed"]) + 501, 0.05,
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary["paired_statistics"] = paired_opening_statistics(
        records, model_name, int(experiment["bootstrap_seed"]) + 1000, bootstrap_samples=2000,
    )
    summary["validation_policy_loss"] = model.checkpoint_metadata.get("validation", {}).get("policy_loss")
    summary["model_name"] = model_name
    summary["checkpoint"] = str(checkpoint.relative_to(root))
    summary["checkpoint_sha256"] = digest
    _atomic_json(output.parent / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    experiment_path = root / "research/f_v2/config/experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment["f1_checkpoint_sha256"] != hashlib.sha256(
        (root / "research/f_v2/checkpoints/f1_reference.pt").read_bytes()
    ).hexdigest():
        raise ValueError("frozen F1 reference changed after schedule registration")
    torch.set_num_threads(3)
    pipeline_path = root / "research/f_v2/results/training_and_selection.json"
    if pipeline_path.exists():
        state = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if state.get("schedule_sha256") != experiment["schedule_sha256"]:
            raise ValueError("persisted training selection state differs from frozen schedule")
    else:
        state = {
            "schema_version": 1, "schedule_sha256": experiment["schedule_sha256"],
            "started_at_epoch_seconds": time.time(), "runs": {}, "candidates": {},
        }
        _atomic_json(pipeline_path, state)

    f1_summary = state["candidates"].get("F1")
    if not f1_summary:
        f1_summary = _run_candidate(
            root, experiment, "F1", root / "research/f_v2/checkpoints/f1_reference.pt"
        )
        state["candidates"]["F1"] = f1_summary
        _atomic_json(pipeline_path, state)

    selected_run = "tactical-30"
    for run_name, config_name in (("tactical-30", "training_tactical_30.json"),):
        run_summary = _run_training(root, root / "research/f_v2/config" / config_name, experiment)
        state["runs"][run_name] = run_summary
        _atomic_json(pipeline_path, state)
        candidate_ids = []
        for checkpoint in sorted((root / "research/f_v2/checkpoints" / run_name).glob("epoch-*.pt")):
            candidate_id = f"{run_name}-{checkpoint.stem.removeprefix('epoch-')}"
            candidate_summary = _run_candidate(root, experiment, candidate_id, checkpoint)
            state["candidates"][candidate_id] = candidate_summary
            candidate_ids.append(candidate_id)
            _atomic_json(pipeline_path, state)
        best_30 = max(
            (state["candidates"][candidate_id] for candidate_id in candidate_ids),
            key=lambda row: (float(row["score_rate"]), -float(row["validation_policy_loss"] or float("inf"))),
        )
        state["best_30_candidate_id"] = best_30["candidate_id"]
        state["run_40_trigger"] = float(best_30["score_rate"]) < 0.55
        _atomic_json(pipeline_path, state)

    if state.get("run_40_trigger"):
        run_summary = _run_training(
            root, root / "research/f_v2/config/training_tactical_40.json", experiment,
        )
        state["runs"]["tactical-40"] = run_summary
        _atomic_json(pipeline_path, state)
        for checkpoint in sorted((root / "research/f_v2/checkpoints/tactical-40").glob("epoch-*.pt")):
            candidate_id = f"tactical-40-{checkpoint.stem.removeprefix('epoch-')}"
            if candidate_id not in state["candidates"]:
                candidate_summary = _run_candidate(root, experiment, candidate_id, checkpoint)
                state["candidates"][candidate_id] = candidate_summary
                _atomic_json(pipeline_path, state)

    candidate_rows = [row for key, row in state["candidates"].items() if key != "F1"]
    if not candidate_rows:
        raise ValueError("training produced no saved candidate checkpoints")
    winner = max(
        candidate_rows,
        key=lambda row: (float(row["score_rate"]), -float(row["validation_policy_loss"] or float("inf"))),
    )
    source = root / winner["checkpoint"]
    selected = root / "research/f_v2/checkpoints/best_f2.pt"
    temp = selected.with_suffix(".pt.tmp")
    shutil.copyfile(source, temp)
    temp.replace(selected)
    state.update({
        "selected_candidate_id": winner["candidate_id"],
        "selected_checkpoint": str(selected.relative_to(root)),
        "selected_checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "selected_development_score_rate": float(winner["score_rate"]),
        "selected_validation_policy_loss": winner["validation_policy_loss"],
        "selection_rule": "maximum paired development score; ties resolved by lower validation policy cross-entropy",
        "completed": True,
    })
    _atomic_json(root / "research/f_v2/results/model_selection.json", state)
    _atomic_json(pipeline_path, state)
    print(json.dumps({key: state[key] for key in (
        "selected_candidate_id", "selected_checkpoint", "selected_checkpoint_sha256",
        "selected_development_score_rate", "selected_validation_policy_loss", "run_40_trigger",
    )}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
