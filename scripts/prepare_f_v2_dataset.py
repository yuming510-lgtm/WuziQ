#!/usr/bin/env python3
"""Build immutable 30%/40% train-only tactical mixes from F1 and DAgger data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.dataset import validate_game_record
from src.research.imitation_f.v2_data import build_weighted_training_rows


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--dagger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    schedule = json.loads((args.schedule or root / "research/f_v2/config/experiment.json").read_text(encoding="utf-8"))
    v1 = root / "research/f_v1/data"
    dagger_path = args.dagger or root / "research/f_v2/results/dagger/dagger_annotations.jsonl"
    output = args.output or root / "research/f_v2/data"
    train_rows = _jsonl(v1 / "train.jsonl")
    validation = _jsonl(v1 / "validation.jsonl")
    blind = _jsonl(v1 / "blind_test.jsonl")
    dagger_rows = _jsonl(dagger_path)
    validation_keys = {str(row["state_key"]) for row in validation}
    blind_keys = {str(row["state_key"]) for row in blind}
    if validation_keys.intersection(blind_keys):
        raise ValueError("V1 validation and blind sets share canonical states")

    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for share in schedule["tactical_shares"]:
        rows, manifest = build_weighted_training_rows(
            train_rows, dagger_rows, validation_keys, blind_keys,
            float(share), int(schedule["seed"]) + int(round(float(share) * 1000)),
        )
        path = output / f"train_tactical_{int(round(float(share) * 100)):02d}.jsonl"
        _write_jsonl(path, rows)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.update({"file": path.name, "sha256": digest, "unique_rows": len(rows)})
        manifests[str(float(share))] = manifest
    # Copy the original held-out partitions byte-for-byte. They are evaluation
    # inputs only; neither exact labels nor balancing are applied to them.
    for name in ("validation.jsonl", "blind_test.jsonl"):
        shutil.copyfile(v1 / name, output / name)

    dagger_games = _jsonl(root / "research/f_v2/results/dagger/dagger_games.jsonl")
    for game in dagger_games:
        validate_game_record(game)
        if game["group_id"] not in {row["opening_group_id"] for row in schedule["training_openings"]}:
            raise ValueError("DAgger game is outside the frozen training opening groups")
    provenance = {
        "schema_version": 1,
        "experiment_id": "wuziq-imitation-learning-f-v2",
        "schedule_sha256": schedule["schedule_sha256"],
        "f1_checkpoint_sha256": schedule["f1_checkpoint_sha256"],
        "v1_train_sha256": hashlib.sha256((v1 / "train.jsonl").read_bytes()).hexdigest(),
        "dagger_annotations_sha256": hashlib.sha256(dagger_path.read_bytes()).hexdigest(),
        "train_variants": manifests,
        "validation_positions": len(validation),
        "blind_test_positions": len(blind),
        "validation_canonical_states": len(validation_keys),
        "blind_canonical_states": len(blind_keys),
        "validation_blind_overlap": len(validation_keys.intersection(blind_keys)),
        "training_heldout_overlap": 0,
        "dagger_games": len(dagger_games),
        "dagger_complete_games": len(dagger_games),
    }
    (output / "manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
