#!/usr/bin/env python3
"""Create a compact, self-contained F V2 recovery archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.imitation_f.v2_snapshot import MANIFEST_NAME, sha256_file


def _copy(source: Path, staging: Path, relative: str) -> None:
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
    experiment = json.loads((root / "research/f_v2/config/experiment.json").read_text(encoding="utf-8"))
    selected = root / "research/f_v2/checkpoints/best_f2.pt"
    tracked_script_names = (
        "collect_f_v2_data.py", "prepare_f_v2_dataset.py", "train_f_v2.py",
        "run_f_v2_development.py", "run_f_v2_training_and_selection.py",
        "run_f_v2_final_blind.py", "report_f_v2.py", "preflight_f_v2.py",
        "create_f_v2_snapshot.py", "restore_f_v2_snapshot.py",
    )
    with tempfile.TemporaryDirectory(prefix="wuziq-f-v2-snapshot-") as scratch:
        staging = Path(scratch) / "snapshot"
        staging.mkdir()
        shutil.copytree(root / "src", staging / "src")
        for name in tracked_script_names:
            path = root / "scripts" / name
            if path.exists():
                _copy(path, staging, f"scripts/{name}")
        for name in ("pyproject.toml", "requirements.txt", "requirements-f.txt", "requirements-research.txt", "pytest.ini"):
            path = root / name
            if path.exists():
                _copy(path, staging, name)
        for name in (
            "test_f_v1_training.py", "test_f_v2_data.py", "test_f_v2_evaluation.py",
            "test_f_v2_experiment.py", "test_f_v2_snapshot.py",
        ):
            path = root / "tests" / name
            if path.exists():
                _copy(path, staging, f"tests/{name}")
        for name in ("train.jsonl", "validation.jsonl", "blind_test.jsonl"):
            _copy(root / "research/f_v1/data" / name, staging, f"research/f_v1/data/{name}")
        _copy(root / "research/f_v1/checkpoints/best.pt", staging, "research/f_v1/checkpoints/best.pt")
        _copy(
            root / "research/f_v1/checkpoints/training_history.jsonl",
            staging,
            "research/f_v1/checkpoints/training_history.jsonl",
        )
        shutil.copytree(root / "research/f_v2", staging / "research/f_v2")
        tracked = sorted(path for path in staging.rglob("*") if path.is_file())
        files = {
            str(path.relative_to(staging).as_posix()): {
                "sha256": sha256_file(path), "bytes": path.stat().st_size,
            }
            for path in tracked
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": commit, "branch": branch,
            "experiment_schedule_sha256": experiment["schedule_sha256"],
            "f1_checkpoint_sha256": experiment["f1_checkpoint_sha256"],
            "model_checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            "active_blind_opening_groups": [row["opening_group_id"] for row in experiment["blind_openings"]],
            "dagger_progress": json.loads((root / "research/f_v2/results/dagger/dagger_progress.json").read_text(encoding="utf-8")),
            "files": files,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output.with_suffix(output.suffix + ".tmp")
        with zipfile.ZipFile(temp_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(staging / MANIFEST_NAME, MANIFEST_NAME)
            for path in tracked:
                archive.write(path, path.relative_to(staging).as_posix())
        temp_output.replace(output)
    print(json.dumps({
        "archive": str(output), "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": len(files), "code_commit": commit, "branch": branch,
        "experiment_schedule_sha256": experiment["schedule_sha256"],
        "model_checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
