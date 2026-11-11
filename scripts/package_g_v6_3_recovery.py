from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
import zipfile

import torch

from src.research.alphazero_g.v5_reference import ReferencePolicyValueNet
from src.research.alphazero_g.v5_training import load_training_checkpoint


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path("research/g_v6_3")
ARM = "reference400_empty"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(root: Path) -> list[Path]:
    artifact = root / ARTIFACT
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    return sorted(path for path in artifact.rglob("*") if path.is_file()
                  and not path.name.endswith(".tmp")
                  and "recovery_archives" not in path.parts
                  and "charts" not in path.relative_to(artifact).parts
                  and path.name not in {"REPORT.md", "training_trajectory.csv", "match_summary.csv",
                                        "integrity_audit.json"})


def _source_files(root: Path) -> list[Path]:
    wanted = [
        root / "research/g_v5/config/experiment.json",
        root / "src/research/alphazero_g/v5_training.py",
        root / "src/research/alphazero_g/v5_reference.py",
        root / "src/research/alphazero_g/v5_evaluation.py",
        root / "src/research/alphazero_g/v6_3_training.py",
        root / "src/research/alphazero_g/v6_3_evaluation.py",
        root / "src/ai.py", root / "src/gomoku.py",
        root / "scripts/prepare_g_v6_3.py", root / "scripts/run_g_v6_3.py",
        root / "scripts/package_g_v6_3_recovery.py",
        root / "scripts/audit_g_v6_3.py",
        root / "scripts/report_g_v6_3.py",
        root / "scripts/register_g_v6_3_snapshot.py",
        root / "requirements-research.txt",
    ]
    third_party = root / "third_party/AlphaZero_Gomoku"
    wanted.extend(third_party.rglob("*.py"))
    missing = [path for path in wanted if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"recovery source missing: {missing[0]}")
    return sorted(set(wanted))


def create_archive(root: str | Path = ROOT, output: str | Path | None = None) -> dict:
    started_cpu = time.process_time()
    root = Path(root).resolve()
    if output is None:
        output = Path(tempfile.gettempdir()) / "wuziq-g-v6-3-recovery.zip"
    output = Path(output).resolve()
    files = _artifact_files(root) + _source_files(root)
    files = sorted(set(files))
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                 capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-3-5k-continuation",
        "branch": branch,
        "code_commit": code_commit,
        "files": {path.relative_to(root).as_posix(): _sha_file(path) for path in files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("V6_3_RECOVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return {"path": str(output), "sha256": _sha_file(output), "size_bytes": output.stat().st_size,
            "file_count": len(files), "code_commit": code_commit, "branch": branch,
            "games_completed": _current_game_count(root),
            "packaging_cpu_seconds": time.process_time() - started_cpu}


def _current_game_count(root: Path) -> int:
    path = root / ARTIFACT / "checkpoints" / ARM / "training_state.pt"
    if not path.is_file():
        return -1
    return int(torch.load(path, map_location="cpu", weights_only=False)["game_count"])


def verify_empty_restore(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("recovery restore destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("V6_3_RECOVERY_MANIFEST.json"))
        for member in archive.namelist():
            target = PurePosixPath(member)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError("unsafe path in recovery archive")
        archive.extractall(destination)
    for relative, expected in manifest["files"].items():
        path = destination / relative
        if not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"recovery content hash mismatch: {relative}")
    artifact = destination / ARTIFACT
    config = json.loads((artifact / "config/reference_training_config.json").read_text(encoding="utf-8"))
    state_path = artifact / "checkpoints" / ARM / "training_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    model = ReferencePolicyValueNet(9, seed=int(payload["base_seed"]))
    old_python = __import__("random").getstate()
    import numpy as np
    old_numpy, old_torch = np.random.get_state(), torch.get_rng_state().clone()
    try:
        loaded = load_training_checkpoint(
            state_path, model=model, arm=ARM, config_hash=config["config_sha256"],
            buffer_size=int(config["training"]["buffer_size"]),
        )
        checkpoint_restore = {
            "model_restored": all(torch.equal(model.get_policy_param()[key], payload["model_state"][key])
                                   for key in payload["model_state"]),
            "adam_restored": bool(model.optimizer.state_dict()["state"]),
            "replay_restored": len(loaded[4]) == len(payload["replay_buffer"]),
            "python_rng_restored": __import__("random").getstate() == payload["python_rng_state"],
            "numpy_rng_restored": np.random.get_state()[0] == payload["numpy_rng_state"][0]
                and np.array_equal(np.random.get_state()[1], payload["numpy_rng_state"][1]),
            "torch_rng_restored": torch.equal(torch.get_rng_state(), payload["torch_rng_state"]),
        }
    finally:
        __import__("random").setstate(old_python)
        np.random.set_state(old_numpy)
        torch.set_rng_state(old_torch)
    if not all(checkpoint_restore.values()):
        raise ValueError(f"empty-directory training-state restore failed: {checkpoint_restore}")
    return {
        "empty_directory_restore": True,
        "archive_sha256": _sha_file(archive_path),
        "manifest_files_verified": len(manifest["files"]),
        "code_commit": manifest["code_commit"],
        "branch": manifest["branch"],
        "training_state": {
            "sha256": _sha_file(state_path), "game_count": int(payload["game_count"]),
            "environment_states": int(payload["environment_states"]),
            "optimizer_updates": int(payload["update_count"]),
            "replay_rows": len(loaded[4]), **checkpoint_restore,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or restore-verify a V6.3 recovery snapshot")
    parser.add_argument("--output", default="/tmp/wuziq-g-v6-3-recovery.zip")
    parser.add_argument("--restore-to", default="/tmp/wuziq-g-v6-3-empty-restore")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_empty_restore(args.output, args.restore_to)
    else:
        result = create_archive(ROOT, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
