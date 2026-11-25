"""Bundle and restore independent V6.9 throughput runs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import torch

from . import v6_5_training


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(run_dir: str | Path, output: str | Path) -> dict:
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    summary_path = run_dir / "results/run_summary.json"
    audit_path = run_dir / "results/audit.json"
    checkpoint_path = run_dir / "checkpoints/training_state.pt"
    node_files = list((run_dir / "checkpoints").glob("node_G*.pt"))
    if not all(path.is_file() for path in (summary_path, audit_path, checkpoint_path)) or len(node_files) != 1:
        raise FileNotFoundError("a complete, audited V6.9 run with one model node is required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("valid") or int(audit.get("games", -1)) != int(summary["games_measured"]):
        raise ValueError("V6.9 bundle requires a complete independent game audit")
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and not path.name.endswith(".tmp"))
    relative_hashes = {path.relative_to(run_dir.parent).as_posix(): sha_file(path) for path in files}
    manifest = {
        "format_version": 1,
        "experiment_id": "wuziq-alphazero-g-v6-9-parallel-efficiency",
        "run_id": str(summary["run_id"]),
        "workers": int(summary["workers"]),
        "games_measured": int(summary["games_measured"]),
        "base_commit": summary["base"]["commit"],
        "baseline_archive_sha256": summary["base"]["archive_sha256"],
        "baseline_training_state_sha256": summary["base"]["training_state_sha256"],
        "training_state_sha256": sha_file(checkpoint_path),
        "training_state_path": checkpoint_path.relative_to(run_dir.parent).as_posix(),
        "node_path": node_files[0].relative_to(run_dir.parent).as_posix(),
        "audit_valid": True,
        "file_hashes": relative_hashes,
        "file_count": len(relative_hashes),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(run_dir.parent).as_posix())
        archive.writestr("V6_9_BUNDLE_MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {
        "path": str(output),
        "sha256": sha_file(output),
        "size_bytes": output.stat().st_size,
        "training_state_sha256": manifest["training_state_sha256"],
        "file_count": len(relative_hashes),
        "manifest": manifest,
    }


def restore_bundle(archive_path: str | Path, destination: str | Path, *, root: str | Path) -> dict:
    archive_path, destination, root = Path(archive_path), Path(destination), Path(root).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("V6.9 restore destination must be empty")
    archive_sha = sha_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"V6.9 ZIP CRC failed: {bad}")
        manifest = json.loads(archive.read("V6_9_BUNDLE_MANIFEST.json"))
        if manifest.get("experiment_id") != "wuziq-alphazero-g-v6-9-parallel-efficiency":
            raise ValueError("archive is not a V6.9 throughput bundle")
        for name in archive.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("unsafe V6.9 archive member path")
        archive.extractall(destination)
    for relative, expected in manifest["file_hashes"].items():
        path = destination / relative
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f"V6.9 bundle hash mismatch: {relative}")
    training_state_path = destination / manifest["training_state_path"]
    node_path = destination / manifest["node_path"]
    validation = v6_5_training.validate_training_state(training_state_path, root=root)
    if validation["sha256"] != manifest["training_state_sha256"]:
        raise ValueError("restored V6.9 full-state hash differs from its bundle manifest")
    state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    node = torch.load(node_path, map_location="cpu", weights_only=False)
    if not v6_5_training._state_equal(state["model_state"], node["model_state"]):
        raise ValueError("restored V6.9 model node differs from its full-state checkpoint")
    if int(state["game_count"]) != int(manifest["games_measured"]) + 20_000:
        raise ValueError("restored V6.9 game count does not match its manifest")
    return {
        "archive_sha256": archive_sha,
        "empty_directory_restore": True,
        "run_id": manifest["run_id"],
        "games_measured": manifest["games_measured"],
        "training_state_sha256": validation["sha256"],
        "file_count": manifest["file_count"],
        "checkpoint": validation,
        "model_node_matches_full_state": True,
        "valid": True,
    }
