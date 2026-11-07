from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from src.research.alphazero_g.v6_2_registration import validate_schedule
from src.research.alphazero_g.v6_2_training import ROOT, validate_full_training_state


ARTIFACT = Path("research/g_v6_2")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files(root: Path) -> list[Path]:
    required = [
        root / ARTIFACT / "config",
        root / ARTIFACT / "checkpoints",
        root / ARTIFACT / "raw/selfplay",
        root / ARTIFACT / "results",
        root / "research/g_v5/config/experiment.json",
        root / "research/g_v5/config/opening_schedule.json",
        root / "research/g_v6/config/opening_schedule.json",
        root / "src/research/alphazero_g/v5_training.py",
        root / "src/research/alphazero_g/v5_reference.py",
        root / "src/research/alphazero_g/v5_evaluation.py",
        root / "src/research/alphazero_g/v5_audit.py",
        root / "src/research/alphazero_g/v6_evaluation.py",
        root / "src/research/alphazero_g/v6_registration.py",
        root / "src/research/alphazero_g/v6_audit.py",
        root / "src/research/alphazero_g/v6_2_training.py",
        root / "src/research/alphazero_g/v6_2_registration.py",
        root / "src/research/alphazero_g/v6_2_evaluation.py",
        root / "scripts/run_g_v6_2.py",
        root / "scripts/prepare_g_v6_2.py",
        root / "scripts/audit_g_v6_2.py",
        root / "scripts/report_g_v6_2.py",
        root / "scripts/package_g_v6_2_recovery.py",
        root / "tests/test_alphazero_g_v6_2_training.py",
        root / "requirements-research.txt",
    ]
    files = []
    for item in required:
        if item.is_file():
            files.append(item)
        elif item.is_dir():
            files.extend(
                path for path in item.rglob("*")
                if path.is_file() and "charts" not in path.parts
                and path.name not in {"snapshot_status.json", "integrity_audit.json", "REPORT.md"}
            )
        else:
            raise FileNotFoundError(item)
    source = root / "src"
    files.extend(path for path in source.rglob("*.py") if path.is_file())
    third_party = root / "third_party/AlphaZero_Gomoku"
    files.extend(path for path in third_party.rglob("*.py") if path.is_file())
    return sorted(set(files))


def create_archive(root: str | Path = ROOT, output: str | Path | None = None) -> dict:
    root = Path(root).resolve()
    if output is None:
        output = Path(tempfile.gettempdir()) / "wuziq-g-v6-2-recovery.zip"
    output = Path(output).resolve()
    files = _tracked_files(root)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "experiment_id": "wuziq-alphazero-g-v6-2-9x9-selfplay-scale",
        "branch": "research/alphazero-g-v6-2-selfplay-scale",
        "code_commit": commit,
        "files": {str(path.relative_to(root)): _sha_file(path) for path in files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("V6_2_RECOVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return {"path": str(output), "sha256": _sha_file(output), "size_bytes": output.stat().st_size,
            "file_count": len(files), "code_commit": commit}


def verify_empty_restore(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("recovery test destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
        manifest = json.loads(archive.read("V6_2_RECOVERY_MANIFEST.json"))
    for relative, expected in manifest["files"].items():
        path = destination / relative
        if not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"recovery archive content check failed: {relative}")
    schedule = json.loads((destination / ARTIFACT / "config/opening_schedule.json").read_text(encoding="utf-8"))
    validate_schedule(schedule, root=destination)
    checkpoint = destination / ARTIFACT / "checkpoints/reference400_empty/training_state.pt"
    state_validation = validate_full_training_state(destination, checkpoint)
    return {
        "empty_directory_restore": True,
        "archive_sha256": _sha_file(archive_path),
        "manifest_files_verified": len(manifest["files"]),
        "training_state_restore": state_validation,
        "openings_verified": schedule["validation"],
        "code_commit": manifest["code_commit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and restore-check a minimal G V6.2 recovery archive")
    parser.add_argument("--output", default="/tmp/wuziq-g-v6-2-recovery.zip")
    parser.add_argument("--restore-to", default="/tmp/wuziq-g-v6-2-empty-restore")
    args = parser.parse_args()
    result = create_archive(ROOT, args.output)
    result["restore_validation"] = verify_empty_restore(args.output, args.restore_to)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
