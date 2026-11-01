"""Allowlisted, hash-checked G V3 recovery bundles and empty-tree restore."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile
from typing import Any

from .evaluation import F2_SHA256, load_evaluation_log
from .replay import load_game_log
from .v3_budget import active_wall_seconds_committed, cpu_seconds_committed
from .v3_runner import _config_digest
from .v3_schedule import validate_v3_schedule

MANIFEST = "manifest.json"
DATA_FILES = (
    "research/g_v3/config/experiment.json",
    "research/g_v3/config/openings.json",
    "research/g_v3/results/preflight/preflight.json",
    "research/g_v3/results/bn_probe/probe.json",
    "research/g_v3/results/bn_probe/selfplay.jsonl",
    "research/g_v3/results/bn_probe/evaluation.jsonl",
    "research/g_v3/results/recovery/drive_status.json",
    "research/f_v2/checkpoints/best_f2.pt",
)
DATA_DIRS = (
    "research/g_v3/checkpoints",
    "research/g_v3/results/selfplay",
    "research/g_v3/results/evaluation",
    "research/g_v3/results/selection",
    "research/g_v3/results/summary",
    "research/g_v3/figures",
    "research/g_v3/results/runtime",
)
CODE_FILES = (
    "src/gomoku.py",
    "src/ai.py",
    "src/research/imitation_f/model.py",
    "src/research/alphazero_g/__init__.py",
    "src/research/alphazero_g/puct.py",
    "src/research/alphazero_g/selfplay.py",
    "src/research/alphazero_g/replay.py",
    "src/research/alphazero_g/training.py",
    "src/research/alphazero_g/evaluation.py",
    "src/research/alphazero_g/v3_schedule.py",
    "src/research/alphazero_g/v3_preflight.py",
    "src/research/alphazero_g/v3_bn.py",
    "src/research/alphazero_g/v3_config.py",
    "src/research/alphazero_g/v3_budget.py",
    "src/research/alphazero_g/v3_runner.py",
    "src/research/alphazero_g/v3_evaluation.py",
    "src/research/alphazero_g/v3_snapshot.py",
    "src/research/alphazero_g/v3_report.py",
    "scripts/preflight_g_v3.py",
    "scripts/register_g_v3_budget.py",
    "scripts/amend_g_v3_estimate.py",
    "scripts/diagnose_g_v3_batchnorm.py",
    "scripts/create_g_v3_schedule.py",
    "scripts/run_g_v3.py",
    "scripts/evaluate_g_v3_development.py",
    "scripts/evaluate_g_v3_blind.py",
    "scripts/report_g_v3.py",
    "scripts/create_g_v3_snapshot.py",
    "scripts/restore_g_v3_snapshot.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("snapshot contains an unsupported or unsafe path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {".", MANIFEST}:
        raise ValueError("snapshot contains an unsupported or unsafe path")
    return path.as_posix()


def next_v3_snapshot_threshold(active_seconds: float) -> int:
    value = float(active_seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError("active runtime must be finite and nonnegative")
    thresholds = [3600, 10800, 25200, 46800, 82800, 140400]
    for threshold in thresholds:
        if value < threshold:
            return threshold
    return thresholds[-1] + (int((value - thresholds[-1]) // (26 * 3600)) + 1) * 26 * 3600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _identity() -> dict[str, Any]:
    root = _repo_root()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != "research/alphazero-g-v3":
        raise ValueError("G V3 recovery snapshots require the dedicated V3 branch")
    hashes = {}
    for relative in CODE_FILES:
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"recovery source is missing or unsafe: {relative}")
        hashes[relative] = _sha(source.read_bytes())
    return {"branch": branch, "commit": commit, "code_sha256": hashes}


def _collect(root: Path) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    for relative in DATA_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"G V3 recovery requires {relative}")
        contents[relative] = path.read_bytes()
    for relative_dir in DATA_DIRS:
        directory = root / relative_dir
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"unsafe recovery directory: {relative_dir}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"recovery data may not contain symlinks: {path}")
            if not path.is_file() or path.name.startswith(".") or path.name.endswith(".tmp"):
                continue
            if path.suffix == ".zip":
                continue
            contents[_safe_path(path.relative_to(root).as_posix())] = path.read_bytes()
    for name, data in list(contents.items()):
        if name.endswith(".jsonl"):
            contents[name] = data[:data.rfind(b"\n") + 1] if b"\n" in data else b""
    return contents


def _validate_contents(contents: dict[str, bytes]) -> dict[str, Any]:
    try:
        config = json.loads(contents[DATA_FILES[0]])
        schedule = json.loads(contents[DATA_FILES[1]])
        preflight = json.loads(contents[DATA_FILES[2]])
        probe = json.loads(contents[DATA_FILES[3]])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("G V3 snapshot has malformed JSON evidence") from exc
    if (config.get("registered") is not True
            or config.get("branch") != "research/alphazero-g-v3"
            or config.get("base_commit") != "c95eb5aa919c769af11ea6b7fc22df2594ae1c28"
            or config.get("config_sha256") != _config_digest(config)):
        raise ValueError("G V3 snapshot does not contain the registered experiment")
    validate_v3_schedule(schedule)
    if _sha(contents[DATA_FILES[1]]) != config.get("opening_schedule_sha256"):
        raise ValueError("snapshot opening schedule differs from registered config")
    if _sha(contents[DATA_FILES[2]]) != config.get("preflight_sha256"):
        raise ValueError("snapshot preflight evidence hash differs from config")
    if _sha(contents[DATA_FILES[3]]) != config.get("bn_probe_sha256"):
        raise ValueError("snapshot BN probe hash differs from config")
    if _sha(contents[DATA_FILES[7]]) != F2_SHA256:
        raise ValueError("snapshot frozen F2 hash mismatch")
    if not preflight.get("resources") or probe.get("formal_training") is not False:
        raise ValueError("snapshot preflight/BN evidence is incomplete")
    for name, data in contents.items():
        if name.endswith(".jsonl"):
            temporary = Path(tempfile.mkdtemp(prefix="g_v3_snapshot_log_"))
            try:
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if "/selfplay/" in name:
                    load_game_log(target)
                elif name.endswith("/tactics.jsonl"):
                    run_ids = set()
                    for line_number, line in enumerate(data.splitlines(), start=1):
                        row = json.loads(line)
                        if (not row.get("run_id") or row["run_id"] in run_ids
                                or row.get("puzzles") != len(row.get("records", []))
                                or not isinstance(row.get("stats"), dict)):
                            raise ValueError(f"invalid G V3 tactical log row in {name}:{line_number}")
                        run_ids.add(row["run_id"])
                elif "/evaluation/" in name:
                    load_evaluation_log(target)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
    return config


def create_v3_snapshot(root: str | Path, output: str | Path) -> dict[str, Any]:
    root_path, output_path = Path(root).resolve(), Path(output)
    data = _collect(root_path)
    config = _validate_contents(data)
    identity = _identity()
    source_root = _repo_root()
    for relative in CODE_FILES:
        data[f"source/{relative}"] = (source_root / relative).read_bytes()
    file_hashes = {name: _sha(value) for name, value in sorted(data.items())}
    selfplay_games = sum(len([line for line in value.splitlines() if line])
                         for name, value in data.items() if "/selfplay/" in name)
    evaluation_games = sum(len([line for line in value.splitlines() if line])
                           for name, value in data.items() if "/evaluation/" in name)
    active_wall = active_wall_seconds_committed(root_path)
    manifest = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "branch": identity["branch"], "commit": identity["commit"],
        "config_sha256": _sha(data[DATA_FILES[0]]),
        "schedule_sha256": config["opening_schedule_sha256"],
        "f2_sha256": F2_SHA256,
        "active_cpu_seconds": cpu_seconds_committed(root_path),
        "active_wall_seconds": active_wall,
        "next_snapshot_threshold_seconds": next_v3_snapshot_threshold(active_wall),
        "selfplay_game_records": selfplay_games,
        "evaluation_game_records": evaluation_games,
        "code_sha256": identity["code_sha256"],
        "files": file_hashes,
    }
    data[MANIFEST] = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted(data.items()):
                archive.writestr(name, value)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        parent = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    raw = output_path.read_bytes()
    return {
        "path": str(output_path), "sha256": _sha(raw), "size_bytes": len(raw),
        "branch": identity["branch"], "commit": identity["commit"],
        "config_sha256": config["config_sha256"],
        "active_cpu_seconds": manifest["active_cpu_seconds"],
        "active_wall_seconds": active_wall,
        "next_snapshot_threshold_seconds": manifest["next_snapshot_threshold_seconds"],
        "selfplay_game_records": selfplay_games,
        "evaluation_game_records": evaluation_games,
    }


def restore_v3_snapshot(archive_path: str | Path, destination: str | Path) -> dict[str, Any]:
    source, target = Path(archive_path), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    with zipfile.ZipFile(source, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or names.count(MANIFEST) != 1:
            raise ValueError("snapshot has duplicate paths or no unique manifest")
        for info in infos:
            if info.is_dir():
                raise ValueError("snapshot archives may contain files only")
            if info.filename != MANIFEST:
                _safe_path(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("snapshot archive may not contain symlinks")
        manifest = json.loads(archive.read(MANIFEST))
        members = {name: archive.read(name) for name in names if name != MANIFEST}
    hashes = manifest.get("files")
    if manifest.get("format_version") != 1 or not isinstance(hashes, dict) or set(hashes) != set(members):
        raise ValueError("snapshot manifest file list is incomplete")
    for name, value in members.items():
        _safe_path(name)
        if _sha(value) != hashes[name]:
            raise ValueError(f"snapshot SHA-256 mismatch for {name}")
    data = {name: value for name, value in members.items() if not name.startswith("source/")}
    allowed = set(DATA_FILES)
    allowed.update(name for name in data if any(name.startswith(f"{directory}/") for directory in DATA_DIRS))
    if set(data) != allowed or {name for name in members if name.startswith("source/")} != {
        f"source/{relative}" for relative in CODE_FILES
    }:
        raise ValueError("snapshot contains unsupported or missing recovery files")
    config = _validate_contents(data)
    if (manifest.get("experiment_id") != config["experiment_id"]
            or manifest.get("branch") != "research/alphazero-g-v3"
            or manifest.get("config_sha256") != _sha(data[DATA_FILES[0]])
            or manifest.get("schedule_sha256") != config["opening_schedule_sha256"]
            or manifest.get("f2_sha256") != F2_SHA256
            or set(manifest.get("code_sha256", {})) != set(CODE_FILES)):
        raise ValueError("snapshot manifest does not match its registered experiment")
    for relative in CODE_FILES:
        if _sha(members[f"source/{relative}"]) != manifest["code_sha256"][relative]:
            raise ValueError(f"snapshot recovery source checksum mismatch for {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir() or any(target.iterdir()):
        raise ValueError("snapshot restore requires an empty destination directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore.", dir=target.parent))
    try:
        for name, value in members.items():
            output = stage / _safe_path(name)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        target.rmdir()
        os.replace(stage, target)
        parent = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "empty_restore": True, "archive_sha256": _sha(source.read_bytes()),
        "branch": manifest["branch"], "commit": manifest["commit"],
        "active_cpu_seconds": manifest["active_cpu_seconds"],
        "active_wall_seconds": manifest["active_wall_seconds"],
        "next_snapshot_threshold_seconds": manifest["next_snapshot_threshold_seconds"],
        "selfplay_game_records": manifest["selfplay_game_records"],
        "evaluation_game_records": manifest["evaluation_game_records"],
        "restored_files": sorted(members),
    }
