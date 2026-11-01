"""Allowlisted, hash-verified G V2 recovery archives and empty-directory restore."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile
from typing import Any

from .evaluation import F2_SHA256, load_evaluation_log
from .replay import load_game_log
from .v2_budget import active_wall_seconds_committed, cpu_seconds_committed
from .v2_schedule import validate_v2_schedule

MANIFEST = "manifest.json"
DATA_FILES = (
    "research/g_v2/config/experiment.json",
    "research/g_v2/config/openings.json",
    "research/f_v2/checkpoints/best_f2.pt",
)
DATA_DIRS = (
    "research/g_v2/checkpoints",
    "research/g_v2/results/selfplay",
    "research/g_v2/results/evaluation",
    "research/g_v2/results/selection",
)
CODE_FILES = (
    "src/gomoku.py",
    "src/ai.py",
    "src/research/imitation_f/model.py",
    "src/research/alphazero_g/puct.py",
    "src/research/alphazero_g/selfplay.py",
    "src/research/alphazero_g/replay.py",
    "src/research/alphazero_g/training.py",
    "src/research/alphazero_g/evaluation.py",
    "src/research/alphazero_g/v2_schedule.py",
    "src/research/alphazero_g/v2_preflight.py",
    "src/research/alphazero_g/v2_runner.py",
    "src/research/alphazero_g/v2_budget.py",
    "src/research/alphazero_g/v2_selection.py",
    "src/research/alphazero_g/v2_blind.py",
    "src/research/alphazero_g/v2_snapshot.py",
    "src/research/alphazero_g/v2_report.py",
    "scripts/preflight_g_v2.py",
    "scripts/register_g_v2_budget.py",
    "scripts/run_g_v2.py",
    "scripts/evaluate_g_v2_development.py",
    "scripts/run_g_v2_growth.py",
    "scripts/create_g_v2_snapshot.py",
    "scripts/restore_g_v2_snapshot.py",
    "scripts/report_g_v2.py",
)


def next_v2_snapshot_threshold(active_seconds: float) -> int:
    """Return cumulative 1, 3, 7, 13, 23, 39, 65 hour backup checkpoints."""
    value = float(active_seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError("active wall time must be finite and nonnegative")
    thresholds = [3600, 10800, 25200, 46800, 82800, 140400]
    if value < thresholds[0]:
        return thresholds[0]
    for threshold in thresholds:
        if value < threshold:
            return threshold
    return thresholds[-1] + (int((value - thresholds[-1]) // (26 * 3600)) + 1) * 26 * 3600


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _identity() -> dict[str, str]:
    root = _repo_root()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != "research/alphazero-g-v2":
        raise ValueError("G V2 snapshots must use the dedicated research/alphazero-g-v2 branch")
    code_hashes = {}
    for relative in CODE_FILES:
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"G V2 recovery source is missing or unsafe: {relative}")
        code_hashes[relative] = _sha(source.read_bytes())
    return {"branch": branch, "commit": commit, "code_sha256": code_hashes}


def _safe_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("snapshot contains an unsupported or unsafe path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or name == MANIFEST:
        raise ValueError("snapshot contains an unsupported or unsafe path")
    return path.as_posix()


def _collect_data(root: Path) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    for relative in DATA_FILES:
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"G V2 recovery snapshot requires {relative}")
        contents[relative] = source.read_bytes()
    for relative_dir in DATA_DIRS:
        source_dir = root / relative_dir
        if not source_dir.exists():
            continue
        if source_dir.is_symlink() or not source_dir.is_dir():
            raise ValueError(f"G V2 recovery path is not a safe directory: {relative_dir}")
        for source in sorted(source_dir.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"G V2 recovery data may not contain a symlink: {source}")
            if not source.is_file():
                continue
            if source.name.startswith(".") or source.name.endswith(".tmp"):
                continue
            relative = source.relative_to(root).as_posix()
            contents[_safe_path(relative)] = source.read_bytes()
    return contents


def _validate_data(contents: dict[str, bytes], *, require_f2: bool = True) -> dict[str, Any]:
    try:
        config = json.loads(contents[DATA_FILES[0]])
        schedule = json.loads(contents[DATA_FILES[1]])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("G V2 snapshot config or schedule is invalid JSON") from exc
    if config.get("registered") is not True or config.get("base_commit") != "6e2bf48aa6b630c0fa4f20284ceb3b5aab295080":
        raise ValueError("G V2 snapshot does not contain the registered V1 baseline configuration")
    validate_v2_schedule(schedule)
    if schedule.get("schedule_sha256") != config.get("opening_schedule_sha256"):
        raise ValueError("G V2 snapshot schedule differs from its registered configuration")
    if require_f2 and _sha(contents[DATA_FILES[2]]) != F2_SHA256:
        raise ValueError("G V2 snapshot F2 checkpoint checksum differs from the frozen baseline")
    for name, data in contents.items():
        if name.endswith(".jsonl"):
            # Parse each durable log through the same validators used by resume and analysis.
            path = PurePosixPath(name)
            temporary_root = Path(tempfile.mkdtemp(prefix="g_v2_snapshot_log_"))
            try:
                target = temporary_root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if "/selfplay/" in name:
                    load_game_log(target)
                elif "/evaluation/" in name:
                    load_evaluation_log(target)
            finally:
                shutil.rmtree(temporary_root, ignore_errors=True)
    return config


def create_v2_snapshot(root: str | Path, destination: str | Path) -> dict[str, Any]:
    """Build an atomic recovery archive with only V2 state and its frozen F2 baseline."""
    root_path, output = Path(root).resolve(), Path(destination)
    contents = _collect_data(root_path)
    # A concurrently interrupted append may leave only a partial final line; it
    # cannot represent a complete game and is intentionally absent from recovery.
    for name, data in list(contents.items()):
        if name.endswith(".jsonl"):
            end = data.rfind(b"\n") + 1
            contents[name] = data[:end]
    config = _validate_data(contents)
    identity = _identity()
    repo_root = _repo_root()
    for relative in CODE_FILES:
        contents[f"source/{relative}"] = (repo_root / relative).read_bytes()
    selfplay_games = 0
    evaluation_games = 0
    for name, data in contents.items():
        if name.endswith(".jsonl"):
            lines = [line for line in data.splitlines() if line]
            if "/selfplay/" in name:
                selfplay_games += len(lines)
            elif "/evaluation/" in name:
                evaluation_games += len(lines)
    files = {name: _sha(data) for name, data in sorted(contents.items())}
    meta: dict[str, Any] = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "branch": identity["branch"],
        "commit": identity["commit"],
        "code_sha256": identity["code_sha256"],
        "config_sha256": _sha(contents[DATA_FILES[0]]),
        "schedule_sha256": config["opening_schedule_sha256"],
        "f2_sha256": F2_SHA256,
        "active_cpu_seconds": cpu_seconds_committed(root_path),
        "active_wall_seconds": active_wall_seconds_committed(root_path),
        "next_snapshot_threshold_seconds": next_v2_snapshot_threshold(active_wall_seconds_committed(root_path)),
        "selfplay_game_records": selfplay_games,
        "evaluation_game_records": evaluation_games,
        "files": files,
    }
    contents[MANIFEST] = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(contents.items()):
                archive.writestr(name, data)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    raw = output.read_bytes()
    return {
        "path": str(output), "sha256": _sha(raw), "size_bytes": len(raw),
        "experiment_id": config["experiment_id"], "branch": identity["branch"],
        "commit": identity["commit"], "active_cpu_seconds": meta["active_cpu_seconds"],
        "active_wall_seconds": meta["active_wall_seconds"],
        "next_snapshot_threshold_seconds": meta["next_snapshot_threshold_seconds"],
        "selfplay_game_records": selfplay_games, "evaluation_game_records": evaluation_games,
    }


def restore_v2_snapshot(snapshot_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Validate every member, then atomically restore the archive into an empty directory."""
    source, target = Path(snapshot_path), Path(destination)
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
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError("snapshot archive may not contain symlinks")
        try:
            manifest = json.loads(archive.read(MANIFEST))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("snapshot manifest is malformed") from exc
        members = {name: archive.read(name) for name in names if name != MANIFEST}
    hashes = manifest.get("files")
    if manifest.get("format_version") != 1 or not isinstance(hashes, dict) or set(hashes) != set(members):
        raise ValueError("snapshot manifest file list is incomplete")
    for name, data in members.items():
        _safe_path(name)
        if _sha(data) != hashes[name]:
            raise ValueError(f"snapshot SHA-256 mismatch for {name}")
    data_members = {name: data for name, data in members.items() if not name.startswith("source/")}
    allowed_data = set(DATA_FILES)
    allowed_data.update(
        name for name in data_members
        if any(name.startswith(f"{directory}/") for directory in DATA_DIRS)
    )
    expected_sources = {f"source/{relative}" for relative in CODE_FILES}
    if set(data_members) != allowed_data or {name for name in members if name.startswith("source/")} != expected_sources:
        raise ValueError("snapshot contains unsupported or missing recovery files")
    config = _validate_data(data_members)
    if (manifest.get("experiment_id") != config.get("experiment_id")
            or manifest.get("config_sha256") != _sha(data_members[DATA_FILES[0]])
            or manifest.get("schedule_sha256") != config.get("opening_schedule_sha256")
            or manifest.get("f2_sha256") != F2_SHA256
            or manifest.get("branch") != "research/alphazero-g-v2"):
        raise ValueError("snapshot manifest does not match the registered experiment identity")
    repo_root = _repo_root()
    if set(manifest.get("code_sha256", {})) != set(CODE_FILES):
        raise ValueError("snapshot source-code manifest is incomplete")
    for relative in CODE_FILES:
        source_name = f"source/{relative}"
        if (source_name not in members or _sha(members[source_name]) != manifest["code_sha256"][relative]):
            raise ValueError(f"snapshot recovery source checksum mismatch for {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir() or any(target.iterdir()):
        raise ValueError("snapshot restore requires an empty destination directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore.", dir=target.parent))
    try:
        for name, data in members.items():
            relative = _safe_path(name)
            destination_file = stage / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            with destination_file.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        target.rmdir()
        os.replace(stage, target)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "empty_restore": True, "archive_sha256": _sha(source.read_bytes()),
        "experiment_id": config["experiment_id"], "branch": manifest["branch"],
        "commit": manifest["commit"], "active_cpu_seconds": manifest["active_cpu_seconds"],
        "active_wall_seconds": manifest["active_wall_seconds"],
        "next_snapshot_threshold_seconds": manifest["next_snapshot_threshold_seconds"],
        "restored_files": sorted(members),
        "selfplay_game_records": manifest["selfplay_game_records"],
        "evaluation_game_records": manifest["evaluation_game_records"],
    }
