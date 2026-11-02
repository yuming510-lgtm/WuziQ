"""Hash-verified, allowlisted G V4 recovery snapshots and empty-tree restore."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile
from typing import Any

from .replay import load_game_log
from .v3_evaluation import load_evaluation_log
from .v3_schedule import validate_v3_schedule
from .v4_budget import active_wall_seconds_committed, cpu_seconds_committed, next_snapshot_threshold
from .v4_config import BRANCH, config_digest

MANIFEST = "manifest.json"
DATA_FILES = (
    "research/g_v4/config/experiment.json",
    "research/g_v4/config/openings.json",
    "research/g_v4/results/preflight/preflight.json",
    "research/g_v4/results/preflight/v3_continuation_audit.json",
    "research/g_v4/results/recovery/drive_status.json",
    "research/g_v3/checkpoints/G0/seed-510901/node_20000.pt",
    "research/g_v3/results/selfplay/G0-seed-510901.jsonl",
    "research/f_v2/checkpoints/best_f2.pt",
)
DATA_DIRS = (
    "research/g_v4/checkpoints",
    "research/g_v4/results/selfplay",
    "research/g_v4/results/evaluation",
    "research/g_v4/results/selection",
    "research/g_v4/results/training",
    "research/g_v4/results/runtime",
    "research/g_v4/results/summary",
    "research/g_v4/figures",
)
OPTIONAL_FILES = ("research/g_v4/REPORT.md",)
CODE_FILES = (
    "src/gomoku.py", "src/ai.py", "src/research/imitation_f/model.py",
    "src/research/alphazero_g/__init__.py", "src/research/alphazero_g/puct.py",
    "src/research/alphazero_g/selfplay.py", "src/research/alphazero_g/replay.py",
    "src/research/alphazero_g/training.py", "src/research/alphazero_g/evaluation.py",
    "src/research/alphazero_g/v3_bn.py", "src/research/alphazero_g/v3_budget.py",
    "src/research/alphazero_g/v3_runner.py", "src/research/alphazero_g/v3_evaluation.py",
    "src/research/alphazero_g/v3_schedule.py", "src/research/alphazero_g/v4_budget.py",
    "src/research/alphazero_g/v4_config.py", "src/research/alphazero_g/v4_preflight.py",
    "src/research/alphazero_g/v4_runner.py", "src/research/alphazero_g/v4_report.py",
    "src/research/alphazero_g/v4_snapshot.py", "scripts/create_g_v4_schedule.py",
    "scripts/preflight_g_v4.py", "scripts/register_g_v4_budget.py", "scripts/run_g_v4.py",
    "scripts/create_g_v4_snapshot.py", "scripts/restore_g_v4_snapshot.py",
    "scripts/report_g_v4.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("snapshot contains unsupported path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {".", MANIFEST}:
        raise ValueError("snapshot contains unsafe path")
    return path.as_posix()


def _identity(root: Path) -> tuple[str, str]:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise ValueError(f"G V4 snapshot requires branch {BRANCH}")
    return branch, commit


def _collect(root: Path) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    for relative in DATA_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required G V4 recovery file is missing: {relative}")
        contents[relative] = path.read_bytes()
    for relative in OPTIONAL_FILES:
        path = root / relative
        if path.is_file():
            contents[relative] = path.read_bytes()
    for directory_name in DATA_DIRS:
        directory = root / directory_name
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"unsafe G V4 recovery directory: {directory_name}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlinks are forbidden in a recovery archive: {path}")
            if not path.is_file() or path.name.startswith(".") or path.name.endswith(".tmp") or path.suffix == ".zip":
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
        audit = json.loads(contents[DATA_FILES[3]])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("malformed G V4 recovery metadata") from exc
    if config.get("registered") is not True or config.get("branch") != BRANCH:
        raise ValueError("snapshot does not contain registered G V4 configuration")
    if config.get("config_sha256") != config_digest(config):
        raise ValueError("G V4 config digest mismatch")
    if _sha(contents[DATA_FILES[1]]) != config.get("opening_schedule_sha256"):
        raise ValueError("G V4 opening schedule differs from its registration")
    validate_v3_schedule(schedule)
    if _sha(contents[DATA_FILES[2]]) != config.get("preflight_sha256"):
        raise ValueError("G V4 preflight hash mismatch")
    if _sha(contents[DATA_FILES[3]]) != config.get("continuation_audit_sha256"):
        raise ValueError("G V4 continuation audit hash mismatch")
    if not preflight.get("resources") or not audit.get("checkpoint_verified"):
        raise ValueError("G V4 resource or continuation evidence is incomplete")
    if not contents[DATA_FILES[5]] or _sha(contents[DATA_FILES[5]]) != audit.get("checkpoint_sha256"):
        raise ValueError("V3 champion weight verification failed")
    for name, data in contents.items():
        if not name.endswith(".jsonl"):
            continue
        temp_root = Path(tempfile.mkdtemp(prefix="g_v4_snapshot_validate_"))
        try:
            target = temp_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if "/selfplay/" in name:
                load_game_log(target)
            elif "/evaluation/" in name and not name.endswith("/tactics.jsonl"):
                load_evaluation_log(target)
            else:
                for line in data.splitlines():
                    json.loads(line)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
    return config


def create_v4_snapshot(root: str | Path, output: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    contents = _collect(root_path)
    config = _validate_contents(contents)
    branch, commit = _identity(root_path)
    code_hashes: dict[str, str] = {}
    for relative in CODE_FILES:
        path = root_path / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"snapshot recovery source is missing: {relative}")
        data = path.read_bytes()
        code_hashes[relative] = _sha(data)
        contents[f"source/{relative}"] = data
    hashes = {name: _sha(data) for name, data in sorted(contents.items())}
    active_cpu = cpu_seconds_committed(root_path)
    active_wall = active_wall_seconds_committed(root_path)
    selfplay_games = sum(len([line for line in data.splitlines() if line])
                         for name, data in contents.items() if "/results/selfplay/" in name)
    evaluation_games = sum(len([line for line in data.splitlines() if line])
                           for name, data in contents.items() if "/results/evaluation/" in name)
    manifest = {
        "format_version": 1, "experiment_id": config["experiment_id"],
        "branch": branch, "commit": commit, "config_sha256": config["config_sha256"],
        "opening_schedule_sha256": config["opening_schedule_sha256"],
        "active_cpu_seconds": active_cpu, "active_wall_seconds": active_wall,
        "next_snapshot_threshold_seconds": next_snapshot_threshold(active_wall),
        "selfplay_game_records": selfplay_games, "evaluation_game_records": evaluation_games,
        "code_sha256": code_hashes, "files": hashes,
    }
    contents[MANIFEST] = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(contents.items()):
                archive.writestr(name, data)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        dirfd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    raw = output_path.read_bytes()
    return {"path": str(output_path), "sha256": _sha(raw), "size_bytes": len(raw),
            "branch": branch, "commit": commit, "config_sha256": config["config_sha256"],
            "active_cpu_seconds": active_cpu, "active_wall_seconds": active_wall,
            "next_snapshot_threshold_seconds": manifest["next_snapshot_threshold_seconds"],
            "selfplay_game_records": selfplay_games, "evaluation_game_records": evaluation_games}


def restore_v4_snapshot(archive_path: str | Path, destination: str | Path) -> dict[str, Any]:
    source = Path(archive_path)
    target = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() and any(target.iterdir()):
        raise ValueError("G V4 recovery destination must be empty")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or names.count(MANIFEST) != 1:
            raise ValueError("G V4 snapshot contains duplicate paths or no manifest")
        for info in infos:
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("snapshot entries must be regular files")
            if info.filename != MANIFEST:
                _safe_path(info.filename)
        manifest = json.loads(archive.read(MANIFEST))
        members = {name: archive.read(name) for name in names if name != MANIFEST}
    hashes = manifest.get("files")
    if manifest.get("format_version") != 1 or not isinstance(hashes, dict) or set(hashes) != set(members):
        raise ValueError("G V4 snapshot manifest file list is incomplete")
    for name, data in members.items():
        if _sha(data) != hashes[name]:
            raise ValueError(f"snapshot SHA-256 mismatch: {name}")
    data = {name: value for name, value in members.items() if not name.startswith("source/")}
    allowed = set(DATA_FILES)
    allowed.update(name for name in data if any(name.startswith(f"{directory}/") for directory in DATA_DIRS))
    allowed.update(name for name in OPTIONAL_FILES if name in data)
    source_names = {name for name in members if name.startswith("source/")}
    if set(data) != allowed or source_names != {f"source/{name}" for name in CODE_FILES}:
        raise ValueError("G V4 snapshot contains unsupported or missing recovery files")
    config = _validate_contents(data)
    if (manifest.get("experiment_id") != config["experiment_id"]
            or manifest.get("branch") != BRANCH
            or manifest.get("config_sha256") != config["config_sha256"]
            or manifest.get("opening_schedule_sha256") != config["opening_schedule_sha256"]
            or set(manifest.get("code_sha256", {})) != set(CODE_FILES)):
        raise ValueError("G V4 snapshot manifest metadata differs from its config")
    for relative in CODE_FILES:
        if _sha(members[f"source/{relative}"]) != manifest["code_sha256"][relative]:
            raise ValueError(f"G V4 source-code hash mismatch: {relative}")
    for name, content in data.items():
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    return {"empty_restore": True, "archive_sha256": _sha(source.read_bytes()),
            "branch": manifest["branch"], "commit": manifest["commit"],
            "active_cpu_seconds": manifest["active_cpu_seconds"],
            "active_wall_seconds": manifest["active_wall_seconds"],
            "selfplay_game_records": manifest["selfplay_game_records"],
            "evaluation_game_records": manifest["evaluation_game_records"],
            "restored_files": sorted(data)}

