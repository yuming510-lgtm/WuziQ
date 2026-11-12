from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_archive(archive: str | Path, output_dir: str | Path, *, part_size: int = 20 * 1024 * 1024) -> dict:
    archive, output_dir = Path(archive), Path(output_dir)
    if part_size <= 0:
        raise ValueError("part_size must be positive")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    size = archive.stat().st_size
    part_count = (size + part_size - 1) // part_size
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    with archive.open("rb") as source:
        for index in range(part_count):
            name = f"{archive.name}.part-{index + 1:03d}-of-{part_count:03d}"
            target = output_dir / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            digest = hashlib.sha256()
            part_size_actual = 0
            with temporary.open("wb") as destination:
                while part_size_actual < part_size:
                    chunk = source.read(min(1024 * 1024, part_size - part_size_actual))
                    if not chunk:
                        break
                    destination.write(chunk)
                    digest.update(chunk)
                    part_size_actual += len(chunk)
            os.replace(temporary, target)
            parts.append({
                "index": index + 1, "file_name": name,
                "size_bytes": part_size_actual, "sha256": digest.hexdigest(),
            })
    manifest = {
        "format_version": 1,
        "archive_name": archive.name,
        "size_bytes": size,
        "sha256": _sha256(archive),
        "parts": parts,
    }
    if sum(part["size_bytes"] for part in parts) != size:
        raise ValueError("split parts do not match archive size")
    return manifest


def join_archive(manifest: dict | str | Path, part_dir: str | Path, output: str | Path) -> dict:
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    part_dir, output = Path(part_dir), Path(output)
    parts = manifest.get("parts", [])
    if not parts:
        raise ValueError("multipart snapshot manifest has no parts")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    temporary = Path(temp_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as destination:
            for expected_index, part in enumerate(parts, start=1):
                name = str(part["file_name"])
                if Path(name).name != name or int(part.get("index", expected_index)) != expected_index:
                    raise ValueError("unsafe or out-of-order multipart file name")
                source = part_dir / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                part_digest = hashlib.sha256()
                part_size = 0
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                        part_digest.update(chunk)
                        size += len(chunk)
                        part_size += len(chunk)
                if part_size != int(part["size_bytes"]) or part_digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"part hash mismatch: {name}")
        if size != int(manifest["size_bytes"]) or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("reassembled archive hash mismatch")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(output), "size_bytes": size, "sha256": digest.hexdigest(), "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Split and restore-verify large V6.3 Drive snapshots")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--archive", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--part-size", type=int, default=20 * 1024 * 1024)
    split.add_argument("--manifest-out", required=True)
    join = commands.add_parser("join")
    join.add_argument("--manifest", required=True)
    join.add_argument("--part-dir", required=True)
    join.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "split":
        result = split_archive(args.archive, args.output_dir, part_size=args.part_size)
        Path(args.manifest_out).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        result = join_archive(args.manifest, args.part_dir, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
