#!/usr/bin/env python3
"""Validate all G V2 artifacts and rebuild the Chinese report and figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v2_report import (
    build_v2_report,
    render_v2_report,
    write_v2_integrity_manifest,
)


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = build_v2_report(root)
    recovery = root / "research/g_v2/results/recovery/drive_status.json"
    if recovery.is_file():
        summary["drive_status"] = json.loads(recovery.read_text())
    else:
        summary["drive_status"] = {"status": "未验证"}
    outputs = render_v2_report(root, summary)
    integrity = write_v2_integrity_manifest(root)
    result = {"outputs": outputs, "integrity_artifacts": len(integrity["artifacts"]),
              "training_states": sum(item["states"] for item in summary["training"]["candidates"].values()),
              "blind_games": summary["metadata"]["blind_games"],
              "actual_cpu_seconds": summary["metadata"]["total_actual_cpu_seconds"]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
