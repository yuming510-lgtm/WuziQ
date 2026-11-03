#!/usr/bin/env python3
"""Validate paired V4.1 evaluation logs and write efficiency summaries."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_1_analysis import write_phase_summary
from src.research.alphazero_g.v4_1_runner import _cpu_used, _record_cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "blind"))
    args = parser.parse_args()
    source = ROOT / f"research/g_v4_1/results/evaluation/{args.phase}.jsonl"
    target = ROOT / f"research/g_v4_1/results/evaluation/{args.phase}_summary.json"
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    summary = write_phase_summary(source, target, phase=args.phase)
    cpu_seconds, wall_seconds = time.process_time() - cpu_started, time.perf_counter() - wall_started
    if _cpu_used(ROOT) + cpu_seconds > 6 * 3600:
        raise RuntimeError("G V4.1 six CPU-hour hard cap reached during result analysis")
    _record_cpu(ROOT, f"analysis:{args.phase}:{time.time_ns()}", cpu_seconds, wall_seconds, "analysis")
    print(json.dumps({"summary_path": str(target.relative_to(ROOT)),
                      "games": summary["games"], "comparisons": summary["comparison_count"],
                      "cpu_seconds": cpu_seconds}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
