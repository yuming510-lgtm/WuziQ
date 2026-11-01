#!/usr/bin/env python3
"""Create the deterministic, history-excluded V3 training/evaluation openings."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v3_preflight import _atomic_json
from src.research.alphazero_g.v3_schedule import build_v3_schedule


def _historical_openings(root: Path) -> list[list[dict[str, object]]]:
    result = []
    for version in ("g_v1", "g_v2"):
        path = root / f"research/{version}/config/openings.json"
        schedule = json.loads(path.read_text(encoding="utf-8"))
        for section in ("train_openings", "development_openings", "blind_openings"):
            for group in schedule.get(section, []):
                opening = group.get("opening", [])
                if opening:
                    result.append(opening)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[510901, 510902])
    parser.add_argument("--train-groups-per-seed", type=int, default=2048)
    parser.add_argument("--development-groups", type=int, default=12)
    parser.add_argument("--blind-groups", type=int, default=20)
    args = parser.parse_args()
    root = args.root.resolve()
    schedule = build_v3_schedule(
        args.seeds,
        train_groups_per_seed=args.train_groups_per_seed,
        development_groups=args.development_groups,
        blind_groups=args.blind_groups,
        excluded_openings=_historical_openings(root),
    )
    output = root / "research/g_v3/config/openings.json"
    _atomic_json(output, schedule)
    print(json.dumps({
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "schedule_sha256": schedule["schedule_sha256"],
        "seeds": schedule["seeds"],
        "train_groups_per_seed": schedule["train_groups_per_seed"],
        "development_groups": schedule["development_groups"],
        "blind_groups": schedule["blind_groups"],
        "historical_openings_excluded": len(schedule["excluded_opening_digests"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
