#!/usr/bin/env python3
"""Run or resume one registered G V2 learning-rate candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.research.alphazero_g.v2_runner import run_v2_candidate


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("research/g_v2/config/experiment.json"))
    parser.add_argument("--candidate", choices=("lr_1e4", "lr_3e5", "lr_1e3"), required=True)
    parser.add_argument("--until-node", type=int)
    parser.add_argument("--stop-after-games", type=int)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    if config.get("registered") is not True:
        raise ValueError("G V2 candidate runner requires the frozen registered experiment config")
    result = run_v2_candidate(
        root, config, args.candidate, until_node=args.until_node,
        stop_after_games=args.stop_after_games,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
