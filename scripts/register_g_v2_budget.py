#!/usr/bin/env python3
"""Freeze G V2 quotas from measured preflight and the checked schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.research.alphazero_g.v2_preflight import register_budget_in_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight", type=Path, default=Path("research/g_v2/results/preflight/preflight.json"))
    parser.add_argument("--config", type=Path, default=Path("research/g_v2/config/experiment.json"))
    parser.add_argument("--schedule", type=Path, default=Path("research/g_v2/config/openings.json"))
    args = parser.parse_args()
    root = args.root.resolve()

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    config = register_budget_in_config(
        absolute(args.preflight), absolute(args.config), absolute(args.schedule), absolute(args.config)
    )
    print(f"Registered {config['states_per_candidate']} states per LR candidate at PUCT-{config['simulation_budget']}; "
          f"nodes={config['stage_nodes']}; conservative_cpu_seconds={config['estimates']['conservative_cpu_seconds']:.1f}")


if __name__ == "__main__":
    main()
