from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_9_training import prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage the verified G20,000 full state for V6.9")
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--source-node", required=True, type=Path)
    parser.add_argument("--drive-folder-id", default="1Tk9gkA0MwzMMB6rueZB-IpFWLVrVx6Qq")
    args = parser.parse_args()
    print(json.dumps(prepare_experiment(source_state_path=args.source_state,
                                        source_node_path=args.source_node,
                                        drive_folder_id=args.drive_folder_id),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
