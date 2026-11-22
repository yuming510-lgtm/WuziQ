from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.alphazero_g.v6_7_training import prepare_experiment
from src.research.alphazero_g.v6_7_evaluation import create_opening_pools


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage the verified G8000 V6.7 continuation point")
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--source-node", required=True, type=Path)
    parser.add_argument("--drive-folder-id", default="1bkakjo5fdhgELAzyuOohDE8wynVBY1fh")
    args = parser.parse_args()
    result = prepare_experiment(source_state_path=args.source_state,
                                source_node_path=args.source_node,
                                drive_folder_id=args.drive_folder_id)
    result["opening_pools"] = create_opening_pools()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
