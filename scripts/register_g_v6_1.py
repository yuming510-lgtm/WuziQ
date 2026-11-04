from __future__ import annotations

import json

from src.research.alphazero_g.v6_1_experiment import register_v61


if __name__ == "__main__":
    registration = register_v61()
    print(json.dumps({
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "checkpoint_sha256": {name: row["sha256"] for name, row in registration["models"].items()},
        "new_tactics": registration["new_tactics"],
        "openings": registration["opening_schedule_seeds"],
        "selection_rule_id": registration["selection_rule_id"],
    }, ensure_ascii=False, indent=2))
