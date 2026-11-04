from __future__ import annotations

import json
from pathlib import Path

from src.research.alphazero_g.v6_1_tournament import audit_v61_games


if __name__ == "__main__":
    result = audit_v61_games(Path(__file__).resolve().parents[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
