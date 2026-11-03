from __future__ import annotations

import json
from pathlib import Path

from src.research.alphazero_g.v5_audit import audit_v5


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = audit_v5(ROOT)
    destination = ROOT / "research/g_v5/results/integrity_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
