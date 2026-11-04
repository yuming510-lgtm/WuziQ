from __future__ import annotations

import json
from pathlib import Path

from src.research.alphazero_g.v6_audit import audit_v6


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = audit_v6(ROOT)
    output = ROOT / "research/g_v6/results/integrity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    temp.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
