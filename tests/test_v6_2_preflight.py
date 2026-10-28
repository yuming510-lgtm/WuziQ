from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_script_imports_its_frozen_engine_adapter() -> None:
    source = ROOT / "scripts/preflight_v6_2.py"
    spec = importlib.util.spec_from_file_location("v6_2_preflight", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module._decision)
    assert callable(module._estimates)
