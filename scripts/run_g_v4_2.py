#!/usr/bin/env python3
"""Resume and run the complete registered WuziQ G V4.2 experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.alphazero_g.v4_2_audit import audit_v4_2_results
from src.research.alphazero_g.v4_2_config import load_v4_2_config
from src.research.alphazero_g.v4_2_experiment import run_v4_2_diagnostics
from src.research.alphazero_g.v4_2_final import (
    evaluate_arm_final_head_to_head, freeze_final_candidate, run_v4_2_blind,
)
from src.research.alphazero_g.v4_2_runtime import BudgetExhausted, V4_2Runtime
from src.research.alphazero_g.v4_2_storage import atomic_write_json, load_json
from src.research.alphazero_g.v4_2_train import train_v4_2_arms


def main() -> int:
    config = load_v4_2_config(ROOT)
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    preflight_path = ROOT / "research/g_v4_2/results/preflight/preflight.json"
    if not preflight_path.is_file():
        raise SystemExit("run python scripts/preflight_g_v4_2.py before formal execution")
    preflight = load_json(preflight_path)
    runtime = V4_2Runtime(ROOT, preflight_cpu_seconds=float(preflight["preflight_cpu_seconds"]))
    runtime.save("formal_run_started", config_sha256=config["config_sha256"])
    try:
        diagnostics_path = ROOT / "research/g_v4_2/results/diagnostics/diagnostics.json"
        if diagnostics_path.exists():
            diagnostics = load_json(diagnostics_path)
            print("V4.2 frozen-weight/search diagnostics already complete; resuming after them.", flush=True)
        else:
            print("V4.2 phase 1/4: historical checkpoint and PUCT diagnostics", flush=True)
            diagnostics = run_v4_2_diagnostics(ROOT, runtime)
        print(json.dumps(diagnostics["search_selection"], ensure_ascii=False), flush=True)

        training_path = ROOT / "research/g_v4_2/results/training/summary.json"
        if training_path.exists():
            training = load_json(training_path)
            print("V4.2 controlled training is already complete; reusing verified progress.", flush=True)
        else:
            print("V4.2 phase 2/4: controlled G0 self-play training and stage evaluations", flush=True)
            training = train_v4_2_arms(ROOT, runtime)
        print(json.dumps({"active_arms": training["active_arms"],
                          "states_by_arm": training["states_by_arm"]}, ensure_ascii=False), flush=True)

        schedule = load_json(ROOT / config["opening_schedule_path"])
        internal_path = ROOT / "research/g_v4_2/results/training/arm_internal_summary.json"
        if not internal_path.exists():
            evaluate_arm_final_head_to_head(ROOT, config, schedule, training["active_arms"], runtime)
        freeze_final_candidate(ROOT, config, training["active_arms"])

        blind_path = ROOT / "research/g_v4_2/results/blind/summary.json"
        if blind_path.exists():
            blind = load_json(blind_path)
            print("V4.2 final blind report already complete; verifying it.", flush=True)
        else:
            print("V4.2 phase 3/4: frozen-candidate blind tests on new 20-pair openings", flush=True)
            blind = run_v4_2_blind(ROOT, runtime)
        print(json.dumps({"champion_decision": blind["champion_decision"],
                          "games": blind["games"]}, ensure_ascii=False), flush=True)

        print("V4.2 phase 4/4: complete result, replay, budget and provenance audit", flush=True)
        audit = audit_v4_2_results(ROOT)
        runtime.save("all_replays_and_data_audited", audit=audit, completed=True)
        print(json.dumps({"audit": audit, "total_cpu_seconds": runtime.total_cpu_seconds}, ensure_ascii=False), flush=True)
        from src.research.alphazero_g.v4_2_report import build_v4_2_report
        build_v4_2_report(ROOT)
        runtime.save("completed", report="research/g_v4_2/REPORT.md")
        print("V4.2 experiment completed; REPORT.md and integrity audit are ready.", flush=True)
        return 0
    except BudgetExhausted as exc:
        runtime.save("paused_at_registered_cpu_cap", status="incomplete", reason=str(exc))
        print(f"V4.2 stopped at hard CPU budget without claiming completion: {exc}", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
