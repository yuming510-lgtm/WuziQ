from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from src.research.abcd_ladder_v5_1 import build_schedule, load_config
from src.research.abcd_ladder_v5_1_analysis import summarize_records
from src.research.abcd_ladder_v5_1_report import build_report
from tests.test_abcd_ladder_v5_1_runner import _complete_legal_record


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "abcd_ladder_v5_1.json"


def test_report_contains_all_routes_real_results_and_v5_score_correction(tmp_path):
    config = load_config(CONFIG)
    schedule = build_schedule(config)
    records = [_complete_legal_record(game, config) for game in schedule[0]["games"]]
    summary = summarize_records(records, config, schedule=schedule)
    results = tmp_path / "results"
    results.mkdir()
    (results / "history_comparison.json").write_text(
        (ROOT / "research/v5/results/history_comparison.json").read_text(), encoding="utf-8"
    )

    report_path = build_report(summary, results, tmp_path / "report")
    report = report_path.read_text(encoding="utf-8")
    svg_path = tmp_path / "report" / "figures" / "ladder_gradient.svg"
    svg_root = ET.parse(svg_path).getroot()

    assert report_path.exists()
    assert "A、B、C、D" in report
    assert "D@20s>B" in report and "D@50s>C" in report
    assert "wins + draws/2" in report
    assert "A-C" in report and "0.5" in report and "0.2" in report
    assert "未执行" in report
    assert svg_root.tag.endswith("svg")
    assert len(summary["route_state"]["routes"]) == 66
    assert len(summary["version_pairwise"]) == 168
