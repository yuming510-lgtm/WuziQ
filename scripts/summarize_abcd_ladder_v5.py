import argparse
import json
from pathlib import Path

from src.research.abcd_ladder_v5_analysis import write_analysis
from src.research.abcd_ladder_v5_report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Revalidate and summarize V5 ladder results.")
    parser.add_argument("--results-dir", default="research/v5/results")
    parser.add_argument("--output-dir", default="research/v5")
    args = parser.parse_args()
    results = Path(args.results_dir)
    summary = write_analysis(results, results)
    report = build_report(summary, results, args.output_dir)
    print(json.dumps({
        "games": summary["actual_games"], "groups": summary["actual_groups"],
        "report": str(report), "routes": summary["route_state"]["routes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
