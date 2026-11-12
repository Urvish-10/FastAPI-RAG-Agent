"""
Rebuild metrics.json and report.md from a saved results.json (no API calls).
Useful after changing metric code.

Usage:
    uv run python -m app.evaluation.report                 # latest run in eval_results/
    uv run python -m app.evaluation.report eval_results/20261112-190859
"""
import argparse
import json
from pathlib import Path

from app.evaluation.metrics import compute_metrics, render_markdown
from app.evaluation.runner import RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir or sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir())[-1]
    data = json.loads((run_dir / "results.json").read_text())
    scored = [r for r in data["records"] if not r["error"]]
    metrics = compute_metrics(scored)
    (run_dir / "metrics.json").write_text(json.dumps({"meta": data["meta"], "metrics": metrics}, indent=2))
    report = render_markdown(metrics, data["meta"])
    (run_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
