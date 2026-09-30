"""Run the whole pipeline from the command line.

    python main.py                 # both tiers, full data
    python main.py --tier clinical # only the clinical risk tier
    python main.py --tier ecg      # only the ECG diagnosis tier
    python main.py --fast          # quick smoke test on 8,000 rows
"""
import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

from cardio_ml import report  # noqa: E402
from cardio_ml.workflows import ClinicalRiskWorkflow, ECGDiagnosisWorkflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=["clinical", "ecg", "all"], default="all")
    parser.add_argument("--fast", action="store_true", help="subsample to 8,000 rows (smoke test only)")
    args = parser.parse_args()

    workflows = {"clinical": ClinicalRiskWorkflow, "ecg": ECGDiagnosisWorkflow}
    selected = workflows if args.tier == "all" else {args.tier: workflows[args.tier]}
    for name, cls in selected.items():
        start = time.perf_counter()
        tracker = cls(fast=args.fast).run_all()
        print(f"\n{name} tier finished in {(time.perf_counter() - start) / 60:.1f} min -> {tracker.root}")
    if not args.fast:
        report.main()


if __name__ == "__main__":
    main()
