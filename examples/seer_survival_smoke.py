"""Run a PyHealth SEER Dataset + Task smoke test without printing patient records."""
from __future__ import annotations

import argparse

from pyhealth.datasets.seer import SEERDataset
from pyhealth.tasks.seer_survival import SEERSurvivalPrediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Approved input directory")
    parser.add_argument("--cache", required=True, help="Approved cache directory")
    parser.add_argument("--months", type=int, choices=[12, 60], default=12)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    dataset = SEERDataset(root=args.root, cache_dir=args.cache, dev=args.dev)
    samples = dataset.set_task(SEERSurvivalPrediction(args.months), num_workers=1)
    print("Task:", args.months, "months")
    print("Eligible samples:", len(samples))
    print("No patient-level records were printed")


if __name__ == "__main__":
    main()
