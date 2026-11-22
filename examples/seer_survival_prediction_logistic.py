"""SEER survival prediction: Logistic Regression class-weighting ablation.

Run from PyHealth repository root, with all patient-level files and caches in
approved SEER research storage OUTSIDE the Git repository.

Examples (PowerShell or bash):

    python examples/seer_survival_prediction_logistic.py \
      --root <approved_root> --cache <approved_cache> --months 12 --ablation

    python examples/seer_survival_prediction_logistic.py \
      --root <approved_root> --cache <approved_cache> --months 60 --ablation

This is an adapted breast-cancer reproduction. For each configuration,
``seer_train_baselines.main()`` fits a majority-class dummy model and logistic
regression, tunes C using validation only, and prints aggregate metrics.

The ablation runs regular Logistic Regression and class_weight='balanced'
using the SAME patient-level split seed (42) and C search; it compares death
recall/precision/F1, ROC-AUC and accuracy. No trained model weights or
individual predictions are saved by the underlying training script.
"""

from __future__ import annotations

import argparse
import sys

from seer_train_baselines import main as baseline_main


def main() -> None:
    """Run one baseline or both class-weight configurations sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--months", choices=(12, 60), type=int, default=12)
    parser.add_argument("--c-values", default="0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run both ordinary and class-weighted Logistic Regression",
    )
    args = parser.parse_args()
    original_argv = sys.argv
    try:
        variants = (False, True) if args.ablation else (False,)
        for weighted in variants:
            print("\n=== SEER class-weighting ablation: " +
                  ("balanced" if weighted else "none") + " ===")
            sys.argv = [
                "seer_train_baselines.py",
                "--root", args.root,
                "--cache", args.cache,
                "--months", str(args.months),
                "--c-values", args.c_values,
                "--max-iter", str(args.max_iter),
                "--batch-size", str(args.batch_size),
            ]
            if args.dev:
                sys.argv.append("--dev")
            if weighted:
                sys.argv.append("--class-weight-balanced")
            baseline_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
