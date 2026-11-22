"""Train a PyHealth MLP on SEER 12/60-month survival classification.

Keep the input CSV and PyHealth cache within approved SEER research storage.
This script prints aggregate performance only and does not save model weights.
"""

import argparse
import random

import numpy as np
import torch

from pyhealth.datasets import PatientSplit, get_dataloader
from pyhealth.datasets.seer import SEERDataset
from pyhealth.models import MLP
from pyhealth.tasks.seer_survival import SEERSurvivalPrediction
from pyhealth.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Directory with seer_cases.csv")
    p.add_argument("--cache", required=True, help="Approved secure PyHealth cache")
    p.add_argument("--months", type=int, choices=(12, 60), default=12)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dev", action="store_true", help="Use a small smoke-test cohort")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = SEERDataset(
        root=args.root, cache_dir=args.cache, num_workers=1, dev=args.dev
    )
    task = SEERSurvivalPrediction(args.months)
    train_ds, val_ds, test_ds = dataset.set_task(
        task,
        num_workers=1,
        split=PatientSplit(ratios=(0.8, 0.1, 0.1), seed=seed),
    )
    print(f"Task: {args.months}-month survival")
    print(f"Train/val/test samples: {len(train_ds):,}/{len(val_ds):,}/{len(test_ds):,}")

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    def print_class_counts(name, loader):
        survived = 0
        died = 0

        for batch in loader:
            labels = torch.as_tensor(
                batch["label"]
            ).reshape(-1).cpu()

            survived += int((labels == 1).sum())
            died += int((labels == 0).sum())

        print(
            f"{name}: survived={survived}, "
            f"cancer_deaths={died}"
        )

        return survived, died


    train_counts = print_class_counts("Train", train_loader)
    val_counts = print_class_counts("Validation", val_loader)
    test_counts = print_class_counts("Test", test_loader)

    if min(val_counts) == 0:
        print(
            "WARNING: Validation contains only one class. "
            "ROC-AUC cannot be calculated."
        )

    model = MLP(dataset=train_ds, embedding_dim=64, hidden_dim=64, n_layers=2)
    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "f1", "accuracy"],
        device=args.device,
        enable_logging=False,  # Prevent writing model checkpoints into public repos.
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="loss" if args.dev else "roc_auc",
        monitor_criterion="min" if args.dev else "max",
        patience=2,
    )
    scores = trainer.evaluate(test_loader)
    print("Held-out test metrics (aggregate only):")
    for name in ("roc_auc", "pr_auc", "f1", "accuracy", "loss"):
        if name in scores:
            print(f"  {name}: {scores[name]:.4f}")


if __name__ == "__main__":
    main()
