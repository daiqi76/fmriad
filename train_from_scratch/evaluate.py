# evaluate.py
# Evaluation script for CN/AD classification using 5-fold checkpoint ensemble.
# Loads the best checkpoint from each fold, averages raw logits across all folds
# on the held-out test set, and reports Accuracy, AUC, F1, Sensitivity, and
# Specificity.  Results are saved to results/{model}_{plane}_results.json.
#
# Usage:
#   python evaluate.py --model resnet50 --plane sagittal

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import IMAGE_ROOT, ADNIDataset, get_val_transform
from models import ClassifierModule


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> dict:
    """Run 5-fold ensemble evaluation on the held-out ADNI test set.

    For each fold, the best checkpoint is loaded and logits are collected for
    every test sample.  The per-fold logits are averaged (soft voting) before
    computing final metrics.

    Returns:
        Dictionary containing all computed metrics.
    """

    config_path = args.config or f"configs/{args.model}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Test dataset ──────────────────────────────────────────────────────
    with open(IMAGE_ROOT / "ADNI_test.json") as f:
        test_records = json.load(f)

    test_ds = ADNIDataset(test_records, args.plane, get_val_transform())
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect logits from each fold ─────────────────────────────────────
    all_fold_logits: list[torch.Tensor] = []
    true_labels:     torch.Tensor | None = None

    for fold in range(5):
        ckpt_path = Path(
            f"checkpoints/{cfg['model_name']}/{args.plane}/fold_{fold}/best.ckpt"
        )
        if not ckpt_path.exists():
            print(f"  [WARNING] Checkpoint not found, skipping: {ckpt_path}")
            continue

        print(f"  Loading fold {fold}: {ckpt_path}")
        model = ClassifierModule.load_from_checkpoint(str(ckpt_path))
        model.eval()
        model.to(device)

        fold_logits: list[torch.Tensor] = []
        fold_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                fold_logits.append(logits.cpu())
                fold_labels.append(y)

        all_fold_logits.append(torch.cat(fold_logits, dim=0))   # (N, 2)

        # Collect ground-truth labels only once
        if true_labels is None:
            true_labels = torch.cat(fold_labels, dim=0)         # (N,)

    if not all_fold_logits:
        raise RuntimeError("No valid checkpoints found. Run train.py for all 5 folds first.")

    n_folds = len(all_fold_logits)
    print(f"\nEnsemble over {n_folds} fold(s).")

    # ── Ensemble: average logits, then softmax ────────────────────────────
    ensemble_logits = torch.stack(all_fold_logits, dim=0).mean(dim=0)  # (N, 2)
    probs  = torch.softmax(ensemble_logits, dim=1)[:, 1].numpy()       # AD probability
    preds  = ensemble_logits.argmax(dim=1).numpy()
    labels = true_labels.numpy()

    # ── Metrics ───────────────────────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # AD recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # CN recall

    # ── Print results ─────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Model : {cfg['model_name']}   Plane : {args.plane}")
    print(f"  Folds used in ensemble : {n_folds}")
    print(sep)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  AUC         : {auc:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}  (AD recall)")
    print(f"  Specificity : {specificity:.4f}  (CN recall)")
    print(f"  Confusion matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(sep)

    # ── Save to JSON ──────────────────────────────────────────────────────
    results = {
        "model":       cfg["model_name"],
        "plane":       args.plane,
        "n_folds":     n_folds,
        "accuracy":    float(acc),
        "auc":         float(auc),
        "f1":          float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }

    out_dir  = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{cfg['model_name']}_{args.plane}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {out_path}\n")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5-fold ensemble evaluation on the held-out ADNI test set."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet50", "vitb16", "swinb"],
        help="Backbone architecture.",
    )
    parser.add_argument(
        "--plane",
        required=True,
        choices=["sagittal", "coronal", "axial", "all"],
        help="Input MRI plane.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file. Defaults to configs/{model}.yaml.",
    )
    args = parser.parse_args()
    evaluate(args)
