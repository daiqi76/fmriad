# train.py
# Training script for a single fold of 5-fold cross-validation.
# Trains ResNet-50, ViT-B/16, or Swin-B on the ADNI CN/AD classification task
# without domain-specific pretraining (ImageNet weights only).
# The backbone is always frozen; only the classification head is trained.
# Logs metrics to both WandB (cloud visualisation) and a local CSV file
# (offline analysis).  A JSON summary of the best validation metrics is also
# written at the end of each run.
#
# Usage:
#   python train.py --model resnet50 --plane sagittal --fold 0
#   python train.py --model vitb16   --plane all      --fold 2 --config configs/vitb16.yaml

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from dataset import ADNIDataModule
from models import ClassifierModule


# ---------------------------------------------------------------------------
# Training-summary callback
# ---------------------------------------------------------------------------

class TrainingSummaryCallback(Callback):
    """Tracks best validation metrics across epochs and writes a JSON summary
    at the end of training.

    The summary file is saved to:
        logs/{model_name}/{plane}/fold_{fold}/summary.json

    It records the best val/auc, the corresponding val/acc, the epoch at which
    the best was achieved, and the total number of epochs trained.
    """

    def __init__(self, log_dir: Path, model_name: str, plane: str, fold: int):
        self.out_path   = log_dir / "summary.json"
        self.model_name = model_name
        self.plane      = plane
        self.fold       = fold

        self._best_auc   = 0.0
        self._best_acc   = 0.0
        self._best_epoch = 0

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        metrics = trainer.callback_metrics
        val_auc = float(metrics.get("val/auc", 0.0))
        val_acc = float(metrics.get("val/acc", 0.0))
        if val_auc > self._best_auc:
            self._best_auc   = val_auc
            self._best_acc   = val_acc
            self._best_epoch = trainer.current_epoch

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        summary = {
            "model":         self.model_name,
            "plane":         self.plane,
            "fold":          self.fold,
            "best_val_auc":  self._best_auc,
            "best_val_acc":  self._best_acc,
            "best_epoch":    self._best_epoch,
            "total_epochs":  trainer.current_epoch + 1,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[summary] Saved training summary → {self.out_path}")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> str:
    """Run training for one fold and return the path to the best checkpoint."""

    # Load model config from YAML
    config_path = args.config or f"configs/{args.model}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    dm = ADNIDataModule(
        plane=args.plane,
        fold=args.fold,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ClassifierModule(
        model_name=cfg["model_name"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        max_epochs=cfg["max_epochs"],
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    run_name = f"{cfg['model_name']}_{args.plane}_fold{args.fold}"
    ckpt_dir = Path(f"checkpoints/{cfg['model_name']}/{args.plane}/fold_{args.fold}")
    log_dir  = Path(f"logs/{cfg['model_name']}/{args.plane}/fold_{args.fold}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val/auc",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/auc",
        patience=cfg["patience"],
        mode="max",
        verbose=True,
    )

    summary_cb = TrainingSummaryCallback(
        log_dir=log_dir,
        model_name=cfg["model_name"],
        plane=args.plane,
        fold=args.fold,
    )

    # ── Loggers ───────────────────────────────────────────────────────────
    # WandB: uploads metrics to the cloud for interactive visualisation.
    # Runs are grouped by model+plane so all 5 folds appear together.
    wandb_logger = WandbLogger(
        project="fmriad_without_pretrain",
        name=run_name,
        group=f"{cfg['model_name']}_{args.plane}",   # group folds for comparison
        tags=[cfg["model_name"], args.plane, f"fold{args.fold}"],
        save_dir="logs",    # local WandB cache (logs/wandb/)
        log_model=False,    # do not upload checkpoints to WandB
    )
    wandb_logger.log_hyperparams(cfg)

    # CSV: writes per-epoch metrics to logs/{model}/{plane}/fold_{k}/metrics.csv
    # for offline analysis without WandB access.
    csv_logger = CSVLogger(
        save_dir="logs",
        name=f"{cfg['model_name']}/{args.plane}",
        version=f"fold_{args.fold}",
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        callbacks=[checkpoint_cb, early_stop_cb, summary_cb],
        logger=[wandb_logger, csv_logger],
        accelerator="gpu",
        devices=1,
        deterministic=True,
        log_every_n_steps=5,
    )

    trainer.fit(model, dm)

    best_path = checkpoint_cb.best_model_path
    print(f"\n[fold {args.fold}] Best checkpoint saved: {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one fold for CN/AD classification (without pretraining)."
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
        "--fold",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4],
        help="Fold index (0-based) for 5-fold cross-validation.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file. Defaults to configs/{model}.yaml.",
    )
    args = parser.parse_args()
    train(args)
