# models.py
# Model definitions for CN/AD classification without domain-specific pretraining.
# Implements ResNet-50, ViT-B/16, and Swin-B with two-layer FC classification heads.
# All weights are randomly initialised (no pretrained weights); the full network
# (backbone + head) is trained end-to-end as a supervised baseline for comparison
# with MAE-pretrained models in Step 3.

from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, F1Score
from torchvision import models

# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def build_backbone(model_name: str) -> Tuple[nn.Module, int]:
    """Instantiate an ImageNet-pretrained backbone with its classifier removed.

    Returns:
        backbone:    The feature-extraction network (no classification head).
        feature_dim: Dimensionality of the backbone output vector.
    """
    if model_name == "resnet50":
        m = models.resnet50(weights=None)       # random initialisation
        feature_dim = m.fc.in_features          # 2048
        m.fc = nn.Identity()
        return m, feature_dim

    elif model_name == "vitb16":
        m = models.vit_b_16(weights=None)       # random initialisation
        feature_dim = m.heads.head.in_features  # 768
        m.heads = nn.Identity()
        return m, feature_dim

    elif model_name == "swinb":
        m = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,                   # random initialisation
            num_classes=0,
        )
        feature_dim = m.num_features            # 1024
        return m, feature_dim

    else:
        raise ValueError(f"Unsupported model: '{model_name}'. "
                         f"Choose from ['resnet50', 'vitb16', 'swinb'].")


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

def build_head(feature_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Two-layer FC head: Linear → BN → ReLU → Dropout → Linear (logits).

    Args:
        feature_dim: Input dimensionality (from backbone).
        hidden_dim:  Hidden layer width.
        dropout:     Dropout probability between the two FC layers.
    """
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 2),
    )


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class ClassifierModule(pl.LightningModule):
    """Binary CN/AD classifier with randomly initialised backbone and head.

    Both backbone and head are trained end-to-end from random weights.
    This serves as the supervised baseline (Step 1) against which MAE-pretrained
    models (Step 3) are compared.  Learning rate follows a cosine annealing
    schedule from ``lr`` to near-zero over ``max_epochs``.
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        max_epochs: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone, feature_dim = build_backbone(model_name)
        self.head = build_head(feature_dim, hidden_dim, dropout)

        # All parameters are trainable — no freezing

        # Torchmetrics — instantiated per split to avoid state leakage
        self.train_acc = Accuracy(task="binary")
        self.val_acc   = Accuracy(task="binary")
        self.val_auc   = AUROC(task="binary")
        self.test_acc  = Accuracy(task="binary")
        self.test_auc  = AUROC(task="binary")
        self.test_f1   = F1Score(task="binary")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # ------------------------------------------------------------------
    # Training / validation / test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y, label_smoothing=0.1)
        preds  = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss,           on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_auc(probs, y)
        self.log("val/loss", loss,          on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc,  on_epoch=True, prog_bar=True)
        self.log("val/auc",  self.val_auc,  on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        self.test_acc(preds, y)
        self.test_auc(probs, y)
        self.test_f1(preds, y)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/auc", self.test_auc, on_epoch=True)
        self.log("test/f1",  self.test_f1,  on_epoch=True)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW on all parameters, with cosine annealing LR schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "name":      "lr",
            },
        }
