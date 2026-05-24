# dataset.py
# ADNI dataset loader and PyTorch Lightning DataModule for CN/AD binary classification.
# Supports 4 input planes (sagittal, coronal, axial, all) and 5-fold stratified cross-validation.

import json
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IMAGE_ROOT = Path("F:/daiqi_fmriad/images/without_pretrain")

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform() -> T.Compose:
    """Augmentation + normalisation pipeline for the training split.

    Augmentation strategy is intentionally strong to counteract overfitting on
    the small ADNI dataset (~294 training samples per fold):
      - RandomHorizontalFlip + RandomAffine: geometric diversity
      - GaussianBlur: simulates acquisition noise / scanner variation
      - ColorJitter: mild intensity variation
    """
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> T.Compose:
    """Normalisation-only pipeline for validation and test splits."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ADNIDataset(Dataset):
    """Loads 2-D MRI slice PNGs from ADNI JSON metadata.

    Grayscale planes (sagittal / coronal / axial) are replicated to 3 channels
    so that ImageNet-pretrained models can be used directly.  The composite
    'all' plane is already stored as an RGB PNG.
    """

    LABEL_MAP = {"CN": 0, "AD": 1}

    def __init__(self, records: List[dict], plane: str, transform=None):
        """
        Args:
            records:   List of metadata dicts loaded from ADNI_*.json.
            plane:     One of {'sagittal', 'coronal', 'axial', 'all'}.
            transform: torchvision transform applied to the PIL image.
        """
        self.records   = records
        self.plane     = plane
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec      = self.records[idx]
        img_path = IMAGE_ROOT / rec[self.plane]
        img      = Image.open(img_path)

        if self.plane == "all":
            img = img.convert("RGB")
        else:
            # Grayscale → 3-channel RGB (channel replication)
            img = img.convert("L").convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.LABEL_MAP[rec["group"]]
        return img, label


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ADNIDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapping 5-fold stratified CV on ADNI.

    Call ``setup()`` before accessing any dataloader.  The test set is always
    loaded from ``ADNI_test.json`` and is never part of any fold.
    """

    def __init__(
        self,
        plane: str,
        fold: int,
        batch_size: int = 8,
        n_splits: int = 5,
        num_workers: int = 0,
    ):
        """
        Args:
            plane:       Input plane {'sagittal', 'coronal', 'axial', 'all'}.
            fold:        Zero-based fold index in [0, n_splits).
            batch_size:  Samples per mini-batch.
            n_splits:    Number of CV folds (default 5).
            num_workers: DataLoader worker processes (0 = main process only).
        """
        super().__init__()
        self.plane       = plane
        self.fold        = fold
        self.batch_size  = batch_size
        self.n_splits    = n_splits
        self.num_workers = num_workers

        self._train_records: Optional[List[dict]] = None
        self._val_records:   Optional[List[dict]] = None
        self._test_records:  Optional[List[dict]] = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        """Loads JSON metadata and computes fold splits."""
        # Train / val pool
        with open(IMAGE_ROOT / "ADNI_train_val.json") as f:
            all_records = json.load(f)

        # Held-out test set
        with open(IMAGE_ROOT / "ADNI_test.json") as f:
            self._test_records = json.load(f)

        # Stratified 5-fold split (seed fixed for reproducibility)
        labels = [1 if r["group"] == "AD" else 0 for r in all_records]
        skf    = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(all_records, labels))

        train_idx, val_idx = splits[self.fold]
        self._train_records = [all_records[i] for i in train_idx]
        self._val_records   = [all_records[i] for i in val_idx]

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        ds = ADNIDataset(self._train_records, self.plane, get_train_transform())
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        ds = ADNIDataset(self._val_records, self.plane, get_val_transform())
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        ds = ADNIDataset(self._test_records, self.plane, get_val_transform())
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
