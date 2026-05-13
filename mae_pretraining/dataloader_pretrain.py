import json
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold


class IXIOASISDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapping IXIOASIS Dataset.

    Call ``setup()`` before accessing any dataloader.  
    """

    def __init__(
        self,
        plane: str,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        """
        Args:
            plane:       Input plane {'sagittal', 'coronal', 'axial', 'all'}.
            batch_size:  Samples per mini-batch.
            n_splits:    Number of CV folds (default 5).
            num_workers: DataLoader worker processes (0 = main process only).
        """
        super().__init__()
        self.plane       = plane
        self.batch_size  = batch_size
        self.n_splits    = n_splits
        self.num_workers = num_workers

        self._train_records: Optional[List[dict]] = None
        self._val_records:   Optional[List[dict]] = None

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
        ds = IXIOASISDataset(self._train_records, self.plane, get_train_transform())
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        ds = IXIOASISDataset(self._val_records, self.plane, get_val_transform())
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

