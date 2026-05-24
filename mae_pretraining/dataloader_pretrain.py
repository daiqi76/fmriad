import json
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from dataset_pretrain import IXIOASISDataset


def transform() -> T.Compose:
    """Normalisation-only pipeline"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class IXIOASISDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapping IXIOASIS Dataset.

    Call ``setup()`` before accessing any dataloader.  
    """

    def __init__(
        self,
        plane: str,
        path: Path,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        """
        Args:
            plane:       Input plane {'sagittal', 'coronal', 'axial', 'all'}.
            path:        Path to the dataset.
            batch_size:  Samples per mini-batch.
            num_workers: DataLoader worker processes (0 = main process only).
        """
        super().__init__()
        self.plane       = plane
        self.path        = path
        self.batch_size  = batch_size
        self.num_workers = num_workers


    # ------------------------------------------------------------------
    def train_dataloader(self):
        ds = IXIOASISDataset("train", self.plane, self.path, transform())
        return ds, DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                        num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        ds = IXIOASISDataset("val", self.plane, self.path, transform())
        return ds, DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=True)

