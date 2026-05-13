import json
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


IMAGE_ROOT = Path("/Data")

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

class IXIOASISDataset(Dataset):
    """Loads 2-D MRI slice PNGs from ADNI JSON metadata.

    Grayscale planes (sagittal / coronal / axial) are replicated to 3 channels
    so that ImageNet-pretrained models can be used directly.  The composite
    'all' plane is already stored as an RGB PNG.
    """

    LABEL_MAP = {"CN": 0, "AD": 1}

    def __init__(self, plane: str, transform=None):
        """
        Args:
            plane:     One of {'sagittal', 'coronal', 'axial', 'all'}.
            transform: torchvision transform applied to the PIL image.
        """
        self.plane     = plane
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
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