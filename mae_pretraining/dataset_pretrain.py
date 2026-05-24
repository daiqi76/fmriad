from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class IXIOASISDataset(Dataset):
    def __init__(self, split: str, plane: str, path: Path, transform=None):
        self.transform = transform
        self.plane = plane
        split_plain_dir = self.path / split / plane
        self.records = [
            p for p in sorted(split_plain_dir.glob("*"))
            if p.is_file()
        ]
        

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img = Image.open(self.records[idx]).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img}