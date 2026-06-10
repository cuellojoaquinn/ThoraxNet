from pathlib import Path

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .acquisition import ProjectionStrategy


CX8_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "No Finding"
]


class ChestXray8Dataset(Dataset):

    def __init__(self, root, csv_path, split_ids=None, transform=None,
                 projection: ProjectionStrategy = ProjectionStrategy.AP_ONLY):
        self.root      = Path(root)
        self.transform = transform

        self.df        = pd.read_csv(csv_path)
        self._path_map = self._build_path_map()

        if split_ids is not None:
            self.df = self.df[self.df["Image Index"].isin(set(split_ids))]

        self.df = self._filter_projection(self.df, projection)
        self.df = self.df.reset_index(drop=True)

    def _filter_projection(self, df: pd.DataFrame, strategy: ProjectionStrategy) -> pd.DataFrame:
        if strategy == ProjectionStrategy.ALL:
            return df
        return df[df["View Position"] == strategy.value].reset_index(drop=True)

    def _build_path_map(self) -> dict:
        path_map = {}
        cx8_root = self.root / "ChestXray8"
        for folder in sorted(os.listdir(cx8_root)):
            if folder.startswith("images_"):
                img_dir = cx8_root / folder / "images"
                if img_dir.is_dir():
                    for fname in os.listdir(img_dir):
                        path_map[fname] = img_dir / fname
        return path_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row    = self.df.iloc[idx]
        image  = self._load_image(row)
        labels = self._extract_labels(row)
        return image, labels

    def _load_image(self, row: pd.Series):
        img_path = self._path_map.get(row["Image Index"])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def _extract_labels(self, row: pd.Series) -> dict:
        findings = set(row["Finding Labels"].split("|"))
        return {label: 1 if label in findings else 0 for label in CX8_LABELS}