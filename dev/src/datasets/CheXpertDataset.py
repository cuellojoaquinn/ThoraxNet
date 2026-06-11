from pathlib import Path

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .acquisition import ProjectionStrategy, ViewStrategy


CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture"
]


class CheXpertDataset(Dataset):

    def __init__(self, root, csv_path: str | Path, images_dir: str | Path,
                 transform=None,
                 projection: ProjectionStrategy = ProjectionStrategy.AP_ONLY,
                 view: ViewStrategy = ViewStrategy.FRONTAL_ONLY,
                 split_ids=None):
        self.root       = root
        self.images_dir = Path(images_dir)
        self.transform  = transform

        self.df = pd.read_csv(csv_path).iloc[1:].reset_index(drop=True)
        self.df = self._filter_view(self.df, view)
        self.df = self._filter_projection(self.df, projection)

        if split_ids is not None:
            self.df = self.df[self.df["Path"].isin(split_ids)].reset_index(drop=True)

    def _filter_view(self, df: pd.DataFrame, strategy: ViewStrategy) -> pd.DataFrame:
        if strategy == ViewStrategy.ALL:
            return df
        return df[df["Frontal/Lateral"] == strategy.value].reset_index(drop=True)

    def _filter_projection(self, df: pd.DataFrame, strategy: ProjectionStrategy) -> pd.DataFrame:
        if strategy == ProjectionStrategy.ALL:
            return df
        return df[df["AP/PA"] == strategy.value].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        try:
            image = self._load_image(row)
        except Exception:
            return self.__getitem__(idx + 1)
        labels = self._extract_labels(row)
        return image, labels

    def _load_image(self, row: pd.Series):
        image = Image.open(os.path.join(self.root, row["Path"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
    def _load_image_raw(self, row: pd.Series) -> Image.Image:
        # CheXpert
        return Image.open(os.path.join(self.root, row["Path"])).convert("RGB")

    def _extract_labels(self, row: pd.Series) -> dict:
        labels = {}
        for label in CHEXPERT_LABELS:
            value = row[label]
            labels[label] = 0 if pd.isna(value) else int(value)
        return labels