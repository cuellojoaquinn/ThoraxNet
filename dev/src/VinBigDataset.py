# datasets/vinbig.py

from pathlib import Path

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


VIN_LABELS = [
    "No finding", "Atelectasis", "Cardiomegaly", "Consolidation",
    "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Infiltration", "Nodule/Mass"
]


class VinBigDataset(Dataset):

    def __init__(self, root, csv_path, images_dirs: list, split_ids=None, transform=None):
        self.root      = Path(root)
        self.transform = transform

        df = pd.read_csv(csv_path)
        if split_ids is not None:
            df = df[df["image_id"].isin(split_ids)]

        self.df        = self._build_majority_vote(df)
        self._path_map = self._build_path_map(images_dirs)

    def _build_majority_vote(self, df: pd.DataFrame) -> pd.DataFrame:
        votes = (
            df.groupby(["image_id", "class_name"])["rad_id"]
            .nunique().reset_index().rename(columns={"rad_id": "n_votes"})
        )
        n_rads            = df.groupby("image_id")["rad_id"].nunique()
        votes["threshold"] = votes["image_id"].map(n_rads) * 0.5
        votes["positive"]  = (votes["n_votes"] >= votes["threshold"]).astype(float)

        all_images = df["image_id"].unique()
        result = (
            votes.pivot(index="image_id", columns="class_name", values="positive")
            .reindex(index=all_images, columns=VIN_LABELS)
            .fillna(0)
            .reset_index()
        )
        result.columns.name = None
        return result

    def _build_path_map(self, images_dirs: list) -> dict:
        path_map = {}
        for images_dir in images_dirs:
            for fname in os.listdir(images_dir):
                name = os.path.splitext(fname)[0]
                path_map[name] = Path(images_dir) / fname
        return path_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row    = self.df.iloc[idx]
        image  = self._load_image(row)
        labels = self._extract_labels(row)
        return image, labels

    def _load_image(self, row: pd.Series):
        img_path = self._path_map.get(row["image_id"])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def _extract_labels(self, row: pd.Series) -> dict:
        return {label: int(row[label]) for label in VIN_LABELS}