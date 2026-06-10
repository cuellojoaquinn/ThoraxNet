# datasets/unified.py

import torch
import pandas as pd
from torch.utils.data import Dataset


CANONICAL_LABELS = [
    "No_Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Fracture",
    "Hernia",
    "Infiltration",
    "Lung_Lesion",
    "Lung_Opacity",
    "Nodule",
    "Mass",
    "Enlarged_Cardiomediastinum",
    "Pleural_Other",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

CHEXPERT_TO_CANONICAL = {
    "No Finding":                 "No_Finding",
    "Enlarged Cardiomediastinum": "Enlarged_Cardiomediastinum",
    "Cardiomegaly":               "Cardiomegaly",
    "Lung Opacity":               "Lung_Opacity",
    "Lung Lesion":                "Lung_Lesion",
    "Edema":                      "Edema",
    "Consolidation":              "Consolidation",
    "Pneumonia":                  "Pneumonia",
    "Atelectasis":                "Atelectasis",
    "Pneumothorax":               "Pneumothorax",
    "Pleural Effusion":           "Effusion",
    "Pleural Other":              "Pleural_Other",
    "Fracture":                   "Fracture",
}

CX8_TO_CANONICAL = {
    "Atelectasis":        "Atelectasis",
    "Consolidation":      "Consolidation",
    "Infiltration":       "Infiltration",
    "Pneumothorax":       "Pneumothorax",
    "Edema":              "Edema",
    "Emphysema":          "Emphysema",
    "Fibrosis":           "Fibrosis",
    "Effusion":           "Effusion",
    "Pneumonia":          "Pneumonia",
    "Pleural_Thickening": "Pleural_Thickening",
    "Cardiomegaly":       "Cardiomegaly",
    "Nodule":             "Nodule",
    "Mass":               "Mass",
    "Hernia":             "Hernia",
    "No Finding":         "No_Finding",
}

VIN_TO_CANONICAL = {
    "No finding":         "No_Finding",
    "Atelectasis":        "Atelectasis",
    "Cardiomegaly":       "Cardiomegaly",
    "Consolidation":      "Consolidation",
    "Pleural effusion":   "Effusion",
    "Pleural thickening": "Pleural_Thickening",
    "Pneumothorax":       "Pneumothorax",
    "Infiltration":       "Infiltration",
    "Nodule/Mass":        ["Nodule", "Mass"],  # uno-a-muchos
}


class UnifiedDataset(Dataset):

    DATASET_EVALUATES = {
        "chexpert": set(CHEXPERT_TO_CANONICAL.values()),
        "cx8":      set(CX8_TO_CANONICAL.values()),
        "vin":      set(
            label
            for v in VIN_TO_CANONICAL.values()
            for label in (v if isinstance(v, list) else [v])
        ),
    }

    _MAPPINGS = {
        "chexpert": CHEXPERT_TO_CANONICAL,
        "cx8":      CX8_TO_CANONICAL,
        "vin":      VIN_TO_CANONICAL,
    }

    def __init__(self, datasets: dict):
        self.datasets   = datasets
        self._index_map = self._build_index_map()

    def _build_index_map(self) -> list:
        return [
            (name, i)
            for name, ds in self.datasets.items()
            for i in range(len(ds))
        ]

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple:
        name, local_idx   = self._index_map[idx]
        image, raw_labels = self.datasets[name][local_idx]
        labels            = self._map_labels(raw_labels, self._MAPPINGS[name])
        labels_tensor     = torch.tensor(
            [labels[l] for l in CANONICAL_LABELS], dtype=torch.float32
        )
        return image, labels_tensor

    def _map_labels(self, raw_labels: dict, mapping: dict) -> dict:
        canonical = {label: 0 for label in CANONICAL_LABELS}
        for pat, value in raw_labels.items():
            target = mapping.get(pat)
            if target is None:
                continue
            if isinstance(target, list):
                for t in target:
                    canonical[t] = value
            else:
                canonical[target] = value
        return canonical

    def class_distribution(self) -> dict:
        counts = {label: 0 for label in CANONICAL_LABELS}
        for name, ds in self.datasets.items():
            mapping = self._MAPPINGS[name]
            for _, row in ds.df.iterrows():
                labels = ds._extract_labels(row)
                mapped = self._map_labels(labels, mapping)
                for label, value in mapped.items():
                    if value == 1:
                        counts[label] += 1
        return counts