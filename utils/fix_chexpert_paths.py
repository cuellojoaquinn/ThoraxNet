"""
Fix CheXpert CSV paths to match the current flat directory structure.

Old format:  CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
New format:  chexpert/train/patient00001_study1_view1_frontal.jpg
"""

import pandas as pd
from pathlib import Path

CHEXPERT_DIR = Path(__file__).resolve().parents[1] / "chexpert"


def remap_path(old_path: str, split: str) -> str:
    parts = old_path.replace("\\", "/").split("/")
    # last 3 parts: patientXXXX / studyY / viewZ.jpg
    patient, study, fname = parts[-3], parts[-2], parts[-1]
    new_fname = f"{patient}_{study}_{fname}"
    return f"chexpert/{split}/{new_fname}"


def fix_csv(split: str) -> None:
    csv_path = CHEXPERT_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)

    original_sample = df["Path"].iloc[0]
    df["Path"] = df["Path"].apply(lambda p: remap_path(p, split))
    new_sample = df["Path"].iloc[0]

    df.to_csv(csv_path, index=False)

    print(f"[{split}.csv]  {len(df):,} rows updated")
    print(f"  before: {original_sample}")
    print(f"  after:  {new_sample}")


if __name__ == "__main__":
    fix_csv("train")
    fix_csv("valid")
    print("\nDone.")
