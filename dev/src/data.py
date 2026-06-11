import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src import (
    CheXpertDataset, ChestXray8Dataset, VinBigDataset,
    UnifiedDataset, ProjectionStrategy, ViewStrategy
)
from transforms import FourierAmplitudeMixup


def make_chexpert_splits(dataset_root, seed=42):
    df = pd.read_csv(os.path.join(dataset_root, "chexpert", "train.csv")).iloc[1:].reset_index(drop=True)
    df["patient_id"] = df["Path"].apply(lambda x: x.split("/")[2])

    patients = df["patient_id"].unique()
    patients = np.array(patients) 
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    train_patients = set(patients[:int(n * 0.8)])
    val_patients   = set(patients[int(n * 0.8):int(n * 0.9)])
    test_patients  = set(patients[int(n * 0.9):])

    train_paths = set(df[df["patient_id"].isin(train_patients)]["Path"])
    val_paths   = set(df[df["patient_id"].isin(val_patients)]["Path"])
    test_paths  = set(df[df["patient_id"].isin(test_patients)]["Path"])

    print(f"CheXpert — train: {len(train_paths)} | val: {len(val_paths)} | test: {len(test_paths)}")
    return train_paths, val_paths, test_paths


def make_cx8_splits(dataset_root, seed=42):
    df = pd.read_csv(os.path.join(dataset_root, "ChestXray8", "Data_Entry_2017.csv"))
    train_val = set(open(os.path.join(dataset_root, "ChestXray8", "train_val_list.txt")).read().splitlines())
    test_ids  = set(open(os.path.join(dataset_root, "ChestXray8", "test_list.txt")).read().splitlines())

    df_trainval = df[df["Image Index"].isin(train_val)]
    patients = df_trainval["Patient ID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    train_patients = set(patients[:int(n * 0.89)])
    val_patients   = set(patients[int(n * 0.89):])

    train_ids = set(df_trainval[df_trainval["Patient ID"].isin(train_patients)]["Image Index"])
    val_ids   = set(df_trainval[df_trainval["Patient ID"].isin(val_patients)]["Image Index"])

    print(f"CX8     — train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


def make_vinbig_splits(dataset_root, seed=42):
    df = pd.read_csv(os.path.join(dataset_root, "VinBigData", "train.csv"))
    image_ids = df["image_id"].unique()
    image_ids = np.array(image_ids) 
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    train_ids = set(image_ids[:int(n * 0.8)])
    val_ids   = set(image_ids[int(n * 0.8):int(n * 0.9)])
    test_ids  = set(image_ids[int(n * 0.9):])

    print(f"VinBig  — train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


def get_loaders(dataset_root: str, batch_size: int = 32, num_workers: int = 0):

    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base_transforms = T.Compose([
        T.Resize(320), T.CenterCrop(224), T.ToTensor(), normalize,
    ])

    chexpert_train, chexpert_val, chexpert_test = make_chexpert_splits(dataset_root)
    cx8_train,      cx8_val,      cx8_test      = make_cx8_splits(dataset_root)
    vin_train,      vin_val,      vin_test       = make_vinbig_splits(dataset_root)

    chexpert_csv  = os.path.join(dataset_root, "chexpert", "train.csv")
    chexpert_imgs = os.path.join(dataset_root, "chexpert", "train")
    cx8_csv       = os.path.join(dataset_root, "ChestXray8", "Data_Entry_2017.csv")
    vin_csv       = os.path.join(dataset_root, "VinBigData", "train.csv")
    vin_imgs      = [
        os.path.join(dataset_root, "VinBigData", "train"),
        os.path.join(dataset_root, "VinBigData", "test"),
    ]

    train_chexpert = CheXpertDataset(root=dataset_root, csv_path=chexpert_csv,
                                     images_dir=chexpert_imgs, split_ids=chexpert_train,
                                     projection=ProjectionStrategy.ALL,
                                     view=ViewStrategy.FRONTAL_ONLY)
    val_chexpert   = CheXpertDataset(root=dataset_root, csv_path=chexpert_csv,
                                     images_dir=chexpert_imgs, split_ids=chexpert_val,
                                     projection=ProjectionStrategy.ALL,
                                     view=ViewStrategy.FRONTAL_ONLY)
    test_chexpert  = CheXpertDataset(root=dataset_root, csv_path=chexpert_csv,
                                     images_dir=chexpert_imgs, split_ids=chexpert_test,
                                     projection=ProjectionStrategy.ALL,
                                     view=ViewStrategy.FRONTAL_ONLY)

    train_cx8 = ChestXray8Dataset(root=dataset_root, csv_path=cx8_csv,
                                  projection=ProjectionStrategy.ALL, split_ids=cx8_train)
    val_cx8   = ChestXray8Dataset(root=dataset_root, csv_path=cx8_csv,
                                  projection=ProjectionStrategy.ALL, split_ids=cx8_val)
    test_cx8  = ChestXray8Dataset(root=dataset_root, csv_path=cx8_csv,
                                  projection=ProjectionStrategy.ALL, split_ids=cx8_test)

    train_vin = VinBigDataset(root=dataset_root, csv_path=vin_csv,
                              images_dirs=vin_imgs, split_ids=vin_train)
    val_vin   = VinBigDataset(root=dataset_root, csv_path=vin_csv,
                              images_dirs=vin_imgs, split_ids=vin_val)
    test_vin  = VinBigDataset(root=dataset_root, csv_path=vin_csv,
                              images_dirs=vin_imgs, split_ids=vin_test)

    fourier_mixup = FourierAmplitudeMixup(
        datasets=[train_chexpert, train_cx8], beta=0.01, p=0.5,
    )
    train_augs = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.75, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        fourier_mixup, T.ToTensor(), normalize,
    ])

    train_chexpert.transform = train_augs
    train_cx8.transform      = train_augs
    train_vin.transform      = base_transforms
    val_chexpert.transform   = base_transforms
    val_cx8.transform        = base_transforms
    val_vin.transform        = base_transforms
    test_chexpert.transform  = base_transforms
    test_cx8.transform       = base_transforms
    test_vin.transform       = base_transforms

    train_unified = UnifiedDataset(datasets={
        "chexpert": train_chexpert, "cx8": train_cx8, "vin": train_vin,
    })
    val_unified = UnifiedDataset(datasets={
        "chexpert": val_chexpert, "cx8": val_cx8, "vin": val_vin,
    })
    test_unified = UnifiedDataset(datasets={
        "chexpert": test_chexpert, "cx8": test_cx8, "vin": test_vin,
    })

    train_loader = DataLoader(train_unified, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_unified,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_unified,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader