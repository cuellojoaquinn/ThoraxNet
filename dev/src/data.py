# src/data.py

import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src import (
    CheXpertDataset, ChestXray8Dataset, VinBigDataset,
    UnifiedDataset, ProjectionStrategy, ViewStrategy
)
from transforms import FourierAmplitudeMixup


def get_loaders(dataset_root: str, batch_size: int = 32, num_workers: int = 0):

    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    base_transforms = T.Compose([
        T.Resize(320),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    # --- Datasets sin transform (se asigna después) ---
    train_ids = set(open(os.path.join(dataset_root, "ChestXray8", "train_val_list.txt")).read().splitlines())
    test_ids  = set(open(os.path.join(dataset_root, "ChestXray8", "test_list.txt")).read().splitlines())

    train_chexpert = CheXpertDataset(
        root=dataset_root,
        csv_path=os.path.join(dataset_root, "chexpert", "train.csv"),
        images_dir=os.path.join(dataset_root, "chexpert", "train"),
        projection=ProjectionStrategy.ALL,
        view=ViewStrategy.FRONTAL_ONLY,
    )
    train_chest8 = ChestXray8Dataset(
        root=dataset_root,
        csv_path=os.path.join(dataset_root, "ChestXray8", "Data_Entry_2017.csv"),
        projection=ProjectionStrategy.ALL,
        split_ids=train_ids,
    )
    test_chexpert = CheXpertDataset(
        root=dataset_root,
        csv_path=os.path.join(dataset_root, "chexpert", "valid.csv"),
        images_dir=os.path.join(dataset_root, "chexpert", "valid"),
        projection=ProjectionStrategy.ALL,
        view=ViewStrategy.FRONTAL_ONLY,
    )
    test_chest8 = ChestXray8Dataset(
        root=dataset_root,
        csv_path=os.path.join(dataset_root, "ChestXray8", "Data_Entry_2017.csv"),
        projection=ProjectionStrategy.ALL,
        split_ids=test_ids,
    )
    vinbig = VinBigDataset(
        root=dataset_root,
        csv_path=os.path.join(dataset_root, "VinBigData", "train.csv"),
        images_dirs=[
            os.path.join(dataset_root, "VinBigData", "train"),
            os.path.join(dataset_root, "VinBigData", "test"),
        ],
    )

    # --- Fourier Mixup con referencias de train ---
    fourier_mixup = FourierAmplitudeMixup(
        datasets=[train_chexpert, train_chest8],
        beta=0.01,
        p=0.5,
    )

    train_augs = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.75, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        fourier_mixup,
        T.ToTensor(),
        normalize,
    ])

    # --- Asignar transforms ---
    train_chexpert.transform = train_augs
    train_chest8.transform   = train_augs
    vinbig.transform         = base_transforms
    test_chexpert.transform  = base_transforms
    test_chest8.transform    = base_transforms

    # --- Unified datasets ---
    train_unified = UnifiedDataset(datasets={"chexpert": train_chexpert, "cx8": train_chest8})
    val_unified   = UnifiedDataset(datasets={"vin": vinbig})
    test_unified  = UnifiedDataset(datasets={"chexpert": test_chexpert, "cx8": test_chest8})

    # --- Loaders ---
    torch.manual_seed(42)
    train_loader = DataLoader(train_unified, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_unified,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_unified,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader