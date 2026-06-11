# src/training/experiment.py

from src.models.dannynet  import get_dannynet
from src.training.losses  import chexnet_loss, focal_loss
from src.training.train   import train_fine_tuning


def run_experiments(experiments, train_loader, val_loader, output_dir='output'):
    results = {}

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Experimento: {exp['name']}")
        print(f"{'='*50}")

        model   = exp['model_fn']()
        loss_fn = exp['loss_fn']()

        history = train_fine_tuning(
            model           = model,
            train_loader    = train_loader,
            val_loader      = val_loader,
            learning_rate   = exp['lr'],
            loss_fn         = loss_fn,
            optimizer_name  = exp['optimizer_name'],
            experiment_name = exp['name'],
            output_dir      = output_dir,
        )

        results[exp['name']] = history

    return results


EXPERIMENTS = [
    {
        # Baseline: replica CheXNet con SGD y BCE
        "name":           "dannynet_bce_sgd",
        "model_fn":       get_dannynet,
        "loss_fn":        chexnet_loss,
        "lr":             1e-4,
        "optimizer_name": "sgd",
    },
    {
        # Cambia solo el optimizer: SGD → AdamW
        "name":           "dannynet_bce_adamw",
        "model_fn":       get_dannynet,
        "loss_fn":        chexnet_loss,
        "lr":             1e-4,
        "optimizer_name": "adamw",
    },
    {
        # Cambia solo la loss: BCE → Focal Loss
        # Justificación: desbalance severo (Effusion 85k vs Pneumonia 5k)
        "name":           "dannynet_focal_adamw",
        "model_fn":       get_dannynet,
        "loss_fn":        focal_loss,
        "lr":             1e-4,
        "optimizer_name": "adamw",
    },
]