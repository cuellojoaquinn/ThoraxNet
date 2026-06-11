# src/training/train.py

import os
import torch


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    L = 0.0
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.float().to(device)
        optimizer.zero_grad()
        l = loss_fn(model(X), y)
        l.backward()
        optimizer.step()
        L += l.item()
        if (i + 1) % 10 == 0:
            print(f'batch {i+1}/{len(loader)} | loss {L/(i+1):.4f}')
    return L / len(loader)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    L = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.float().to(device)
            L += loss_fn(model(X), y).item()
    return L / len(loader)


def get_optimizer(model, optimizer_name, learning_rate, param_group=True):
    if param_group:
        params_1x = [p for n, p in model.named_parameters()
                     if n not in ["classifier.weight", "classifier.bias"]]
        param_groups = [
            {'params': params_1x},
            {'params': model.classifier.parameters(), 'lr': learning_rate * 10}
        ]
    else:
        param_groups = model.parameters()

    if optimizer_name == 'sgd':
        return torch.optim.SGD(param_groups, lr=learning_rate, weight_decay=0.001)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=1e-5)
    else:
        raise ValueError(f"Optimizer no soportado: {optimizer_name}. Usar 'sgd' o 'adamw'.")


def train_fine_tuning(model, train_loader, val_loader, learning_rate,
                      num_epochs=5, param_group=True, loss_fn=None,
                      optimizer_name='sgd', experiment_name='model',
                      output_dir='output'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    if loss_fn is None:
        from src.training.losses import chexnet_loss
        loss_fn = chexnet_loss()

    optimizer = get_optimizer(model, optimizer_name, learning_rate, param_group)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch+1}/{num_epochs} ---')
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss   = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'epoch {epoch+1} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}')

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{experiment_name}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Modelo guardado en {checkpoint_path}')

    return history