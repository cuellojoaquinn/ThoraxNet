import torch
import torch.nn as nn


def chexnet_loss():
    """
    Binary cross entropy loss para clasificación multilabel.
    Equivalente a la loss de CheXNet para 14 clases (Rajpurkar et al., 2017).
    reduction='none' para permitir enmascaramiento de etiquetas inciertas (U-Ignore).
    """
    return nn.BCEWithLogitsLoss(reduction='none')


def focal_loss(alpha=1, gamma=2):
    """
    Focal loss para clasificación multilabel con desbalance de clases.
    Reduce el peso de ejemplos fáciles y enfoca el entrenamiento en ejemplos difíciles.
    reduction='none' para permitir enmascaramiento de etiquetas inciertas (U-Ignore).
    Ref: Lin et al., RetinaNet, CVPR 2017.
    """
    class FocalLoss(nn.Module):
        def __init__(self, alpha, gamma):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.bce   = nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, inputs, targets):
            bce_loss  = self.bce(inputs, targets)
            pt        = torch.exp(-bce_loss)
            loss      = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return loss  # sin .mean() — se aplica desde train_one_epoch

    return FocalLoss(alpha=alpha, gamma=gamma)