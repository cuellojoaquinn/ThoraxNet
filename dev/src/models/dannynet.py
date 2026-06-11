import torch.nn as nn
import torchvision.models as models
from src import CANONICAL_LABELS


def get_dannynet():
    """
    DenseNet121 preentrenado en ImageNet con classifier reemplazado
    para clasificación multilabel de patologías torácicas.
    Arquitectura base de CheXNet (Rajpurkar et al., 2017).
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(CANONICAL_LABELS))
    nn.init.xavier_uniform_(model.classifier.weight)
    return model
