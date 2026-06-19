import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import streamlit as st

CANONICAL_LABELS = [
    "No_Finding", "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Pneumonia", "Pneumothorax",
]

LABEL_ES = {
    "No_Finding":    "Sin hallazgos",
    "Atelectasis":   "Atelectasia",
    "Cardiomegaly":  "Cardiomegalia",
    "Consolidation": "Consolidación",
    "Edema":         "Edema pulmonar",
    "Effusion":      "Derrame pleural",
    "Pneumonia":     "Neumonía",
    "Pneumothorax":  "Neumotórax",
}

_base = Path(__file__).parent
MODEL_PATH = (
    _base / "model" / "e2_dannynet_focal_adamw.pth"
    if (_base / "model" / "e2_dannynet_focal_adamw.pth").exists()
    else _base.parent / "model" / "e2_dannynet_focal_adamw.pth"
)

TRANSFORM = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CANONICAL_LABELS))

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, image: Image.Image) -> dict[str, float]:
    img = image.convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze().tolist()
    return dict(zip(CANONICAL_LABELS, probs))


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
