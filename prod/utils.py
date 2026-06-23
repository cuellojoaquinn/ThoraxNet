import io
import json
import base64
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import streamlit as st

_base = Path(__file__).parent
_MODEL_DIR = (
    _base / "model" / "v2.0"
    if (_base / "model" / "v2.0").exists()
    else _base.parent / "model" / "v2.0"
)

with open(_MODEL_DIR / "densenet_focal_moderate_test_metrics.json") as _f:
    _metrics = json.load(_f)

CANONICAL_LABELS = list(_metrics["per_label"].keys())
THRESHOLDS: dict[str, float] = {
    label: info["threshold"] for label, info in _metrics["per_label"].items()
}

LABEL_ES = {
    "No_Finding":                 "Sin hallazgos",
    "Enlarged_Cardiomediastinum": "Ensanchamiento mediastínico",
    "Cardiomegaly":               "Cardiomegalia",
    "Lung_Opacity":               "Opacidad pulmonar",
    "Lung_Lesion":                "Lesión pulmonar",
    "Edema":                      "Edema pulmonar",
    "Consolidation":              "Consolidación",
    "Pneumonia":                  "Neumonía",
    "Atelectasis":                "Atelectasia",
    "Pneumothorax":               "Neumotórax",
    "Effusion":                   "Derrame pleural",
    "Pleural_Other":              "Otra patología pleural",
    "Fracture":                   "Fractura",
}

LABEL_CSV_ES: dict[str, str] = {
    "Atelectasis":        "Atelectasia",
    "Cardiomegaly":       "Cardiomegalia",
    "Consolidation":      "Consolidación",
    "Edema":              "Edema pulmonar",
    "Effusion":           "Derrame pleural",
    "Pneumonia":          "Neumonía",
    "Pneumothorax":       "Neumotórax",
    "No Finding":         "Sin hallazgo",
    "Emphysema":          "Enfisema",
    "Mass":               "Masa pulmonar",
    "Infiltration":       "Infiltración",
    "Lung Opacity":       "Opacidad pulmonar",
    "Pleural Effusion":   "Derrame pleural",
    "Support Devices":    "Dispositivos de soporte",
    "Lung Lesion":        "Lesión pulmonar",
    "Pleural Other":      "Otra patología pleural",
    "Fracture":           "Fractura",
    "Nodule":             "Nódulo",
    "Fibrosis":           "Fibrosis",
    "Hernia":             "Hernia",
    "Pleural_Thickening": "Engrosamiento pleural",
    "Enlarged Cardiomediastinum": "Ensanchamiento mediastínico",
}

EXAMPLE_LABELS: dict[str, dict[str, str | None]] = {
    "Atelactasia": {
        "1.png": "Atelectasis",
        "2.png": "Atelectasis | Cardiomegaly | Emphysema | Mass | Pneumothorax",
        "3.jpg": "Atelectasis | Cardiomegaly | Lung Opacity | Pneumothorax | Support Devices",
    },
    "Cardiomegalia": {
        "8.jpeg":  None,
        "11.jpeg": None,
        "3.png":   "Cardiomegaly",
    },
    "Consolidacion": {
        "1.png": "Atelectasis | Consolidation | Edema | Pneumonia",
        "2.png": "Consolidation",
        "3.jpg": "Consolidation | Lung Opacity | Pleural Effusion | Support Devices",
    },
    "Derrames Pleurales": {
        "6.jpeg": None,
        "2.png":  "Cardiomegaly | Effusion",
        "3.png":  "Effusion | Infiltration",
    },
    "Edema pulmonar": {
        "1.png": "Cardiomegaly | Edema | Effusion",
        "2.png": "Cardiomegaly | Edema | Effusion",
        "3.jpg": "Edema",
    },
    "Neumonía": {
        "5.jpeg":  None,
        "7.jpeg":  None,
        "10.jpeg": None,
        "12.jpeg": None,
    },
    "Neumotorax": {
        "1.png": "Emphysema | Pneumothorax",
        "2.png": "Emphysema | Pneumothorax",
        "3.jpg": "Pneumothorax",
    },
    "Sin hallazgo": {
        "1.jpeg": None,
        "2.jpeg": None,
        "3.jpeg": None,
        "4.jpeg": None,
    },
}


def get_example_images(folder: str) -> list[tuple[Path, str | None]]:
    data_dir = Path(__file__).parent / "data" / folder
    result = []
    for fname, label in EXAMPLE_LABELS.get(folder, {}).items():
        p = data_dir / fname
        if p.exists():
            result.append((p, label))
    return result

MODEL_PATH = _MODEL_DIR / "densenet_focal_moderate.pth"

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
