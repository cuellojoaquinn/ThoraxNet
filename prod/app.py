import io
import base64
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────

CANONICAL_LABELS = [
    "No_Finding", "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Pneumonia", "Pneumothorax",
]

LABEL_ES = {
    "No_Finding":   "Sin hallazgos",
    "Atelectasis":  "Atelectasia",
    "Cardiomegaly": "Cardiomegalia",
    "Consolidation":"Consolidación",
    "Edema":        "Edema pulmonar",
    "Effusion":     "Derrame pleural",
    "Pneumonia":    "Neumonía",
    "Pneumothorax": "Neumotórax",
}

_base = Path(__file__).parent
MODEL_PATH = (
    _base / "model" / "e2_dannynet_focal_adamw.pth"           # HF Spaces / raíz
    if (_base / "model" / "e2_dannynet_focal_adamw.pth").exists()
    else _base.parent / "model" / "e2_dannynet_focal_adamw.pth"  # local (prod/)
)

TRANSFORM = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Modelo ────────────────────────────────────────────────────────────────────

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

# ── Estilos ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Barra de progreso roja */
[data-testid="stProgressBar"] > div > div {
    background-color: #e53935 !important;
}

/* Uploader: zona dashed minimalista */
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: 2px dashed #555 !important;
    border-radius: 10px !important;
    padding: 2rem 1rem !important;
    text-align: center !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #e53935 !important;
}
[data-testid="stFileUploader"] section > div {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    display: none;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 0.85rem;
    color: #888;
}

/* Zona de preview de imagen */
.image-zone {
    border: 2px dashed #555;
    border-radius: 10px;
    padding: 8px;
    text-align: center;
}
.image-zone img {
    border-radius: 6px;
    max-width: 100%;
}

.prob-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.9rem;
    margin-bottom: 2px;
}
.prob-val {
    font-weight: 600;
    min-width: 38px;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫁 Tórax IA")
    st.caption("Apoyo a la lectura · v0.4 (beta)")
    st.divider()
    st.caption("Modelo · DenseNet-121")
    st.caption("Entrada 224×224 · 8 salidas")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Lectura asistida")
st.caption("Subí una radiografía de tórax para obtener predicciones orientativas")

col_upload, col_pred = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("**Radiografía**")

    if st.session_state.img_bytes is None:
        f = st.file_uploader(
            label="Arrastrá una radiografía o hacé clic para buscar",
            type=["png", "jpg", "jpeg"],
        )
        if f:
            st.session_state.img_bytes = f.read()
            st.rerun()
        image = None
        analyze = False
    else:
        image = Image.open(io.BytesIO(st.session_state.img_bytes))
        b64 = img_to_b64(image)
        st.markdown(
            f'<div class="image-zone">'
            f'<img src="data:image/png;base64,{b64}"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        col_btn, col_change = st.columns([3, 1])
        with col_btn:
            analyze = st.button("Analizar imagen", type="primary", use_container_width=True)
        with col_change:
            if st.button("✕", use_container_width=True, help="Cambiar imagen"):
                st.session_state.img_bytes = None
                st.rerun()

with col_pred:
    st.markdown("**Predicciones del modelo**")

    if image is not None and analyze:
        st.caption("Estudio procesado · resultados orientativos")
        st.warning(
            "**Resultado orientativo, no es un diagnóstico.** "
            "Las cifras son probabilidades estimadas por el modelo y pueden contener errores. "
            "Confirme siempre con la lectura radiológica y el criterio clínico."
        )

        st.caption("8 hallazgos · ordenados de mayor a menor probabilidad")

        model = load_model()
        scores = predict(model, image)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for label, prob in sorted_scores:
            st.markdown(
                f'<div class="prob-row">'
                f'<span>{LABEL_ES[label]}</span>'
                f'<span class="prob-val">{prob:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(prob)

        st.caption("ℹ El color no representa gravedad. Solo la longitud de la barra refleja la probabilidad estimada (0–1).")

    elif image is not None:
        st.caption("Presioná **Analizar imagen** para obtener predicciones")
    else:
        st.caption("Cargá una imagen para ver los resultados")
