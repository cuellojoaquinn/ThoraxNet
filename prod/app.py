import io
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from utils import CANONICAL_LABELS, LABEL_ES, LABEL_CSV_ES, EXAMPLE_LABELS, load_model, predict, img_to_b64, get_example_images

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
if "scroll_to_upload" not in st.session_state:
    st.session_state.scroll_to_upload = False

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫁 Tórax IA")
    st.caption("Apoyo a la lectura · v0.4 (beta)")
    st.divider()
    st.caption("Modelo · DenseNet-121")
    st.caption("Entrada 224×224 · 8 salidas")

# ── Ejemplos de prueba ────────────────────────────────────────────────────────

@st.fragment
def examples_section():
    with st.expander("Ejemplos de prueba por patología", expanded=True):
        selected = st.radio(
            "Patología",
            list(EXAMPLE_LABELS.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        examples = get_example_images(selected)

        if examples:
            cols = st.columns(len(examples))
            for col, (img_path, label) in zip(cols, examples):
                with col:
                    st.caption(img_path.name)
                    b64 = img_to_b64(Image.open(img_path))
                    st.markdown(
                        f'<div style="height:220px;display:flex;align-items:center;'
                        f'justify-content:center;margin-bottom:0.5rem;">'
                        f'<img src="data:image/png;base64,{b64}" '
                        f'style="max-height:220px;max-width:100%;object-fit:contain;border-radius:6px;"/>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Cargar", key=str(img_path), use_container_width=True):
                        with open(img_path, "rb") as f:
                            st.session_state.img_bytes = f.read()
                        st.session_state.scroll_to_upload = True
                        st.rerun(scope="app")

            st.markdown("**Patologías presentes en cada imagen**")
            img_pats: dict[str, list[str]] = {}
            all_pats: list[str] = []
            for i, (_, label) in enumerate(examples, 1):
                pats = (
                    [p.strip() for p in label.replace(", ", " | ").split(" | ")]
                    if label else []
                )
                img_pats[f"Img {i}"] = pats
                for p in pats:
                    if p not in all_pats:
                        all_pats.append(p)

            if all_pats:
                img_keys = list(img_pats.keys())
                header    = "| Patología | " + " | ".join(img_keys) + " |"
                separator = "|:---|" + ":---:|" * len(img_keys)
                rows_md   = [header, separator]
                for pat in all_pats:
                    cells = " | ".join("✓" if pat in img_pats[k] else "" for k in img_keys)
                    rows_md.append(f"| {LABEL_CSV_ES.get(pat, pat)} | {cells} |")
                st.markdown("\n".join(rows_md))
            else:
                st.caption("Sin etiquetas CSV disponibles para estas imágenes.")
        else:
            st.caption("Sin imágenes disponibles para esta patología.")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Lectura asistida")
st.markdown("""
1. Cargá un ejemplo de abajo o subí tu propia radiografía
2. Hacé clic en **Analizar imagen**
3. Revisá las predicciones orientativas en el panel derecho
""")

examples_section()

st.divider()

st.markdown('<div id="seccion-radiografia"></div>', unsafe_allow_html=True)

if st.session_state.scroll_to_upload:
    st.session_state.scroll_to_upload = False
    components.html(
        """<script>
        setTimeout(function() {
            var el = window.parent.document.getElementById('seccion-radiografia');
            if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
        }, 200);
        </script>""",
        height=0,
    )

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
