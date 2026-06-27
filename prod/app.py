import io
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from utils import LABEL_ES, LABEL_CSV_ES, EXAMPLE_LABELS, THRESHOLDS, load_model, predict, img_to_b64, get_example_images

PRIMARY       = "#FF4B4B"
PRIMARY_LIGHT = "#FFE8E8"

# ── Tema ──────────────────────────────────────────────────────────────────────

_dark = (st.get_option("theme.base") or "light") == "dark"

if _dark:
    T_LABEL      = "#e8e8e8"
    T_DONE       = "#888"
    T_FUTURE     = "#4a4a4a"
    C_FUTURE_BG  = "#3a3a3a"
    C_FUTURE_FG  = "#666"
    LINE_OFF     = "#3a3a3a"
    BORDER       = "#4a4a4a"
    BAR_BG       = "#3a3a3a"
    THR_COLOR    = "#aaa"
else:
    T_LABEL      = "#222"
    T_DONE       = "#999"
    T_FUTURE     = "#bbb"
    C_FUTURE_BG  = "#ddd"
    C_FUTURE_FG  = "#aaa"
    LINE_OFF     = "#ddd"
    BORDER       = "#bbb"
    BAR_BG       = "#e0e0e0"
    THR_COLOR    = "#333"

# ── Estilos ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
[data-testid="stFileUploader"] section {{
    background: transparent !important;
    border: 2px dashed {BORDER} !important;
    border-radius: 10px !important;
    padding: 2rem 1rem !important;
    text-align: center !important;
    transition: border-color 0.2s;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {PRIMARY} !important;
}}
[data-testid="stFileUploader"] section > div {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
}}
[data-testid="stFileUploaderDropzoneInstructions"] svg {{
    display: none;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    font-size: 0.85rem;
    color: #888;
}}
.image-zone {{
    border: 2px dashed {BORDER};
    border-radius: 10px;
    padding: 8px;
    text-align: center;
}}
.image-zone img {{
    border-radius: 6px;
    max-width: 100%;
    max-height: 420px;
    object-fit: contain;
}}
.prob-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.9rem;
    margin-bottom: 2px;
}}
.prob-val {{
    font-weight: 600;
    min-width: 38px;
    text-align: right;
}}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "scroll_to_upload" not in st.session_state:
    st.session_state.scroll_to_upload = False
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "scores" not in st.session_state:
    st.session_state.scores = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ThoraxNet")
    st.caption("Apoyo a la lectura · v0.4 (beta)")
    st.divider()
    st.markdown("**Modelo**")
    st.markdown("""
<small>
• <b>Red:</b> DenseNet-121<br>
• <b>Optimizador:</b> AdamW<br>
• <b>Pérdida:</b> Focal Loss<br>
• <b>Archivo:</b> densenet_focal_moderate.pth
</small>
""", unsafe_allow_html=True)

# ── Header + Stepper ──────────────────────────────────────────────────────────

st.title("Análisis de imagen · Radiografías")

if st.session_state.img_bytes is None:
    _step = 1
elif not st.session_state.analyzed:
    _step = 2
else:
    _step = 3


def _circle(n: int) -> str:
    if n < _step:
        return (
            f'<div style="width:36px;height:36px;border-radius:50%;background:{PRIMARY};'
            f'color:white;display:flex;align-items:center;justify-content:center;'
            f'font-weight:700;font-size:0.9rem;opacity:0.45;">✓</div>'
        )
    if n == _step:
        return (
            f'<div style="width:36px;height:36px;border-radius:50%;background:{PRIMARY};'
            f'color:white;display:flex;align-items:center;justify-content:center;'
            f'font-weight:700;font-size:1rem;">{n}</div>'
        )
    return (
        f'<div style="width:36px;height:36px;border-radius:50%;background:{C_FUTURE_BG};'
        f'color:{C_FUTURE_FG};display:flex;align-items:center;justify-content:center;'
        f'font-weight:700;font-size:1rem;">{n}</div>'
    )


def _label(text: str, n: int) -> str:
    if n == _step:
        color, weight = T_LABEL, "600"
    elif n < _step:
        color, weight = T_DONE, "400"
    else:
        color, weight = T_FUTURE, "400"
    return (
        f'<div style="margin-top:8px;text-align:center;font-size:0.82rem;'
        f'color:{color};font-weight:{weight};padding:0 6px;line-height:1.4;">{text}</div>'
    )


def _line(after_step: int) -> str:
    color = PRIMARY if _step > after_step else LINE_OFF
    return f'<div style="flex:1;height:2px;background:{color};margin-top:18px;"></div>'


st.markdown(f"""
<div style="display:flex;align-items:flex-start;margin:1.5rem 0 2rem;">
  <div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">
    {_circle(1)}
    {_label("Subí o elegí<br>una radiografía", 1)}
  </div>
  {_line(1)}
  <div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">
    {_circle(2)}
    {_label("Hacé clic en<br><b>Analizar imagen</b>", 2)}
  </div>
  {_line(2)}
  <div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">
    {_circle(3)}
    {_label("Revisá las<br>predicciones", 3)}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Radiografía (ancho completo) ──────────────────────────────────────────────

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

st.markdown("**Radiografía**")

if st.session_state.img_bytes is None:
    f = st.file_uploader(
        label="Arrastrá una radiografía o hacé clic para buscar",
        type=["png", "jpg", "jpeg"],
    )
    if f:
        st.session_state.img_bytes = f.read()
        st.session_state.analyzed = False
        st.session_state.scores = None
        st.rerun()
    image = None
    analyze = False
else:
    image = Image.open(io.BytesIO(st.session_state.img_bytes))
    b64 = img_to_b64(image)
    st.markdown(
        f'<div class="image-zone"><img src="data:image/png;base64,{b64}"/></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    col_btn, col_change = st.columns([6, 1])
    with col_btn:
        analyze = st.button("Analizar imagen", type="primary", use_container_width=True)
    with col_change:
        if st.button("✕", use_container_width=True, help="Cambiar imagen"):
            st.session_state.img_bytes = None
            st.session_state.analyzed = False
            st.session_state.scores = None
            st.rerun()

# ── Predicciones (debajo, ancho completo) ─────────────────────────────────────

if image is not None and analyze:
    model = load_model()
    st.session_state.scores = predict(model, image)
    st.session_state.analyzed = True

if st.session_state.scores is not None:
    st.divider()
    st.markdown("**Predicciones del modelo**")
    st.caption("Estudio procesado · resultados orientativos")
    st.warning(
        "**Resultado orientativo, no es un diagnóstico.** "
        "Las cifras son probabilidades estimadas por el modelo y pueden contener errores. "
        "Confirme siempre con la lectura radiológica y el criterio clínico."
    )
    st.caption("13 hallazgos · ordenados de mayor a menor probabilidad")

    sorted_scores = sorted(st.session_state.scores.items(), key=lambda x: x[1], reverse=True)

    for label, prob in sorted_scores:
        detected = prob >= THRESHOLDS[label]
        opacity = "1" if detected else "0.35"
        badge = (
            f'<span style="color:{PRIMARY};font-weight:700;font-size:0.75rem;'
            f'background:{PRIMARY_LIGHT};padding:1px 6px;border-radius:4px;margin-left:6px;">DETECTADO</span>'
            if detected else ""
        )
        pct = int(prob * 100)
        thr_pct = THRESHOLDS[label] * 100
        bar_color = PRIMARY if detected else "#aaa"
        st.markdown(
            f'<div style="opacity:{opacity};margin-bottom:8px;">'
            f'  <div class="prob-row">'
            f'    <span>{LABEL_ES[label]}{badge}</span>'
            f'    <span class="prob-val">{prob:.2f}</span>'
            f'  </div>'
            f'  <div style="position:relative;background:{BAR_BG};border-radius:4px;height:6px;margin-top:3px;">'
            f'    <div style="width:{pct}%;background:{bar_color};height:6px;border-radius:4px;"></div>'
            f'    <div style="position:absolute;top:-3px;left:{thr_pct:.1f}%;'
            f'width:2px;height:12px;background:{THR_COLOR};border-radius:1px;" '
            f'title="Umbral: {THRESHOLDS[label]:.2f}"></div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption("El color no representa gravedad. Solo la longitud de la barra refleja la probabilidad estimada (0–1).")

elif image is not None:
    st.caption("Presioná **Analizar imagen** para obtener predicciones")

# ── Ejemplos de prueba ────────────────────────────────────────────────────────

st.divider()


@st.fragment
def examples_section():
    st.markdown("#### Ejemplos por patología")
    st.caption("Seleccioná una patología y cargá una imagen de ejemplo directamente.")

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
                if st.button("Cargar imagen", key=str(img_path), use_container_width=True):
                    with open(img_path, "rb") as f:
                        st.session_state.img_bytes = f.read()
                    st.session_state.analyzed = False
                    st.session_state.scores = None
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


examples_section()
