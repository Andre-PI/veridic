from pathlib import Path
from textwrap import dedent
import tempfile

import streamlit as st
import pandas as pd

from processor import (
    DEFAULT_INFERENCE_SIZE,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_ZONES_X,
    DEFAULT_ZONES_Y,
    load_model,
    process_image,
    process_video,
)

st.set_page_config(page_title="Veridic", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def get_cached_model(weights_path):
    return load_model(weights_path)


def save_upload(uploaded_file, suffix=None):
    suffix = suffix or Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def tmp_path(suffix):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    t.close()
    return t.name


def render_html(markup):
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


render_html("""
<style>
    :root {
        --bg: #050912;
        --panel: rgba(12, 18, 31, 0.82);
        --line: rgba(106, 132, 173, 0.28);
        --text: #f5f8ff;
        --muted: #93a4bd;
        --green: #22f06b;
        --blue: #2387ff;
        --red: #ff385c;
    }
    .stApp {
        background:
            radial-gradient(circle at 22% 16%, rgba(28,119,255,0.18), transparent 30%),
            radial-gradient(circle at 80% 0%, rgba(34,240,107,0.12), transparent 28%),
            linear-gradient(135deg, #03060d 0%, #08111f 48%, #03060a 100%);
        color: var(--text);
    }
    .block-container { max-width: 1920px; padding: 1.1rem 2rem 1.4rem; }
    [data-testid="stHeader"] { background: transparent; }

    .top-header {
        display: flex; align-items: center; justify-content: space-between;
        min-height: 72px; padding: 0 1.25rem; margin-bottom: 1rem;
        border: 1px solid var(--line); border-radius: 8px;
        background: linear-gradient(90deg, rgba(9,15,28,0.94), rgba(14,32,58,0.88), rgba(6,12,22,0.94));
        box-shadow: 0 18px 50px rgba(0,0,0,0.35); backdrop-filter: blur(18px);
    }
    .brand-title { font-size: 1.55rem; font-weight: 800; color: var(--text); }
    .status-dot {
        width: 12px; height: 12px; border-radius: 50%;
        background: var(--green); box-shadow: 0 0 16px rgba(34,240,107,0.9);
        display: inline-block; margin-right: 6px;
    }
    .disclaimer {
        font-size: 0.78rem; color: var(--muted);
        padding: 0.5rem 1rem; border: 1px solid var(--line);
        border-radius: 6px; margin-top: 0.5rem;
    }
    .st-key-left_panel, .st-key-right_panel {
        min-height: 520px; padding: 0.75rem;
        border: 1px solid var(--line); border-radius: 8px;
        background: var(--panel);
        box-shadow: 0 26px 70px rgba(0,0,0,0.42), inset 0 0 0 1px rgba(255,255,255,0.05);
        backdrop-filter: blur(18px); overflow: hidden;
    }
    .stButton > button {
        border-radius: 8px; border: 1px solid rgba(83,139,255,0.55);
        background: linear-gradient(180deg, #246bff, #123c9c);
        color: white; font-weight: 900; min-height: 44px;
    }
    .stDownloadButton > button {
        border-radius: 8px; border: 1px solid rgba(34,240,107,0.5);
        background: rgba(34,240,107,0.12); color: #dcffe6; font-weight: 800;
    }
    .metric-box {
        padding: 1rem 1.25rem; border: 1px solid var(--line); border-radius: 8px;
        background: rgba(8,13,24,0.9); text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 900; color: #fff; }
    .metric-label { font-size: 0.82rem; color: var(--muted); font-weight: 700; margin-top: 0.2rem; }
</style>
""")

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuração")
    weights_path = st.text_input("Pesos do modelo", value=DEFAULT_WEIGHTS_PATH)
    max_size = st.selectbox(
        "Tamanho de inferência",
        options=(640, 800, 1024, 1280),
        index=2,
        help="Menor = mais rápido. Maior = mais preciso.",
    )
    sample_interval = st.slider(
        "Amostrar a cada N frames (vídeo)",
        min_value=1, max_value=60, value=15,
        help="1 = todo frame (lento). 15 = a cada 0.5s a 30fps.",
    )
    st.markdown("---")
    st.subheader("Zonas")
    show_zones = st.checkbox("Dividir em zonas", value=False)
    if show_zones:
        _zc1, _zc2 = st.columns(2)
        zones_x = int(_zc1.number_input("Colunas", min_value=2, max_value=6, value=DEFAULT_ZONES_X))
        zones_y = int(_zc2.number_input("Linhas",  min_value=2, max_value=6, value=DEFAULT_ZONES_Y))
    else:
        zones_x, zones_y = DEFAULT_ZONES_X, DEFAULT_ZONES_Y

# ── header ────────────────────────────────────────────────────────────────────
render_html("""
<div class="top-header">
    <div style="display:flex;align-items:center;gap:1rem;">
        <div class="brand-title">Veridic</div>
        <div style="color:#dfffea;font-weight:700;">
            <span class="status-dot"></span>Estimativa de público
        </div>
    </div>
</div>
""")

render_html("""
<div class="disclaimer">
    ⚠️ A estimativa é feita com base em imagens aéreas, segmentação da área e densidade aproximada por setor.
    O resultado não substitui catraca, ingresso ou sistema oficial de controle de acesso.
</div>
""")

st.markdown("")

# ── layout principal ──────────────────────────────────────────────────────────
left_col, right_col = st.columns(2, gap="large")

with left_col:
    with st.container(key="left_panel"):
        left_view = st.empty()
        left_view.markdown("*Faça upload de uma imagem ou vídeo para começar.*")

with right_col:
    with st.container(key="right_panel"):
        right_view = st.empty()
        right_view.markdown("*O mapa de densidade aparecerá aqui.*")

# ── toolbar ───────────────────────────────────────────────────────────────────
st.markdown("")
upload_col, btn_col = st.columns([3, 1])

with upload_col:
    uploaded = st.file_uploader(
        "Imagem ou vídeo",
        type=("jpg", "jpeg", "png", "mp4", "avi", "mov"),
        label_visibility="collapsed",
    )

with btn_col:
    process_clicked = st.button("Analisar", type="primary", disabled=uploaded is None, use_container_width=True)

progress_ph = st.empty()
metrics_ph  = st.empty()
timeline_ph = st.empty()
download_ph = st.empty()

# ── processamento ─────────────────────────────────────────────────────────────
if process_clicked and uploaded is not None:
    if not Path(weights_path).exists():
        st.error(f"Pesos não encontrados: {weights_path}\nBaixe em: https://github.com/leeyeehoo/CSRNet-pytorch — coloque em weights/csrnet_sha.pth")
        st.stop()

    model = get_cached_model(weights_path)
    is_video = uploaded.name.lower().endswith(("mp4", "avi", "mov"))
    suffix = Path(uploaded.name).suffix
    input_path  = save_upload(uploaded, suffix)
    output_path = tmp_path(suffix if is_video else ".jpg")
    heatmap_path = tmp_path(".jpg")

    if is_video:
        progress_bar = progress_ph.progress(0)

        def on_progress(v):
            progress_bar.progress(v)

        def on_preview(frame, count, heatmap):
            rgb_frame   = frame[:, :, ::-1]
            rgb_heatmap = heatmap[:, :, ::-1]
            with left_view.container():
                st.image(rgb_frame, channels="RGB", use_container_width=True)
            with right_view.container():
                st.image(rgb_heatmap, channels="RGB", use_container_width=True)

        result = process_video(
            video_path=input_path,
            model=model,
            output_path=output_path,
            heatmap_path=heatmap_path,
            show_zones=show_zones,
            zones_x=zones_x,
            zones_y=zones_y,
            max_size=max_size,
            sample_interval=sample_interval,
            progress_callback=on_progress,
            preview_callback=on_preview,
        )
        progress_bar.progress(1.0)

        with metrics_ph.container():
            m1, m2, m3 = st.columns(3)
            with m1:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{result['peak_count']:,}</div>
                    <div class="metric-label">Pico de público</div>
                </div>""")
            with m2:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{result['avg_count']:,}</div>
                    <div class="metric-label">Média durante o evento</div>
                </div>""")
            with m3:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{result['frames_sampled']}</div>
                    <div class="metric-label">Amostras processadas</div>
                </div>""")

        if result["timeline"]:
            df = pd.DataFrame(result["timeline"])
            df = df.rename(columns={"time_s": "Tempo (s)", "count": "Pessoas estimadas"})
            timeline_ph.line_chart(df.set_index("Tempo (s)")["Pessoas estimadas"])

        with download_ph.container():
            dl1, dl2 = st.columns(2)
            with dl1:
                if Path(output_path).exists():
                    st.download_button(
                        "Download vídeo anotado",
                        data=Path(output_path).read_bytes(),
                        file_name="veridic_resultado.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            with dl2:
                if Path(heatmap_path).exists():
                    st.download_button(
                        "Download mapa de densidade",
                        data=Path(heatmap_path).read_bytes(),
                        file_name="veridic_heatmap.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )
    else:
        with st.spinner("Processando..."):
            result = process_image(
                image_path=input_path,
                model=model,
                output_path=output_path,
                heatmap_path=heatmap_path,
                show_zones=show_zones,
                zones_x=zones_x,
                zones_y=zones_y,
                max_size=max_size,
            )

        with left_view.container():
            st.image(result["annotated"][:, :, ::-1], channels="RGB", use_container_width=True)
        with right_view.container():
            st.image(result["heatmap"][:, :, ::-1], channels="RGB", use_container_width=True)

        with metrics_ph.container():
            m1, m2 = st.columns(2)
            with m1:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{result['count']:,}</div>
                    <div class="metric-label">Pessoas estimadas</div>
                </div>""")
            if result["zone_counts"]:
                with m2:
                    top_zone = max(result["zone_counts"], key=result["zone_counts"].get)
                    render_html(f"""
                    <div class="metric-box">
                        <div class="metric-value">{result['zone_counts'][top_zone]:,}</div>
                        <div class="metric-label">Zona mais densa</div>
                    </div>""")

        with download_ph.container():
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download imagem anotada",
                    data=Path(output_path).read_bytes(),
                    file_name="veridic_resultado.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "Download mapa de densidade",
                    data=Path(heatmap_path).read_bytes(),
                    file_name="veridic_heatmap.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
