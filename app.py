import atexit
import os
from pathlib import Path
from textwrap import dedent
import tempfile

import streamlit as st
import pandas as pd

from report import generate_report
from processor import (
    DEFAULT_INFERENCE_SIZE,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_ZONES_X,
    DEFAULT_ZONES_Y,
    extract_drone_metadata,
    load_model,
    process_image,
    process_video,
)

st.set_page_config(page_title="Veridic", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def get_cached_model(weights_path):
    return load_model(weights_path)


_tmp_files: list[str] = []


def _cleanup_tmp():
    for p in _tmp_files:
        try:
            os.unlink(p)
        except OSError:
            pass


atexit.register(_cleanup_tmp)


def save_upload(uploaded_file, suffix=None):
    suffix = suffix or Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    _tmp_files.append(tmp.name)
    return tmp.name


def tmp_path(suffix):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    t.close()
    _tmp_files.append(t.name)
    return t.name


def render_html(markup):
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def _agreement(a, b):
    if not a or not b:
        return None, "—"
    pct = (1 - abs(a - b) / max(a, b)) * 100
    if pct >= 80:
        color = "#22f06b"
    elif pct >= 50:
        color = "#f0c422"
    else:
        color = "#ff385c"
    return color, f"{pct:.0f}%"


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

# ── presets de evento ────────────────────────────────────────────────────────
_PRESETS = {
    "🎪 Show / Festa junina": dict(sample_interval=8, altitude=50, fov=82.1),
    "✊ Protesto / Marcha":    dict(sample_interval=15, altitude=50, fov=82.1),
    "⛪ Concentração estática": dict(sample_interval=20, altitude=40, fov=82.1),
    "⚙️ Personalizado":         dict(sample_interval=15, altitude=40, fov=82.1),
}

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuração")

    preset_key = st.selectbox("Tipo de evento", list(_PRESETS.keys()), index=0)
    preset = _PRESETS[preset_key]

    weights_path = st.text_input("Pesos do modelo", value=DEFAULT_WEIGHTS_PATH)
    max_size = st.selectbox(
        "Tamanho de inferência",
        options=(640, 800, 1024, 1280),
        index=2,
        help="Menor = mais rápido. Maior = mais preciso.",
    )
    sample_interval = st.slider(
        "Amostrar a cada N frames (vídeo)",
        min_value=1, max_value=60, value=preset["sample_interval"],
        help="1 = todo frame (lento). 15 = a cada 0.5s a 30fps.",
    )
    st.markdown("---")
    st.subheader("Câmera (Jacobs)")
    camera_altitude = st.number_input(
        "Altitude do drone (m)", min_value=5, max_value=300, value=preset["altitude"],
        help="Posicione o drone sobre o centro do venue. A área coberta pelo frame será usada como área do evento.",
    )
    camera_fov = st.number_input(
        "FOV horizontal (°)", min_value=40.0, max_value=150.0, value=preset["fov"],
        help="DJI Mini 4 Pro = 82.1°.",
    )

    gsd_w = round(2 * camera_altitude * __import__('math').tan(__import__('math').radians(camera_fov / 2)), 1)
    gsd_h = round(gsd_w * 9 / 16, 1)
    auto_area = round(gsd_w * gsd_h)
    st.caption(f"📐 Cobertura estimada a {camera_altitude}m: **{gsd_w}m × {gsd_h}m ≈ {auto_area:,} m²**")

    known_area = st.number_input(
        "Área real do venue (m²)",
        min_value=0, max_value=500_000, value=0, step=100,
        help="Deixe 0 para usar a cobertura calculada pelo drone. Preencha quando souber a área exata — aumenta a precisão do Jacobs.",
    )
    if known_area > 0:
        st.caption(f"✅ Usando área informada: **{int(known_area):,} m²**")
    else:
        st.caption(f"🔍 Usando área detectada pelo drone: **{auto_area:,} m²**")

    _density_opts = {
        "Auto (CSRNet)":        None,
        "espaçada — 1 p/m²":   1.0,
        "moderada — 2 p/m²":   2.0,
        "densa — 3 p/m²":      3.0,
        "aglomerada — 4 p/m²": 4.0,
        "comprimida — 5 p/m²": 5.0,
    }
    global_density_sel = st.selectbox(
        "Densidade observada (geral)",
        list(_density_opts.keys()), index=0,
        help="Observador define a densidade predominante do evento. Sobrescreve o CSRNet no Jacobs.",
    )
    global_density = _density_opts[global_density_sel]
    if global_density:
        area_ref = int(known_area) if known_area > 0 else auto_area
        st.caption(f"→ Jacobs estimado: **{int(area_ref * global_density):,} pessoas**")
    st.markdown("---")
    st.subheader("Visualização")
    show_contours = st.checkbox("Mostrar regiões contadas", value=True,
                                help="Contorna as regiões do frame onde a contagem está ocorrendo.")
    if show_contours:
        contour_percentile = st.slider(
            "Sensibilidade",
            min_value=10, max_value=90, value=60, step=5,
            help="Menor = mostra regiões mais fracas também. Maior = só regiões de alta densidade.",
        )
    else:
        contour_percentile = 60
    st.subheader("Zonas")
    show_zones = st.checkbox("Dividir em zonas", value=False)
    if show_zones:
        _zc1, _zc2 = st.columns(2)
        zones_x = int(_zc1.number_input("Colunas", min_value=2, max_value=6, value=DEFAULT_ZONES_X))
        zones_y = int(_zc2.number_input("Linhas",  min_value=2, max_value=6, value=DEFAULT_ZONES_Y))

        _density_options = {
            "Auto (CSRNet)": None,
            "espaçada — 1 p/m²": 1.0,
            "moderada — 2 p/m²": 2.0,
            "densa — 3 p/m²": 3.0,
            "aglomerada — 4 p/m²": 4.0,
            "comprimida — 5 p/m²": 5.0,
        }
        st.caption("**Densidade por zona** — observador pode sobrescrever:")
        manual_densities = {}
        for r in range(zones_y):
            cols_ui = st.columns(zones_x)
            for c in range(zones_x):
                sel = cols_ui[c].selectbox(
                    f"Z{r+1},{c+1}", list(_density_options.keys()),
                    index=0, label_visibility="visible", key=f"zone_{r}_{c}"
                )
                if _density_options[sel] is not None:
                    manual_densities[(r, c)] = _density_options[sel]
    else:
        zones_x, zones_y = DEFAULT_ZONES_X, DEFAULT_ZONES_Y
        manual_densities = {}

    # densidade global sobrescreve todas as zonas
    if global_density:
        for r in range(zones_y):
            for c in range(zones_x):
                manual_densities[(r, c)] = global_density

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

@st.cache_data(show_spinner=False)
def _cached_drone_meta(file_id, file_bytes, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()
    meta = extract_drone_metadata(tmp.name)
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return meta

# extrai metadados do drone assim que o arquivo é carregado
if uploaded is not None and uploaded.name.lower().endswith(("mp4", "avi", "mov")):
    _drone_meta = _cached_drone_meta(
        uploaded.file_id, uploaded.getvalue(), Path(uploaded.name).suffix
    )
    if _drone_meta:
        if "altitude_m" in _drone_meta:
            st.sidebar.success(f"🛸 Altitude detectada: **{_drone_meta['altitude_m']} m**")
        if "fov_deg" in _drone_meta:
            st.sidebar.success(f"📷 FOV detectado: **{_drone_meta['fov_deg']}°**")
    else:
        st.sidebar.caption("ℹ️ Telemetria não encontrada no vídeo — use os valores manuais.")

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
            show_contours=show_contours,
            contour_percentile=contour_percentile,
            max_size=max_size,
            sample_interval=sample_interval,
            camera_altitude=float(camera_altitude),
            camera_fov=float(camera_fov),
            known_area_m2=float(known_area),
            manual_densities=manual_densities,
            progress_callback=on_progress,
            preview_callback=on_preview,
        )
        progress_bar.progress(1.0)

        with metrics_ph.container():
            j = result.get("jacobs") or {}
            jcount = j.get("jacobs_count")
            agr_color, agr_pct = _agreement(result['peak_count'], jcount)
            dc  = j.get('density_class', '—')
            fac = j.get('density_factor', '')

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{result['peak_count']:,}</div>
                    <div class="metric-label">Pico · CSRNet</div>
                </div>""")
            with m2:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{result['avg_count']:,}</div>
                    <div class="metric-label">Média · CSRNet</div>
                </div>""")
            with m3:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:#22f06b">{jcount:,}</div>
                    <div class="metric-label">Estimativa · Jacobs</div>
                </div>""")
            with m4:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value" style="font-size:1.3rem">{dc}</div>
                    <div class="metric-label">{fac} p/m² · densidade</div>
                </div>""")
            with m5:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{j.get('crowd_area_m2', '—')} m²</div>
                    <div class="metric-label">Área detectada</div>
                </div>""")
            with m6:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:{agr_color or '#fff'}">{agr_pct}</div>
                    <div class="metric-label">Concordância CSRNet↔Jacobs</div>
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

            j = result.get("jacobs") or {}
            _, agr_pct_vid = _agreement(result["peak_count"], j.get("jacobs_count"))
            _ann = cv2.VideoCapture(output_path)
            _ok, _frame = _ann.read()
            _ann.release()
            _ann_frame = _frame if _ok else np.zeros((360, 640, 3), dtype=np.uint8)
            _heat_frame = cv2.imread(heatmap_path) or np.zeros((360, 640, 3), dtype=np.uint8)
            pdf_bytes = generate_report(
                event_name=uploaded.name,
                annotated_bgr=_ann_frame,
                heatmap_bgr=_heat_frame,
                csrnet_count=result["peak_count"],
                jacobs_count=j.get("jacobs_count", 0),
                crowd_area_m2=j.get("crowd_area_m2", 0),
                density_class=j.get("density_class", "—"),
                density_factor=j.get("density_factor", 0),
                agreement_pct=agr_pct_vid,
                sectors=j.get("sectors"),
                is_video=True,
                peak_count=result["peak_count"],
                avg_count=result["avg_count"],
            )
            st.download_button(
                "📄 Baixar relatório PDF",
                data=pdf_bytes,
                file_name="veridic_relatorio.pdf",
                mime="application/pdf",
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
                show_contours=show_contours,
                contour_percentile=contour_percentile,
                max_size=max_size,
                camera_altitude=float(camera_altitude),
                camera_fov=float(camera_fov),
                known_area_m2=float(known_area),
                manual_densities=manual_densities,
            )

        with left_view.container():
            st.image(result["annotated"][:, :, ::-1], channels="RGB", use_container_width=True)
        with right_view.container():
            st.image(result["heatmap"][:, :, ::-1], channels="RGB", use_container_width=True)

        with metrics_ph.container():
            j = result.get("jacobs") or {}
            jcount = j.get("jacobs_count")
            agr_color, agr_pct = _agreement(result['count'], jcount)
            dc  = j.get('density_class', '—')
            fac = j.get('density_factor', '')

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{result['count']:,}</div>
                    <div class="metric-label">Estimativa · CSRNet</div>
                </div>""")
            with m2:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:#22f06b">{jcount:,}</div>
                    <div class="metric-label">Estimativa · Jacobs</div>
                </div>""")
            with m3:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value" style="font-size:1.3rem">{dc}</div>
                    <div class="metric-label">{fac} p/m² · densidade</div>
                </div>""")
            with m4:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value">{j.get('crowd_area_m2', '—')} m²</div>
                    <div class="metric-label">Área detectada</div>
                </div>""")
            with m5:
                render_html(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:{agr_color or '#fff'}">{agr_pct}</div>
                    <div class="metric-label">Concordância CSRNet↔Jacobs</div>
                </div>""")

            sectors = j.get("sectors", [])
            if len(sectors) > 1:
                st.markdown("**Densidade por setor (Jacobs)**")
                rows = {}
                for s in sectors:
                    rows.setdefault(s["row"], []).append(s)
                for row_sectors in rows.values():
                    cols = st.columns(len(row_sectors))
                    for col_widget, s in zip(cols, row_sectors):
                        with col_widget:
                            render_html(f"""
                            <div class="metric-box">
                                <div class="metric-value" style="font-size:1.1rem">{s['jacobs_count']:,}</div>
                                <div class="metric-label">{s['density_class']} · {s['density_factor']} p/m²</div>
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

            j = result.get("jacobs") or {}
            _, agr_pct_img = _agreement(result["count"], j.get("jacobs_count"))
            pdf_bytes = generate_report(
                event_name=uploaded.name,
                annotated_bgr=result["annotated"],
                heatmap_bgr=result["heatmap"],
                csrnet_count=result["count"],
                jacobs_count=j.get("jacobs_count", 0),
                crowd_area_m2=j.get("crowd_area_m2", 0),
                density_class=j.get("density_class", "—"),
                density_factor=j.get("density_factor", 0),
                agreement_pct=agr_pct_img,
                sectors=j.get("sectors"),
                is_video=False,
            )
            st.download_button(
                "📄 Baixar relatório PDF",
                data=pdf_bytes,
                file_name="veridic_relatorio.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
