import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

from processor import extract_drone_metadata, process_video, process_image
from report import generate_report
from utils.helpers import save_upload, tmp_path, compute_agreement, render_metric_box, get_cached_model


def render_tab_csrnet(config):
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        with st.container(key="left_panel"):
            left_view = st.empty()
            left_view.markdown("*Faça upload de uma imagem ou vídeo para começar.*")

    with right_col:
        with st.container(key="right_panel"):
            right_view = st.empty()
            right_view.markdown("*O mapa de densidade aparecerá aqui.*")

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
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()
        meta = extract_drone_metadata(tmp.name)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return meta

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

    if process_clicked and uploaded is not None:
        if not Path(config["weights_path"]).exists():
            st.error(f"Pesos não encontrados: {config['weights_path']}\nBaixe em: https://github.com/leeyeehoo/CSRNet-pytorch — coloque em weights/csrnet_sha.pth")
            st.stop()

        model = get_cached_model(config["weights_path"])
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
                show_zones=config["show_zones"],
                zones_x=config["zones_x"],
                zones_y=config["zones_y"],
                show_contours=config["show_contours"],
                contour_percentile=config["contour_percentile"],
                max_size=config["max_size"],
                sample_interval=config["sample_interval"],
                camera_altitude=float(config["camera_altitude"]),
                camera_fov=float(config["camera_fov"]),
                known_area_m2=float(config["known_area"]),
                manual_densities=config["manual_densities"],
                heatmap_mode="jacobs" if config.get("heatmap_mode") == "Jacobs por setor" else "csrnet",
                progress_callback=on_progress,
                preview_callback=on_preview,
            )
            progress_bar.progress(1.0)

            with metrics_ph.container():
                j = result.get("jacobs") or {}
                jcount = j.get("jacobs_count")
                agr_color, agr_pct = compute_agreement(result['peak_count'], jcount)
                dc  = j.get('density_class', '—')
                fac = j.get('density_factor', '')

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                with m1:
                    render_metric_box(f"{result['peak_count']:,}", "Pico · CSRNet")
                with m2:
                    render_metric_box(f"{result['avg_count']:,}", "Média · CSRNet")
                with m3:
                    render_metric_box(f"{jcount:,}", "Estimativa · Jacobs", value_color="#22f06b")
                with m4:
                    render_metric_box(dc, f"{fac} p/m² · densidade", value_size="1.3rem")
                with m5:
                    render_metric_box(f"{j.get('crowd_area_m2', '—')} m²", "Área detectada")
                with m6:
                    render_metric_box(agr_pct, "Concordância CSRNet↔Jacobs", value_color=agr_color or "#fff")

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
                _, agr_pct_vid = compute_agreement(result["peak_count"], j.get("jacobs_count"))

                _cap = cv2.VideoCapture(output_path)
                _ok, _frame = _cap.read()
                _cap.release()
                _ann_frame = _frame if _ok else np.zeros((360, 640, 3), dtype=np.uint8)

                _heat_read = cv2.imread(heatmap_path)
                _heat_frame = _heat_read if _heat_read is not None else np.zeros((360, 640, 3), dtype=np.uint8)

                pdf_bytes = generate_report(
                    event_name=config["event_name"],
                    annotated_bgr=_ann_frame,
                    heatmap_bgr=_heat_frame,
                    csrnet_count=result["peak_count"],
                    jacobs_count=j.get("jacobs_count", 0),
                    crowd_area_m2=j.get("crowd_area_m2", 0),
                    density_class=j.get("density_class", "—"),
                    density_factor=j.get("density_factor", 0),
                    density_desc=j.get("density_desc", ""),
                    agreement_pct=agr_pct_vid,
                    camera_altitude=float(config["camera_altitude"]),
                    camera_fov=float(config["camera_fov"]),
                    sectors=j.get("sectors"),
                    is_video=True,
                    peak_count=result["peak_count"],
                    avg_count=result["avg_count"],
                    timeline=result.get("timeline"),
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
                    show_zones=config["show_zones"],
                    zones_x=config["zones_x"],
                    zones_y=config["zones_y"],
                    show_contours=config["show_contours"],
                    contour_percentile=config["contour_percentile"],
                    max_size=config["max_size"],
                    camera_altitude=float(config["camera_altitude"]),
                    camera_fov=float(config["camera_fov"]),
                    known_area_m2=float(config["known_area"]),
                    manual_densities=config["manual_densities"],
                    heatmap_mode="jacobs" if config.get("heatmap_mode") == "Jacobs por setor" else "csrnet",
                )

            with left_view.container():
                st.image(result["annotated"][:, :, ::-1], channels="RGB", use_container_width=True)
            with right_view.container():
                st.image(result["heatmap"][:, :, ::-1], channels="RGB", use_container_width=True)

            with metrics_ph.container():
                j = result.get("jacobs") or {}
                jcount = j.get("jacobs_count")
                agr_color, agr_pct = compute_agreement(result['count'], jcount)
                dc  = j.get('density_class', '—')
                fac = j.get('density_factor', '')

                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    render_metric_box(f"{result['count']:,}", "Estimativa · CSRNet")
                with m2:
                    render_metric_box(f"{jcount:,}", "Estimativa · Jacobs", value_color="#22f06b")
                with m3:
                    render_metric_box(dc, f"{fac} p/m² · densidade", value_size="1.3rem")
                with m4:
                    render_metric_box(f"{j.get('crowd_area_m2', '—')} m²", "Área detectada")
                with m5:
                    render_metric_box(agr_pct, "Concordância CSRNet↔Jacobs", value_color=agr_color or "#fff")

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
                                render_metric_box(f"{s['jacobs_count']:,}", f"{s['density_class']} · {s['density_factor']} p/m²", value_size="1.1rem")

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
                _, agr_pct_img = compute_agreement(result["count"], j.get("jacobs_count"))
                pdf_bytes = generate_report(
                    event_name=config["event_name"],
                    annotated_bgr=result["annotated"],
                    heatmap_bgr=result["heatmap"],
                    csrnet_count=result["count"],
                    jacobs_count=j.get("jacobs_count", 0),
                    crowd_area_m2=j.get("crowd_area_m2", 0),
                    density_class=j.get("density_class", "—"),
                    density_factor=j.get("density_factor", 0),
                    density_desc=j.get("density_desc", ""),
                    agreement_pct=agr_pct_img,
                    camera_altitude=float(config["camera_altitude"]),
                    camera_fov=float(config["camera_fov"]),
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
