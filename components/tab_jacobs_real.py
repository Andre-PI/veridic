import math
import random
import hashlib
from datetime import date
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from jacobs_real import draw_jacobs_grid, jacobs_estimate
from report import generate_jacobs_report
from utils.helpers import render_html, render_metric_box

def render_tab_jacobs_real(config):
    render_html("""
    <div style="margin-bottom:0.75rem;">
        <span style="font-size:1.1rem;font-weight:800;">Método de Jacobs — contagem manual por grade</span><br>
        <span style="font-size:0.82rem;color:#93a4bd;">
            Faça upload de uma foto tirada pelo drone diretamente acima do público.
            O app sorteia aleatoriamente as células a contar antes da contagem acontecer —
            garantindo amostragem representativa e rastreabilidade via código de auditoria.
        </span>
    </div>
    """)

    j2_uploaded = st.file_uploader(
        "Foto do drone (imagem estática)",
        type=("jpg", "jpeg", "png"),
        key="jacobs_real_upload",
        label_visibility="collapsed",
    )

    if j2_uploaded is None:
        render_html("""
        <div style="text-align:center;padding:3rem;color:#93a4bd;border:1px dashed rgba(106,132,173,0.3);border-radius:8px;">
            📸 Faça upload de uma foto do drone acima do público para começar.
        </div>
        """)
        return

    raw_bytes = j2_uploaded.getvalue()
    file_bytes = np.frombuffer(raw_bytes, np.uint8)
    j2_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    _file_hash = hashlib.md5(raw_bytes).hexdigest()[:8]

    if j2_image is None:
        st.error("Não foi possível abrir a imagem.")
        return

    cfg1, cfg2, cfg3, cfg4, cfg5, cfg6 = st.columns([1, 1, 1, 1, 1, 2])
    with cfg1:
        j2_cols = int(st.number_input("Colunas", min_value=2, max_value=20, value=8, key="j2_cols"))
    with cfg2:
        j2_rows = int(st.number_input("Linhas", min_value=2, max_value=20, value=6, key="j2_rows"))
    with cfg3:
        j2_altitude = st.number_input("Altitude (m)", min_value=5, max_value=300, value=50, key="j2_alt")
    with cfg4:
        j2_fov = st.number_input("FOV (°)", min_value=40.0, max_value=150.0, value=82.1, key="j2_fov")
    with cfg5:
        j2_n_sample = int(st.number_input(
            "Células a sortear", min_value=1, max_value=100, value=10, key="j2_n",
            help="Quantas células serão sorteadas para contagem manual. Mínimo recomendado: 8.",
        ))
    with cfg6:
        j2_event = st.text_input("Nome do evento", value="Evento", key="j2_event_name")

    img_h, img_w = j2_image.shape[:2]
    ground_w = 2 * j2_altitude * math.tan(math.radians(j2_fov / 2))
    ground_h = ground_w * (img_h / img_w)
    cell_area_m2 = (ground_w / j2_cols) * (ground_h / j2_rows)
    total_cells = j2_rows * j2_cols

    st.caption(
        f"📐 Cobertura total: **{ground_w:.1f}m × {ground_h:.1f}m** — "
        f"cada célula ≈ **{cell_area_m2:.1f} m²** "
        f"({j2_rows}×{j2_cols} = {total_cells} células)"
    )
    st.warning(
        "A foto deve ser tirada com o drone **diretamente acima do público** (nadir). "
        "Em fotos inclinadas as células distantes cobrem áreas maiores no solo — "
        "a contagem fica enviesada e a extrapolação perde validade.",
        icon="⚠️",
    )

    st.markdown("---")

    _sk = f"j2_{_file_hash}_{j2_rows}_{j2_cols}"

    def _init(key, val):
        if key not in st.session_state:
            st.session_state[key] = val

    _init(f"{_sk}_excl", set())
    _init(f"{_sk}_excl_locked", False)
    _init(f"{_sk}_sampled", [])
    _init(f"{_sk}_seed", None)

    render_html("""
    <div style="font-size:0.95rem;font-weight:800;margin-bottom:0.5rem;">
        1 — Marque as células sem público
    </div>
    <div style="font-size:0.8rem;color:#93a4bd;margin-bottom:0.75rem;">
        Palco, área técnica, corredores, camarotes fechados. Essas células serão
        excluídas do sorteio e da extrapolação.
    </div>
    """)

    excl_locked = st.session_state[f"{_sk}_excl_locked"]

    if not excl_locked:
        st.info("👆 Desenhe na imagem abaixo para pintar as células vazias/excluídas (palco, área técnica...).", icon="🖌️")
        
        _g = draw_jacobs_grid(j2_image, j2_rows, j2_cols, st.session_state[f"{_sk}_sampled"], set())
        
        h_orig, w_orig = _g.shape[:2]
        scale = min(1.0, 800 / w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        _g_resized = cv2.resize(_g, (new_w, new_h))
        
        pil_img = Image.fromarray(_g_resized[:, :, ::-1])
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 56, 92, 0.4)",
            stroke_width=20,
            stroke_color="rgba(255, 56, 92, 0.6)",
            background_image=pil_img,
            update_streamlit=True,
            height=new_h,
            width=new_w,
            drawing_mode="freedraw",
            key=f"{_sk}_canvas",
        )
        
        if canvas_result.image_data is not None:
            alpha = canvas_result.image_data[:, :, 3]
            ch, cw = alpha.shape
            
            new_excl = set()
            for r in range(j2_rows):
                for c in range(j2_cols):
                    y1, y2 = r * ch // j2_rows, (r + 1) * ch // j2_rows
                    x1, x2 = c * cw // j2_cols, (c + 1) * cw // j2_cols
                    cell_alpha = alpha[y1:y2, x1:x2]
                    if np.any(cell_alpha > 0):
                        cell_num = r * j2_cols + c + 1
                        new_excl.add(cell_num)
            
            st.session_state[f"{_sk}_excl"] = new_excl

        if st.button("Confirmar exclusões e realizar sorteio", type="primary", use_container_width=True):
            crowd_pool = [i for i in range(1, total_cells + 1) if i not in st.session_state[f"{_sk}_excl"]]
            n = min(j2_n_sample, len(crowd_pool))
            seed = random.randint(100_000, 999_999)
            sampled = sorted(random.Random(seed).sample(crowd_pool, n))
            st.session_state[f"{_sk}_excl_locked"] = True
            st.session_state[f"{_sk}_sampled"] = sampled
            st.session_state[f"{_sk}_seed"] = seed
            st.rerun()
    else:
        excl_list = sorted(st.session_state[f"{_sk}_excl"])
        st.success(f"Excluídas: {excl_list if excl_list else 'nenhuma'} ({len(excl_list)} célula{'s' if len(excl_list) != 1 else ''})")
        if st.button("↩ Refazer exclusões", type="secondary", help="Atenção: isso apaga o sorteio atual e exige um novo."):
            st.session_state[f"{_sk}_excl_locked"] = False
            st.session_state[f"{_sk}_sampled"] = []
            st.session_state[f"{_sk}_seed"] = None
            st.rerun()
        
        _g = draw_jacobs_grid(j2_image, j2_rows, j2_cols, st.session_state[f"{_sk}_sampled"], st.session_state[f"{_sk}_excl"])
        st.image(_g[:, :, ::-1], channels="RGB", use_container_width=True)

    sampled = st.session_state[f"{_sk}_sampled"]
    seed    = st.session_state[f"{_sk}_seed"]

    if not sampled:
        return

    st.markdown("---")

    excluded_set   = st.session_state[f"{_sk}_excl"]
    excluded_count = len(excluded_set)
    crowd_cells    = total_cells - excluded_count

    audit_date  = date.today().strftime("%Y%m%d")
    audit_excl  = "-".join(str(x) for x in sorted(excluded_set)) or "0"
    audit_cells = "-".join(str(x) for x in sampled)
    audit_code  = f"VRD·{_file_hash}·{audit_date}·{j2_rows}x{j2_cols}·X{audit_excl}·S{seed}"

    render_html(f"""
    <div style="font-size:0.95rem;font-weight:800;margin-bottom:0.5rem;">
        2 — Células sorteadas para contagem
    </div>
    <div style="padding:1rem 1.25rem;border:1px solid rgba(34,240,107,0.35);border-radius:8px;background:rgba(8,22,14,0.7);margin-bottom:0.75rem;">
        <div style="font-size:0.78rem;color:#93a4bd;font-weight:700;margin-bottom:0.3rem;">
            CÉLULAS A CONTAR ({len(sampled)} de {crowd_cells} de público)
        </div>
        <div style="font-size:1.25rem;font-weight:900;color:#22f06b;letter-spacing:0.04em;">
            {" · ".join(str(c) for c in sampled)}
        </div>
        <div style="font-size:0.72rem;color:#93a4bd;margin-top:0.6rem;">
            Código de auditoria: <code style="color:#f5f8ff;">{audit_code}</code>
        </div>
    </div>
    <div style="font-size:0.8rem;color:#93a4bd;margin-bottom:0.5rem;">
        Fotografe ou anote o código antes de ir a campo. Qualquer pessoa com a mesma imagem, mesma grade e mesmo seed consegue verificar que o sorteio foi feito antes da contagem.
    </div>
    """)

    st.markdown("---")

    render_html("""
    <div style="font-size:0.95rem;font-weight:800;margin-bottom:0.5rem;">
        3 — Registre a contagem de cada célula sorteada
    </div>
    <div style="font-size:0.8rem;color:#93a4bd;margin-bottom:0.75rem;">
        Preencha apenas as células da lista acima. Não altere quais células contar após ver a imagem — isso invalida a aleatoriedade do sorteio.
    </div>
    """)

    count_col, img_col2 = st.columns([2, 3], gap="large")

    with count_col:
        _count_data = [{"Célula": c, "Contagem": pd.NA} for c in sampled]
        _count_df   = pd.DataFrame(_count_data).astype({"Contagem": pd.Int64Dtype()})

        edited_counts = st.data_editor(
            _count_df,
            key=f"j2_count_{_sk}_{seed}",
            column_config={
                "Célula": st.column_config.NumberColumn("Célula", disabled=True, width="small"),
                "Contagem": st.column_config.NumberColumn(
                    "Contagem", min_value=0, step=1, width="medium", help="Pessoas contadas visualmente nesta célula.",
                ),
            },
            hide_index=True,
            use_container_width=True,
            height=min(520, len(sampled) * 36 + 42),
        )

    with img_col2:
        _counted_set = set(edited_counts[edited_counts["Contagem"].notna()]["Célula"].tolist())
        _g2 = draw_jacobs_grid(j2_image, j2_rows, j2_cols, _counted_set, excluded_set)
        st.image(_g2[:, :, ::-1], channels="RGB", use_container_width=True)
        render_html(f"""
        <div style="font-size:0.75rem;color:#93a4bd;margin-top:0.3rem;">
            <span class="legend-dot" style="background:#22f06b;"></span>contada &nbsp;
            <span class="legend-dot" style="background:#3333cc;"></span>excluída &nbsp;
            <span style="color:#f5f8ff;">{len(_counted_set)}/{len(sampled)}</span> preenchidas
        </div>
        """)

    sampled_counts = [int(x) for x in edited_counts["Contagem"].dropna().tolist()]
    st.markdown("")

    if not sampled_counts:
        render_html("""
        <div style="padding:1rem;color:#93a4bd;border:1px dashed rgba(106,132,173,0.3);border-radius:8px;text-align:center;">
            Preencha a contagem de pelo menos uma célula para ver a estimativa.
        </div>
        """)
    else:
        est = jacobs_estimate(sampled_counts, excluded_count, total_cells)

        if est is None:
            st.warning("Todas as células estão excluídas — não há área de público para estimar.")
        else:
            st.markdown("---")
            render_html(f"""
            <div class="jacobs-result">
                <div style="font-size:0.85rem;color:#93a4bd;font-weight:700;margin-bottom:0.25rem;">
                    ESTIMATIVA JACOBS REAL &nbsp;·&nbsp;
                    <span style="font-weight:400;font-size:0.78rem;">{j2_event}</span>
                </div>
                <div class="jacobs-estimate">{est['estimate']:,}</div>
                <div class="jacobs-ci">
                    ± {est['margin']:,} pessoas &nbsp;·&nbsp;
                    intervalo [{est['lower']:,} — {est['upper']:,}]
                </div>
                <div style="font-size:0.72rem;color:#93a4bd;margin-top:0.6rem;">
                    Auditoria: <code style="color:#f5f8ff;">{audit_code}</code>
                </div>
            </div>
            """)

            st.markdown("")

            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                render_metric_box(est['sampled_cells'], "Células contadas", value_size="1.6rem")
            with s2:
                render_metric_box(est['crowd_cells'], "Células de público", value_size="1.6rem")
            with s3:
                render_metric_box(f"{est['coverage_pct']}%", "Cobertura da amostra", value_size="1.6rem")
            with s4:
                render_metric_box(est['avg_per_cell'], "Média por célula", value_size="1.6rem")
            with s5:
                std_display = est['std_per_cell'] if est['std_per_cell'] is not None else "—"
                render_metric_box(std_display, "Desvio padrão / célula", value_size="1.6rem")

            if est["sampled_cells"] < 5:
                st.warning(
                    f"Amostra pequena ({est['sampled_cells']} célula{'s' if est['sampled_cells'] > 1 else ''}). "
                    "Recomenda-se sortear ao menos 8 células para reduzir a margem."
                )
            elif est["coverage_pct"] >= 30:
                st.success(
                    f"Boa cobertura ({est['coverage_pct']}% das células de público). Estimativa com alta confiabilidade."
                )

            if cell_area_m2 > 0 and est["avg_per_cell"] > 0:
                density = est["avg_per_cell"] / cell_area_m2
                st.caption(f"Densidade implícita: **{density:.2f} p/m²** ({est['avg_per_cell']:.1f} pessoas / {cell_area_m2:.1f} m² por célula)")

            st.markdown("")
            
            _cell_counts = {
                int(row["Célula"]): int(row["Contagem"])
                for _, row in edited_counts.iterrows()
                if pd.notna(row["Contagem"])
            }

            pdf_bytes = generate_jacobs_report(
                event_name=j2_event, audit_code=audit_code, grid_rows=j2_rows, grid_cols=j2_cols, total_cells=total_cells,
                excluded_cells=sorted(excluded_set), sampled_cells=sampled, cell_counts=_cell_counts,
                estimate=est["estimate"], margin=est["margin"], lower=est["lower"], upper=est["upper"],
                avg_per_cell=est["avg_per_cell"], std_per_cell=est["std_per_cell"], coverage_pct=est["coverage_pct"],
                crowd_cells=est["crowd_cells"], cell_area_m2=cell_area_m2, camera_altitude=float(j2_altitude),
                camera_fov=float(j2_fov), grid_image_bgr=_g2, ground_w=ground_w, ground_h=ground_h,
            )

            st.download_button(
                "📄 Baixar relatório PDF", data=pdf_bytes, file_name="veridic_jacobs_relatorio.pdf",
                mime="application/pdf", use_container_width=True,
            )
