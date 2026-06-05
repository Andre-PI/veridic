import math
import streamlit as st
from utils.helpers import render_html
from processor import DEFAULT_WEIGHTS_PATH, DEFAULT_ZONES_X, DEFAULT_ZONES_Y

_PRESETS = {
    "🎪 Show / Festa junina": dict(sample_interval=8, altitude=50, fov=82.1),
    "✊ Protesto / Marcha":    dict(sample_interval=15, altitude=50, fov=82.1),
    "⛪ Concentração estática": dict(sample_interval=20, altitude=40, fov=82.1),
    "🏟️ Evento aéreo alto":   dict(sample_interval=10, altitude=114, fov=82.1),
    "⚙️ Personalizado":         dict(sample_interval=15, altitude=40, fov=82.1),
}


def _zone_style(density_value):
    if density_value is None:
        return "background:rgba(59,130,246,0.10);border-color:rgba(148,163,184,0.35);color:#d7e2f2;"
    if density_value <= 0:
        return "background:rgba(15,23,42,0.95);border-color:rgba(148,163,184,0.35);color:#93a4bd;"
    if density_value <= 1:
        return "background:rgba(37,99,235,0.35);border-color:rgba(96,165,250,0.8);color:#eff6ff;"
    if density_value <= 2:
        return "background:rgba(34,197,94,0.28);border-color:rgba(74,222,128,0.75);color:#ecfdf5;"
    if density_value <= 3:
        return "background:rgba(234,179,8,0.32);border-color:rgba(250,204,21,0.9);color:#fff7ed;"
    if density_value <= 4:
        return "background:rgba(249,115,22,0.34);border-color:rgba(251,146,60,0.95);color:#fff7ed;"
    return "background:rgba(239,68,68,0.38);border-color:rgba(248,113,113,0.95);color:#fff1f2;"


def _zone_label(density_value):
    if density_value is None:
        return "Auto"
    if density_value <= 0:
        return "0"
    return str(int(density_value))

def render_sidebar():
    config = {}
    with st.sidebar:
        st.header("Configuração")

        config["event_name"] = st.text_input("Nome do evento", value="Evento", help="Aparece no relatório PDF.")
        preset_key = st.selectbox("Tipo de evento", list(_PRESETS.keys()), index=0)
        preset = _PRESETS[preset_key]

        config["weights_path"] = st.text_input("Pesos do modelo", value=DEFAULT_WEIGHTS_PATH)
        config["max_size"] = st.selectbox(
            "Tamanho de inferência",
            options=(640, 800, 1024, 1280),
            index=2,
            help="Menor = mais rápido. Maior = mais preciso.",
        )
        config["sample_interval"] = st.slider(
            "Amostrar a cada N frames (vídeo)",
            min_value=1, max_value=60, value=preset["sample_interval"],
            help="1 = todo frame (lento). 15 = a cada 0.5s a 30fps.",
        )
        
        st.markdown("---")
        st.subheader("Câmera (Jacobs estimado)")
        
        config["camera_altitude"] = st.number_input(
            "Altitude do drone (m)", min_value=5, max_value=300, value=preset["altitude"],
            help="Posicione o drone sobre o centro do venue. A área coberta pelo frame será usada como área do evento.",
        )
        config["camera_fov"] = st.number_input(
            "FOV horizontal (°)", min_value=40.0, max_value=150.0, value=preset["fov"],
            help="DJI Mini 4 Pro = 82.1°.",
        )

        gsd_w = round(2 * config["camera_altitude"] * math.tan(math.radians(config["camera_fov"] / 2)), 1)
        gsd_h = round(gsd_w * 9 / 16, 1)
        auto_area = round(gsd_w * gsd_h)
        st.caption(f"📐 Cobertura estimada a {config['camera_altitude']}m: **{gsd_w}m × {gsd_h}m ≈ {auto_area:,} m²**")
        config["auto_area"] = auto_area

        config["known_area"] = st.number_input(
            "Área real do venue (m²)",
            min_value=0, max_value=500_000, value=0, step=100,
            help="Deixe 0 para usar a cobertura calculada pelo drone. Preencha quando souber a área exata — aumenta a precisão do Jacobs.",
        )
        
        if config["known_area"] > 0:
            st.caption(f"✅ Usando área informada: **{int(config['known_area']):,} m²**")
        else:
            st.caption(f"🔍 Usando área detectada pelo drone: **{auto_area:,} m²**")

        _density_opts = {
            "Auto (CSRNet)":          None,
            "vazio / palco — 0 p/m²": 0.0,
            "espaçada — 1 p/m²":      1.0,
            "moderada — 2 p/m²":      2.0,
            "densa — 3 p/m²":         3.0,
            "aglomerada — 4 p/m²":    4.0,
            "comprimida — 5 p/m²":    5.0,
        }

        _global_density_opts = {k: v for k, v in _density_opts.items() if v is None or v > 0}
        global_density_sel = st.selectbox(
            "Densidade observada (geral)",
            list(_global_density_opts.keys()), index=0,
            help="Observador define a densidade predominante do evento. Sobrescreve o CSRNet no Jacobs.",
        )
        global_density = _global_density_opts[global_density_sel]
        config["global_density"] = global_density
        
        if global_density:
            area_ref = int(config["known_area"]) if config["known_area"] > 0 else auto_area
            st.caption(f"→ Jacobs estimado: **{int(area_ref * global_density):,} pessoas**")
            
        st.markdown("---")
        st.subheader("Visualização")

        config["heatmap_mode"] = st.radio(
            "Mapa de densidade",
            options=("CSRNet (modelo)", "Jacobs por setor"),
            index=1 if preset_key == "🏟️ Evento aéreo alto" else 0,
            help="Para drone alto e multidão densa, o mapa por setor costuma ficar mais coerente que o heatmap do modelo.",
        )
        
        config["show_contours"] = st.checkbox("Mostrar regiões contadas", value=True,
                                              help="Contorna as regiões do frame onde a contagem está ocorrendo.")
        if config["show_contours"]:
            config["contour_percentile"] = st.slider(
                "Sensibilidade",
                min_value=10, max_value=90, value=60, step=5,
                help="Menor = mostra regiões mais fracas também. Maior = só regiões de alta densidade.",
            )
        else:
            config["contour_percentile"] = 60
            
        st.subheader("Zonas")
        config["show_zones"] = st.checkbox("Dividir em zonas", value=False)
        
        manual_densities = {}
        if config["show_zones"]:
            _zc1, _zc2 = st.columns(2)
            zones_x = int(_zc1.number_input("Colunas", min_value=2, max_value=10, value=DEFAULT_ZONES_X,
                                           help="Número de colunas da grade (máx. 10)."))
            zones_y = int(_zc2.number_input("Linhas",  min_value=2, max_value=10, value=DEFAULT_ZONES_Y,
                                           help="Número de linhas da grade (máx. 10)."))

            st.caption("**Densidade por zona** — observador pode sobrescrever:")
            for r in range(zones_y):
                cols_ui = st.columns(zones_x)
                for c in range(zones_x):
                    sel = cols_ui[c].selectbox(
                        f"Z{r+1},{c+1}", list(_density_opts.keys()),
                        index=0, label_visibility="visible", key=f"zone_{r}_{c}"
                    )
                    if _density_opts[sel] is not None:
                        manual_densities[(r, c)] = _density_opts[sel]

            preview_rows = []
            for r in range(zones_y):
                cells = []
                for c in range(zones_x):
                    sel_key = st.session_state.get(f"zone_{r}_{c}", "Auto (CSRNet)")
                    density_value = _density_opts.get(sel_key)
                    cells.append(
                        f'<div style="flex:1;min-width:0;padding:0.55rem 0.35rem;border-radius:10px;border:1px solid;'
                        f'text-align:center;{_zone_style(density_value)}">'
                        f'<div style="font-size:0.72rem;font-weight:800;opacity:0.9;">Z{r+1},{c+1}</div>'
                        f'<div style="font-size:1rem;font-weight:900;line-height:1.1;">{_zone_label(density_value)}</div>'
                        f'<div style="font-size:0.64rem;opacity:0.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                        f'{sel_key.split(" — ")[0]}</div>'
                        f'</div>'
                    )
                preview_rows.append(
                    '<div style="display:flex;gap:0.35rem;margin-bottom:0.35rem;">' + "".join(cells) + '</div>'
                )

            render_html(
                '<div style="margin-top:0.4rem;padding:0.8rem;border:1px solid rgba(148,163,184,0.18);'
                'border-radius:12px;background:rgba(2,6,23,0.4);">'
                '<div style="font-size:0.72rem;font-weight:800;color:#93a4bd;margin-bottom:0.5rem;">'
                'Pré-visualização das zonas</div>'
                + "".join(preview_rows)
                + '<div style="font-size:0.68rem;color:#93a4bd;margin-top:0.25rem;line-height:1.35;">'
                '0 = vazio / palco, 1-2 = baixa densidade, 3 = densa, 4 = aglomerada, 5 = comprimida.'
                '</div></div>'
            )
        else:
            zones_x, zones_y = DEFAULT_ZONES_X, DEFAULT_ZONES_Y
            
        config["zones_x"] = zones_x
        config["zones_y"] = zones_y

        if global_density:
            for r in range(zones_y):
                for c in range(zones_x):
                    # Preserve explicit manual edits: only set global density
                    # for zones that the user did not already override.
                    if (r, c) not in manual_densities:
                        manual_densities[(r, c)] = global_density
                    
        config["manual_densities"] = manual_densities

    return config
