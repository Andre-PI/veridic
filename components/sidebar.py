import math
import streamlit as st
from processor import DEFAULT_WEIGHTS_PATH, DEFAULT_ZONES_X, DEFAULT_ZONES_Y

_PRESETS = {
    "🎪 Show / Festa junina": dict(sample_interval=8, altitude=50, fov=82.1),
    "✊ Protesto / Marcha":    dict(sample_interval=15, altitude=50, fov=82.1),
    "⛪ Concentração estática": dict(sample_interval=20, altitude=40, fov=82.1),
    "⚙️ Personalizado":         dict(sample_interval=15, altitude=40, fov=82.1),
}

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
        config["global_density"] = global_density
        
        if global_density:
            area_ref = int(config["known_area"]) if config["known_area"] > 0 else auto_area
            st.caption(f"→ Jacobs estimado: **{int(area_ref * global_density):,} pessoas**")
            
        st.markdown("---")
        st.subheader("Visualização")
        
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
            zones_x = int(_zc1.number_input("Colunas", min_value=2, max_value=6, value=DEFAULT_ZONES_X))
            zones_y = int(_zc2.number_input("Linhas",  min_value=2, max_value=6, value=DEFAULT_ZONES_Y))

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
        else:
            zones_x, zones_y = DEFAULT_ZONES_X, DEFAULT_ZONES_Y
            
        config["zones_x"] = zones_x
        config["zones_y"] = zones_y

        if global_density:
            for r in range(zones_y):
                for c in range(zones_x):
                    manual_densities[(r, c)] = global_density
                    
        config["manual_densities"] = manual_densities

    return config
