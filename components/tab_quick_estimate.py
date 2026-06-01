import streamlit as st
from utils.helpers import render_html

def render_tab_quick_estimate(config):
    auto_area = config.get("auto_area", 1000)
    
    render_html("""
    <div style="margin-bottom:1rem;">
        <span style="font-size:1.1rem;font-weight:800;">Estimativa Rápida (Fórmula de Jacobs)</span><br>
        <span style="font-size:0.82rem;color:#93a4bd;">
            Calcule rapidamente o público com base em Área × Densidade × Ocupação.
        </span>
    </div>
    """)
    
    col_area, col_dens = st.columns(2, gap="large")
    
    with col_area:
        st.subheader("1. Área")
        q_area = st.number_input("Área total do evento (m²)", min_value=1, value=int(auto_area) if auto_area > 0 else 1000)
        
        st.subheader("3. Ocupação")
        q_occ = st.slider("Qual porcentagem dessa área está ocupada pelo público?", min_value=0, max_value=100, value=70)
        
    with col_dens:
        st.subheader("2. Densidade")
        q_dens = st.radio(
            "Selecione a densidade visual observada",
            options=[1.0, 2.0, 3.0, 4.0, 5.0],
            format_func=lambda x: {
                1.0: "1 pessoa/m² — Esparso (muito espaço livre)",
                2.0: "2 pessoas/m² — Normal (confortável)",
                3.0: "3 pessoas/m² — Cheio (pouco espaço pessoal)",
                4.0: "4 pessoas/m² — Lotado (ombro a ombro)",
                5.0: "5 pessoas/m² — Tumulto (praticamente sem espaço)"
            }[x],
            index=2
        )
    
    q_pessoas = int(q_area * q_dens * (q_occ / 100.0))
    q_min = int(q_pessoas * 0.8)
    q_max = int(q_pessoas * 1.2)
    
    st.markdown("---")
    render_html(f"""
    <div class="jacobs-result" style="text-align: center;">
        <div style="font-size:0.85rem;color:#93a4bd;font-weight:700;margin-bottom:0.25rem;">
            RESULTADO RÁPIDO
        </div>
        <div class="jacobs-estimate" style="font-size:4rem;">{q_pessoas:,}</div>
        <div class="jacobs-ci" style="font-size:1.1rem;">
            Margem sugerida (±20%): {q_min:,} a {q_max:,} pessoas
        </div>
        <div style="font-size:0.85rem;color:#93a4bd;margin-top:1rem;">
            Cálculo: {q_area:,} m² × {q_dens} p/m² × {q_occ}% de ocupação
        </div>
    </div>
    """)
