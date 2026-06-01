import atexit
import streamlit as st

# Apply patches early before loading components
from utils.patches import apply_patches
apply_patches()

from utils.helpers import cleanup_tmp
from components.styles import inject_css, render_header
from components.sidebar import render_sidebar
from components.tab_csrnet import render_tab_csrnet
from components.tab_jacobs_real import render_tab_jacobs_real
from components.tab_quick_estimate import render_tab_quick_estimate

# Register cleanup for temp files
atexit.register(cleanup_tmp)

st.set_page_config(page_title="Veridic", layout="wide", initial_sidebar_state="collapsed")
inject_css()

# Get global config from sidebar
config = render_sidebar()

render_header()
st.markdown("")

tab1, tab2, tab3 = st.tabs(["📡  CSRNet + Jacobs estimado", "📐  Jacobs real (campo)", "⚡ Estimativa Rápida (Jacobs)"])

with tab1:
    render_tab_csrnet(config)

with tab2:
    render_tab_jacobs_real(config)

with tab3:
    render_tab_quick_estimate(config)
