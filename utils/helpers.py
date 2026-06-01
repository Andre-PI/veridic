import os
import tempfile
from pathlib import Path
from textwrap import dedent
import streamlit as st

_tmp_files = []

def cleanup_tmp():
    for p in _tmp_files:
        try:
            os.unlink(p)
        except OSError:
            pass

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

def render_metric_box(value, label, value_color=None, value_size=None):
    style_str = ""
    if value_color or value_size:
        style_str = ' style="'
        if value_color:
            style_str += f"color:{value_color};"
        if value_size:
            style_str += f"font-size:{value_size};"
        style_str += '"'
    
    render_html(f"""
    <div class="metric-box">
        <div class="metric-value"{style_str}>{value}</div>
        <div class="metric-label">{label}</div>
    </div>""")

def compute_agreement(a, b):
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

from processor import load_model

@st.cache_resource
def get_cached_model(weights_path):
    return load_model(weights_path)

