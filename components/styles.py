from utils.helpers import render_html

def inject_css():
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
        .jacobs-result {
            padding: 1.5rem; border: 1px solid rgba(34,240,107,0.4); border-radius: 10px;
            background: rgba(8,22,14,0.85); margin-top: 1rem;
        }
        .jacobs-estimate { font-size: 3rem; font-weight: 900; color: #22f06b; }
        .jacobs-ci { font-size: 1rem; color: var(--muted); margin-top: 0.25rem; }
        .legend-dot {
            display: inline-block; width: 12px; height: 12px;
            border-radius: 3px; margin-right: 6px; vertical-align: middle;
        }
    </style>
    """)

def render_header():
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
