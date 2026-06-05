"""
app.py  –  Streamlit UI for the Insurance Synthetic Data Pipeline
Run with:  streamlit run app.py
Place this file in the same directory as Pipeline.py, retriever.py, agents.py, etc.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import io
import threading
import time
import random
import queue
from pathlib import Path
from contextlib import redirect_stdout

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Generative Claims · Data Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* Root variables */
:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --surface2:  #1a1e2a;
    --border:    #252a38;
    --accent:    #e8c547;
    --accent2:   #4fc3f7;
    --danger:    #ff6b6b;
    --success:   #69db7c;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    10px;
}

/* Global reset */
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }
* { font-family: 'IBM Plex Sans', sans-serif !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hide Streamlit's built-in sidebar collapse/expand button entirely */
button[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] > div > button[kind="header"] { display: none !important; }

.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }

/* ── Hero header ────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #13161e 0%, #1a1e2a 60%, #0d1220 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(232,197,71,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    color: #fff !important;
    margin: 0 0 0.4rem !important;
    line-height: 1.1 !important;
}
.hero-title span { color: var(--accent); }
.hero-subtitle {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1rem !important;
    color: var(--muted) !important;
    margin: 0 !important;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(232,197,71,0.12);
    border: 1px solid rgba(232,197,71,0.3);
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    padding: 4px 10px; border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Section labels ─────────────────────────────────────────── */
.section-label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Prompt cards ───────────────────────────────────────────── */
.prompt-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    cursor: pointer;
    transition: all 0.18s ease;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.prompt-card:hover {
    border-color: var(--accent);
    background: var(--surface2);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.prompt-card.active {
    border-color: var(--accent) !important;
    background: rgba(232,197,71,0.06) !important;
}
.prompt-card::after {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, transparent 70%, rgba(232,197,71,0.04));
    pointer-events: none;
}
.pc-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; }
.pc-name {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0 0 0.3rem !important;
}
.pc-desc {
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    margin: 0 0 0.6rem !important;
    line-height: 1.45 !important;
}
.pc-tags { display: flex; flex-wrap: wrap; gap: 4px; }
.pc-tag {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6rem !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent2) !important;
    padding: 2px 7px; border-radius: 4px;
}

/* ── Stat cards ─────────────────────────────────────────────── */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.3rem;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: var(--accent) !important;
    margin: 0 !important;
    line-height: 1 !important;
}
.stat-label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-top: 4px !important;
}

/* ── Log terminal ───────────────────────────────────────────── */
.log-box {
    background: #080a0f;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #a0a8b8 !important;
    max-height: 320px;
    overflow-y: auto;
    line-height: 1.7 !important;
    white-space: pre-wrap;
}
.log-success { color: var(--success) !important; }
.log-error   { color: var(--danger)  !important; }
.log-info    { color: var(--accent2) !important; }

/* ── Streamlit widget overrides ─────────────────────────────── */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stNumberInput"] label {
    color: var(--text) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
div[data-testid="stSlider"] .stMarkdown p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }
.sidebar-brand {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    color: var(--accent) !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem !important;
}
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 0.8rem 0 0.4rem !important;
    border-top: 1px solid var(--border) !important;
    margin-top: 0.5rem !important;
}

/* Buttons */
div[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.2px;
}
div[data-testid="stButton"] > button:hover {
    background: #f0d060 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(232,197,71,0.25) !important;
}
div[data-testid="stButton"] > button:disabled {
    background: var(--surface2) !important;
    color: var(--muted) !important;
}

/* Sidebar toggle button — small, ghost style */
div[data-testid="stButton"][id="sidebar-toggle-wrap"] > button {
    background: var(--surface2) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 1rem !important;
    padding: 0.2rem 0.6rem !important;
    width: auto !important;
}
div[data-testid="stButton"][id="sidebar-toggle-wrap"] > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 4px 12px rgba(232,197,71,0.15) !important;
}

/* Download button */
div[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--accent2) !important;
    border: 1px solid var(--accent2) !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(79,195,247,0.1) !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
.stDataFrame { background: var(--surface) !important; }

/* Expander */
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Info/warning boxes */
div[data-testid="stInfo"]    { background: rgba(79,195,247,0.08) !important; border-color: rgba(79,195,247,0.3) !important; }
div[data-testid="stWarning"] { background: rgba(232,197,71,0.08) !important; border-color: rgba(232,197,71,0.3) !important; }
div[data-testid="stSuccess"] { background: rgba(105,219,124,0.08) !important; border-color: rgba(105,219,124,0.3) !important; }
div[data-testid="stError"]   { background: rgba(255,107,107,0.08) !important; border-color: rgba(255,107,107,0.3) !important; }
</style>
""", unsafe_allow_html=True)


# ── Schema constants (from schema.json) ──────────────────────────────────────
SEGMENTS        = ["A", "B1", "B2", "C1", "C2", "Utility"]
FUEL_TYPES      = ["CNG", "Diesel", "Petrol"]
REGION_CODES    = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
                   "C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22"]
TRANSMISSION    = ["Automatic", "Manual"]
STEERING        = ["Electric", "Manual", "Power"]
REAR_BRAKES     = ["Disc", "Drum"]
MODELS          = ["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11"]
ENGINE_TYPES    = ["1.0 SCe","1.2 L K Series Engine","1.2 L K12N Dualjet",
                   "1.5 L U2 CRDi","1.5 Turbocharged Revotorq","1.5 Turbocharged Revotron",
                   "F8D Petrol Engine","G12B","K Series Dual jet","K10C","i-DTEC"]

# ── Pre-generated scenario prompts ───────────────────────────────────────────
PROMPT_SCENARIOS = [
    {
        "id": "urban_diesel",
        "icon": "🏙️",
        "name": "Urban Diesel Fleet",
        "desc": "High-density city vehicles with Diesel engines. Typical commercial insurance profile for metro fleets.",
        "tags": ["Diesel", "C8 Region", "B2/C2", "Dense Urban"],
        "seed": {"fuel_type": "Diesel", "region_code": "C8", "segment": "B2"},
        "claim_rate": 0.08,
        "rows": 100,
    },
    {
        "id": "rural_petrol",
        "icon": "🌾",
        "name": "Rural Petrol Compact",
        "desc": "Low-density rural zones with petrol compacts. Lower premium, longer subscription periods expected.",
        "tags": ["Petrol", "C20/C21", "Segment A", "Low Density"],
        "seed": {"fuel_type": "Petrol", "region_code": "C20", "segment": "A"},
        "claim_rate": 0.05,
        "rows": 100,
    },
    {
        "id": "cng_utility",
        "icon": "⛽",
        "name": "CNG Utility Pack",
        "desc": "Utility-segment CNG vehicles common in mid-tier cities. Moderate claim exposure with predictable mileage.",
        "tags": ["CNG", "Utility", "Mid-tier City", "Moderate Claims"],
        "seed": {"fuel_type": "CNG", "segment": "Utility"},
        "claim_rate": 0.07,
        "rows": 100,
    },
    {
        "id": "high_claims",
        "icon": "📈",
        "name": "High Claims Stress Test",
        "desc": "Stress-test your model with an elevated claim rate. Useful for imbalanced dataset experiments.",
        "tags": ["High Claim Rate", "Balanced Segments", "Stress Test"],
        "seed": {},
        "claim_rate": 0.25,
        "rows": 100,
    },
    {
        "id": "premium_auto",
        "icon": "🚗",
        "name": "Premium Automatic Fleet",
        "desc": "C1/C2 segment automatics with higher NCAP ratings. Lower claim probability, high subscription value.",
        "tags": ["Automatic", "C1/C2 Segment", "Premium", "Low Claims"],
        "seed": {"transmission_type": "Automatic", "segment": "C2"},
        "claim_rate": 0.04,
        "rows": 100,
    },
    {
        "id": "balanced_baseline",
        "icon": "⚖️",
        "name": "Balanced Baseline",
        "desc": "Schema-proportional distribution across all segments. The default benchmark for model training.",
        "tags": ["All Segments", "All Fuels", "Default Distribution"],
        "seed": {},
        "claim_rate": 0.064,
        "rows": 100,
    },
]


# ── Stdout capture utility ────────────────────────────────────────────────────
class QueueStream(io.TextIOBase):
    """Thread-safe stream that pushes lines to a Queue."""
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str) -> int:
        if text.strip():
            self._q.put(text)
        return len(text)

    def flush(self):
        pass


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline_threaded(config: dict, log_queue: queue.Queue, result_container: dict):
    """
    Run the pipeline in a background thread, capturing stdout to log_queue.
    Stores result DataFrame in result_container['df'].
    """
    try:
        project_dir = config.get("project_dir", ".")
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        os.chdir(project_dir)

        from Pipeline import run_pipeline  # noqa

        log_queue.put(f"[INFO] Starting pipeline in: {project_dir}\n")
        log_queue.put(f"[INFO] Target rows: {config['n_rows']} | Claim rate: {config['claim_rate']*100:.1f}%\n")

        n_rows     = config["n_rows"]
        claim_rate = config["claim_rate"]
        seed_base  = config.get("seed_fields", {})

        claim_rows    = round(n_rows * claim_rate)
        no_claim_rows = n_rows - claim_rows

        all_rows = []

        # Batch 1 – rows with claim_status = 1
        if claim_rows > 0:
            log_queue.put(f"\n[BATCH 1/2] Generating {claim_rows} CLAIM rows (claim_status=1)…\n")
            stream = QueueStream(log_queue)
            with redirect_stdout(stream):
                df1 = run_pipeline(
                    schema_path        = config["schema_path"],
                    embeddings_path    = config["embeddings_path"],
                    chroma_path        = config["chroma_path"],
                    output_path        = "__tmp_claims__.csv",
                    n_rows             = claim_rows,
                    seed_fields        = {**seed_base, "claim_status": 1},
                    skip_profiling     = config["skip_profiling"],
                    skip_embedding     = config["skip_embedding"],
                    skip_indexing      = config["skip_indexing"],
                    reset_index        = config["reset_index"],
                    generation_workers = config["gen_workers"],
                    embed_workers      = config["embed_workers"],
                    resume             = False,
                    checkpoint_path    = "__ckpt_claims__.jsonl",
                )
            if df1 is not None and not df1.empty:
                all_rows.append(df1)

        # Batch 2 – rows with claim_status = 0
        if no_claim_rows > 0:
            log_queue.put(f"\n[BATCH 2/2] Generating {no_claim_rows} NO-CLAIM rows (claim_status=0)…\n")
            stream = QueueStream(log_queue)
            with redirect_stdout(stream):
                df2 = run_pipeline(
                    schema_path        = config["schema_path"],
                    embeddings_path    = config["embeddings_path"],
                    chroma_path        = config["chroma_path"],
                    output_path        = "__tmp_noclaims__.csv",
                    n_rows             = no_claim_rows,
                    seed_fields        = {**seed_base, "claim_status": 0},
                    skip_profiling     = True,
                    skip_embedding     = True,
                    skip_indexing      = True,
                    reset_index        = False,
                    generation_workers = config["gen_workers"],
                    embed_workers      = config["embed_workers"],
                    resume             = False,
                    checkpoint_path    = "__ckpt_noclaims__.jsonl",
                )
            if df2 is not None and not df2.empty:
                all_rows.append(df2)

        # Merge and shuffle
        if all_rows:
            df_final = pd.concat(all_rows, ignore_index=True)
            df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
            out = os.path.join(project_dir, config["output_path"])
            df_final.to_csv(out, index=False)
            result_container["df"] = df_final
            log_queue.put(f"\n[✓] DONE — {len(df_final)} rows saved to {out}\n")
        else:
            log_queue.put("[✗] No rows were generated. Check your pipeline config.\n")
            result_container["df"] = pd.DataFrame()

    except ImportError as e:
        log_queue.put(f"[✗] Import error: {e}\n    Make sure Pipeline.py is in the project directory.\n")
        result_container["df"] = None
    except Exception as e:
        import traceback
        log_queue.put(f"[✗] Pipeline error: {e}\n{traceback.format_exc()}\n")
        result_container["df"] = None
    finally:
        log_queue.put("__DONE__")


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "active_prompt":    None,
        "n_rows":           100,
        "claim_rate":       6.0,
        # Widget keys — single source of truth for the selectboxes.
        # Scenario load writes here directly; Streamlit restores them on rerun.
        "sb_segment":       "— any —",
        "sb_fuel":          "— any —",
        "sb_region":        "— any —",
        "sb_transmission":  "— any —",
        "is_running":       False,
        "log_lines":        [],
        "result_df":        None,
        "run_complete":     False,
        "sidebar_open":     True,   # ← drives our custom toggle
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Hardcoded pipeline defaults (paths / workers / stage skips) ───────────────
gen_workers     = 2
embed_workers   = 8
skip_profiling  = True
skip_embedding  = True
skip_indexing   = True
reset_index     = False

# ── Collapse sidebar via CSS when toggled off ─────────────────────────────────
if not st.session_state["sidebar_open"]:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Close button inside the sidebar
    if st.button("✕  Close panel", key="sidebar_close"):
        st.session_state["sidebar_open"] = False
        st.rerun()

    st.markdown('<div class="sidebar-brand">Generative Claims</div>', unsafe_allow_html=True)
    st.caption("Vehicle Insurance · Synthetic Data Generator")

    # ── Generation Parameters ────────────────────────────────────
    st.markdown('<div class="sidebar-section">⚙ Generation</div>', unsafe_allow_html=True)

    n_rows = st.slider(
        "Number of rows",
        min_value=10, max_value=2000, step=10,
        value=st.session_state["n_rows"],
        help="Total synthetic rows to generate",
    )
    st.session_state["n_rows"] = n_rows

    claim_rate = st.slider(
        "Target claim rate (%)",
        min_value=0.0, max_value=100.0, step=1.0,
        value=st.session_state["claim_rate"],
        format="%.1f%%",
        help="Real-world baseline is ~6%. Raise to stress-test models with class imbalance.",
    )
    st.session_state["claim_rate"] = claim_rate

    # Computed split preview
    c_rows  = round(n_rows * claim_rate / 100)
    nc_rows = n_rows - c_rows
    col1, col2 = st.columns(2)
    col1.metric("Claim rows",    c_rows,  delta=None)
    col2.metric("No-claim rows", nc_rows, delta=None)

    # ── File Paths ───────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📁 Paths</div>', unsafe_allow_html=True)

    project_dir     = st.text_input("Project directory",  value=os.getcwd(),
                                     help="Directory containing Pipeline.py, schema.json, etc.")
    schema_path     = st.text_input("schema.json path",   value="schema.json")
    embeddings_path = st.text_input("embeddings.json",    value="embeddings.json")
    chroma_path     = st.text_input("ChromaDB path",      value="./chroma_db")
    output_path     = st.text_input("Output CSV",         value="synthetic_data.csv")

    st.divider()
    st.caption("v1.0 · Ollama + ChromaDB + LLM Agents")


# ── Main page ─────────────────────────────────────────────────────────────────

# Sidebar open button — only visible when sidebar is hidden
if not st.session_state["sidebar_open"]:
    if st.button("☰  Settings", key="sidebar_open_btn"):
        st.session_state["sidebar_open"] = True
        st.rerun()

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ INSURANCE · SYNTHETIC DATA · RAG PIPELINE</div>
  <div class="hero-title">Generative <span>Claims</span></div>
  <div class="hero-subtitle">
    Generate realistic, schema-constrained vehicle insurance data using Ollama LLMs, 
    ChromaDB vector retrieval, and multi-agent validation — fully locally.
  </div>
</div>
""", unsafe_allow_html=True)


# ── Quick Start Scenarios ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">▸ Quick-start scenarios — click to load a configuration</div>',
            unsafe_allow_html=True)

cols = st.columns(3)
for i, scenario in enumerate(PROMPT_SCENARIOS):
    with cols[i % 3]:
        is_active  = st.session_state["active_prompt"] == scenario["id"]
        active_cls = "active" if is_active else ""
        tags_html  = "".join(f'<span class="pc-tag">{t}</span>' for t in scenario["tags"])

        st.markdown(f"""
        <div class="prompt-card {active_cls}">
            <span class="pc-icon">{scenario['icon']}</span>
            <div class="pc-name">{scenario['name']}</div>
            <div class="pc-desc">{scenario['desc']}</div>
            <div class="pc-tags">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Load · {scenario['name']}", key=f"btn_{scenario['id']}",
                     use_container_width=True):
            st.session_state["active_prompt"]  = scenario["id"]
            st.session_state["n_rows"]         = scenario["rows"]
            st.session_state["claim_rate"]     = round(scenario["claim_rate"] * 100, 1)
            seed = scenario["seed"]
            # Write directly to the widget keys — Streamlit's key-owned state
            # always wins over the `index=` argument, so this is the only
            # reliable way to pre-set a selectbox value before rerun.
            st.session_state["sb_segment"]      = seed.get("segment",          "— any —")
            st.session_state["sb_fuel"]         = seed.get("fuel_type",        "— any —")
            st.session_state["sb_region"]       = seed.get("region_code",      "— any —")
            st.session_state["sb_transmission"] = seed.get("transmission_type","— any —")
            st.rerun()

st.divider()


# ── Custom Seed Fields ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">▸ Seed fields — fix specific values (anchor + override)</div>',
            unsafe_allow_html=True)

seg_opts   = ["— any —"] + SEGMENTS
fuel_opts  = ["— any —"] + FUEL_TYPES
reg_opts   = ["— any —"] + REGION_CODES
trans_opts = ["— any —"] + TRANSMISSION

c1, c2, c3, c4 = st.columns(4)

with c1:
    seed_segment = st.selectbox(
        "Segment", options=seg_opts,
        key="sb_segment",
    )
with c2:
    seed_fuel = st.selectbox(
        "Fuel type", options=fuel_opts,
        key="sb_fuel",
    )
with c3:
    seed_region = st.selectbox(
        "Region code", options=reg_opts,
        key="sb_region",
    )
with c4:
    seed_transmission = st.selectbox(
        "Transmission", options=trans_opts,
        key="sb_transmission",
    )

# Build seed_fields dict from UI selections
seed_fields = {}
if seed_segment      != "— any —": seed_fields["segment"]           = seed_segment
if seed_fuel         != "— any —": seed_fields["fuel_type"]         = seed_fuel
if seed_region       != "— any —": seed_fields["region_code"]       = seed_region
if seed_transmission != "— any —": seed_fields["transmission_type"] = seed_transmission

if seed_fields:
    tags = " · ".join(f"`{k}={v}`" for k, v in seed_fields.items())
    st.info(f"🔒 Seeded fields: {tags}  ·  All other fields sampled by the LLM agents.")
else:
    st.info("💡 No seed fields set — the pipeline will randomly anchor segment, fuel_type, and region_code per row.")

st.divider()


# ── Generate button ───────────────────────────────────────────────────────────
col_btn, col_spacer = st.columns([2, 5])
with col_btn:
    generate_clicked = st.button(
        "⚡  Generate Synthetic Data",
        use_container_width=True,
        disabled=st.session_state["is_running"],
    )

if generate_clicked and not st.session_state["is_running"]:
    st.session_state["is_running"]   = True
    st.session_state["log_lines"]    = []
    st.session_state["result_df"]    = None
    st.session_state["run_complete"] = False

    pipeline_config = {
        "n_rows":           st.session_state["n_rows"],
        "claim_rate":       st.session_state["claim_rate"] / 100,
        "seed_fields":      seed_fields,
        "gen_workers":      gen_workers,
        "embed_workers":    embed_workers,
        "project_dir":      project_dir,
        "schema_path":      schema_path,
        "embeddings_path":  embeddings_path,
        "chroma_path":      chroma_path,
        "output_path":      output_path,
        "skip_profiling":   skip_profiling,
        "skip_embedding":   skip_embedding,
        "skip_indexing":    skip_indexing,
        "reset_index":      reset_index,
    }

    log_q         = queue.Queue()
    result_holder = {}

    thread = threading.Thread(
        target=run_pipeline_threaded,
        args=(pipeline_config, log_q, result_holder),
        daemon=True,
    )
    thread.start()

    # Live log display
    st.markdown('<div class="section-label">▸ Pipeline log</div>', unsafe_allow_html=True)
    log_placeholder     = st.empty()
    spinner_placeholder = st.empty()

    with spinner_placeholder:
        with st.spinner("Running pipeline… this may take several minutes depending on row count."):
            lines = []
            while True:
                try:
                    msg = log_q.get(timeout=0.3)
                    if msg == "__DONE__":
                        break
                    lines.append(msg.rstrip())
                    visible  = lines[-60:]
                    log_html = "\n".join(visible)
                    log_placeholder.markdown(
                        f'<div class="log-box">{log_html}</div>',
                        unsafe_allow_html=True,
                    )
                except queue.Empty:
                    if not thread.is_alive():
                        break
            thread.join(timeout=5)

    st.session_state["log_lines"]    = lines
    st.session_state["result_df"]    = result_holder.get("df")
    st.session_state["is_running"]   = False
    st.session_state["run_complete"] = True
    spinner_placeholder.empty()
    st.rerun()


# ── Show persisted log after run ──────────────────────────────────────────────
if st.session_state["log_lines"] and not st.session_state["is_running"]:
    st.markdown('<div class="section-label">▸ Pipeline log</div>', unsafe_allow_html=True)
    log_html = "\n".join(st.session_state["log_lines"][-60:])
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state["run_complete"] and st.session_state["result_df"] is not None:
    df = st.session_state["result_df"]

    if df.empty:
        st.error("⚠ No rows were generated. Check the pipeline log for errors.")
    else:
        st.success(f"✅ Generation complete — **{len(df)} rows** produced successfully.")

        st.divider()
        st.markdown('<div class="section-label">▸ Results overview</div>', unsafe_allow_html=True)

        # Stat cards
        actual_claim_rate = df["claim_status"].mean() * 100 if "claim_status" in df.columns else 0
        n_fields   = len(df.columns)
        n_segments = df["segment"].nunique()   if "segment"     in df.columns else "—"
        n_regions  = df["region_code"].nunique() if "region_code" in df.columns else "—"

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.markdown(f'<div class="stat-card"><div class="stat-value">{len(df)}</div><div class="stat-label">Total Rows</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-card"><div class="stat-value">{actual_claim_rate:.1f}%</div><div class="stat-label">Claim Rate</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-card"><div class="stat-value">{n_fields}</div><div class="stat-label">Fields</div></div>', unsafe_allow_html=True)
        s4.markdown(f'<div class="stat-card"><div class="stat-value">{n_segments}</div><div class="stat-label">Segments</div></div>', unsafe_allow_html=True)
        s5.markdown(f'<div class="stat-card"><div class="stat-value">{n_regions}</div><div class="stat-label">Regions</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🗂️ Full Data", "📈 Field Stats"])

        with tab1:
            ch1, ch2 = st.columns(2)
            with ch1:
                if "segment" in df.columns:
                    st.subheader("Segment distribution")
                    seg_counts = df["segment"].value_counts().reset_index()
                    seg_counts.columns = ["segment", "count"]
                    st.bar_chart(seg_counts.set_index("segment"))
            with ch2:
                if "fuel_type" in df.columns:
                    st.subheader("Fuel type distribution")
                    fuel_counts = df["fuel_type"].value_counts().reset_index()
                    fuel_counts.columns = ["fuel_type", "count"]
                    st.bar_chart(fuel_counts.set_index("fuel_type"))

            ch3, ch4 = st.columns(2)
            with ch3:
                if "claim_status" in df.columns:
                    st.subheader("Claim status split")
                    claim_counts = df["claim_status"].value_counts().reset_index()
                    claim_counts.columns = ["claim_status", "count"]
                    claim_counts["claim_status"] = claim_counts["claim_status"].map({0: "No Claim", 1: "Claim"})
                    st.bar_chart(claim_counts.set_index("claim_status"))
            with ch4:
                if "transmission_type" in df.columns:
                    st.subheader("Transmission type")
                    trans_counts = df["transmission_type"].value_counts().reset_index()
                    trans_counts.columns = ["transmission_type", "count"]
                    st.bar_chart(trans_counts.set_index("transmission_type"))

        with tab2:
            st.dataframe(df, use_container_width=True, height=460)

        with tab3:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                st.dataframe(
                    df[numeric_cols].describe().T.style.format("{:.3f}"),
                    use_container_width=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download synthetic_data.csv",
            data=csv_bytes,
            file_name="synthetic_data.csv",
            mime="text/csv",
            use_container_width=False,
        )

elif st.session_state["run_complete"] and st.session_state["result_df"] is None:
    st.error("❌ Pipeline failed. Check the log above for details.")