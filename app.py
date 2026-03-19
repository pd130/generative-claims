"""Generative Claims — Streamlit UI v2.

A polished, modern interface for generating synthetic insurance claims data.
Features:
    - Smart prompt bar with real-time interpretation preview
    - One-click preset prompt chips
    - Side-by-side synthetic vs. real data comparison
    - Interactive charts (model, fuel, age, claims, segment, transmission)
    - Generation history
    - CSV / JSON / Excel downloads

Run:
    streamlit run app.py
"""

import sys
import time
import warnings
from pathlib import Path
import io
import json as json_lib

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.generator.synthetic_generator import SyntheticGenerator
from src.utils.config import get_settings


# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Generative Claims",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────────── */
    .block-container { padding-top: 2rem; }

    /* ── Header ─────────────────────────────────────────────── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #5b9bd5 0%, #6ec6ff 50%, #90caf9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    /* ── Prompt bar ─────────────────────────────────────────── */
    div[data-testid="stTextInput"] > div > div > input {
        font-size: 1.05rem;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        transition: border-color 0.2s;
    }
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #5b9bd5;
        box-shadow: 0 0 0 3px rgba(91, 155, 213, 0.25);
    }

    /* ── Filter badge ───────────────────────────────────────── */
    .filter-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .fb-blue   { background: #1e3a5f; color: #90caf9; }
    .fb-green  { background: #1b4332; color: #6ee7b7; }
    .fb-purple { background: #312e81; color: #c4b5fd; }
    .fb-orange { background: #78350f; color: #fcd34d; }
    .fb-red    { background: #7f1d1d; color: #fca5a5; }
    .fb-teal   { background: #134e4a; color: #5eead4; }

    /* ── Metric cards ───────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #5b9bd5;
    }
    div[data-testid="stMetric"] label { font-size: 0.75rem; opacity: 0.7; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.4rem; font-weight: 700;
    }

    /* ── Download buttons ───────────────────────────────────── */
    .stDownloadButton > button {
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: transform 0.15s;
    }
    .stDownloadButton > button:hover { transform: translateY(-1px); }

    /* ── Tabs ────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #5b9bd5;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] h1 { font-size: 1.3rem; }

    /* ── History cards ───────────────────────────────────────── */
    .history-card {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #5b9bd5;
        font-size: 0.82rem;
    }
    .history-card .ts { opacity: 0.55; font-size: 0.7rem; }

    /* ── Compare header ─────────────────────────────────────── */
    .compare-header {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ─────────────────────────────────────────────────────
@st.cache_resource
def get_generator() -> SyntheticGenerator:
    gen = SyntheticGenerator()
    gen._load_artifacts()
    return gen


@st.cache_data
def load_real_data_sample(n: int = 2000) -> pd.DataFrame:
    settings = get_settings()
    df = pd.read_csv(settings.raw_data_path)
    # Convert vehicle_age from years → months to match synthetic output
    if "vehicle_age" in df.columns:
        df["vehicle_age"] = (df["vehicle_age"] * 12).round().astype(int)
    return df.sample(n=min(n, len(df)), random_state=42)


# ── Session state init ──────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "generated_df" not in st.session_state:
    st.session_state["generated_df"] = None


# ── Preset prompts ───────────────────────────────────────────────────────
PRESETS = [
    "1000 rows diesel vehicles",
    "2000 rows 34 M3 cars rest divided among others",
    "500 rows 50% M3, 30% M4, 20% M1",
    "only M3 1000 rows",
    "5k rows no M5 no M7",
    "young drivers high risk 500 rows",
    "senior drivers petrol 8% claim rate",
    "2000 rows automatic with parking camera",
    "1k rows segment B2 disc brakes",
    "3000 rows new cars low risk",
    "500 rows utility vehicles ncap 4+",
    "1000 rows petrol or diesel age 40-60",
]


def format_filter_badges(filters: dict) -> str:
    """Convert parsed filters dict into coloured HTML badges."""
    badges: list[str] = []

    def _b(label: str, css: str = "fb-blue") -> None:
        badges.append(f'<span class="filter-badge {css}">{label}</span>')

    if "n_rows" in filters:
        _b(f"📊 {filters['n_rows']:,} rows", "fb-blue")
    if "model_allocation" in filters:
        parts = []
        for mdl, spec in filters["model_allocation"].items():
            if spec["type"] == "count":
                parts.append(f"{spec['value']} {mdl}")
            elif spec["type"] == "percent":
                parts.append(f"{spec['value']}% {mdl}")
            elif spec["type"] == "exclusive":
                parts.append(f"Only {mdl}")
        _b("🚗 " + ", ".join(parts), "fb-purple")
    elif "model" in filters:
        _b(f"🚗 Model {filters['model']}", "fb-purple")
    if "fuel_type" in filters:
        _b(f"⛽ {filters['fuel_type']}", "fb-green")
    if "fuel_types" in filters:
        _b(f"⛽ {' / '.join(filters['fuel_types'])}", "fb-green")
    if "segment" in filters:
        _b(f"🏷️ Segment {filters['segment']}", "fb-teal")
    if "transmission_type" in filters:
        _b(f"⚙️ {filters['transmission_type']}", "fb-teal")
    if "steering_type" in filters:
        _b(f"🔧 {filters['steering_type']} steering", "fb-teal")
    if "rear_brakes_type" in filters:
        _b(f"🛞 {filters['rear_brakes_type']} brakes", "fb-teal")
    if "ncap_min" in filters:
        _b(f"⭐ NCAP ≥{filters['ncap_min']}", "fb-orange")
    if "safety_on" in filters:
        for s in filters["safety_on"]:
            _b(f"✅ {s.replace('is_','').replace('_',' ').title()}", "fb-green")
    if "safety_off" in filters:
        for s in filters["safety_off"]:
            _b(f"❌ No {s.replace('is_','').replace('_',' ').title()}", "fb-red")
    if "age_min" in filters or "age_max" in filters:
        lo, hi = filters.get("age_min", "?"), filters.get("age_max", "?")
        _b(f"👤 Age {lo}–{hi}", "fb-orange")
    if "vehicle_age_min" in filters or "vehicle_age_max" in filters:
        lo, hi = filters.get("vehicle_age_min", "?"), filters.get("vehicle_age_max", "?")
        _b(f"📅 Vehicle age {lo}–{hi}", "fb-orange")
    if "subscription_min" in filters or "subscription_max" in filters:
        lo, hi = filters.get("subscription_min", "?"), filters.get("subscription_max", "?")
        _b(f"📝 Subscription {lo}–{hi}", "fb-orange")
    if "region_code" in filters:
        _b(f"📍 Region {filters['region_code']}", "fb-teal")
    if "exclude_models" in filters:
        _b(f"🚫 Exclude {', '.join(sorted(filters['exclude_models']))}", "fb-red")
    if "rest_strategy" in filters and filters.get("rest_strategy") != "distribute":
        _b(f"🔄 Rest → {filters['rest_strategy']}", "fb-blue")
    if "claim_rate_override" in filters:
        cr = filters["claim_rate_override"]
        _b(f"📋 Claim rate {cr:.0%}", "fb-red")

    return " ".join(badges) if badges else '<span class="filter-badge fb-blue">Default settings</span>'


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Generative Claims")
    st.caption("Synthetic insurance data generator powered by statistical profiling & functional dependencies")

    st.divider()

    st.markdown("### ⚙️ Defaults")
    n_rows = st.slider("Rows", 10, 10_000, 500, 10, help="Fallback when prompt doesn't specify count")
    claim_rate = st.slider("Claim rate", 0.00, 0.30, 0.06, 0.01, format="%.2f", help="Fallback claim rate")
    seed = st.number_input("Seed", 0, 99999, 42, help="Random seed for reproducibility")

    st.divider()

    # ── Data info ────────────────────────────────────────────────
    gen = get_generator()
    with st.expander("📂 Dataset Info", expanded=False):
        models = gen.get_available_models()
        segments = gen.get_available_segments()
        fuels = gen.get_available_fuel_types()
        st.markdown(f"**Models:** {', '.join(models)}")
        st.markdown(f"**Segments:** {', '.join(segments)}")
        st.markdown(f"**Fuel types:** {', '.join(fuels)}")
        st.markdown("**Age range:** 35–75")
        st.markdown("**Vehicle age:** 0–20")
        st.markdown("**Subscription:** 0–14")
        st.markdown("**Regions:** C1–C22")

    # ── Prompt syntax reference ──────────────────────────────────
    with st.expander("📖 Prompt Syntax", expanded=False):
        st.markdown("""
| Feature | Example |
|---|---|
| **Row count** | `2000 rows`, `1k rows` |
| **Model count** | `34 M3 cars` |
| **Model %** | `50% M3, 30% M4` |
| **Exclusive** | `only M3` |
| **Exclude** | `no M5 no M7` |
| **Rest split** | `rest equally`, `rest M1` |
| **Fuel** | `diesel`, `petrol or cng` |
| **Segment** | `segment B2`, `utility` |
| **Transmission** | `automatic`, `manual` |
| **Steering** | `power steering` |
| **Brakes** | `disc brakes` |
| **Safety ON** | `with parking camera` |
| **Safety OFF** | `without esc` |
| **NCAP** | `ncap 4+`, `5 star` |
| **Age** | `age 40-60`, `young`, `senior` |
| **Vehicle age** | `new cars`, `old cars` |
| **Claim rate** | `8% claim rate`, `no claims` |
        """)

    # ── Generation history ───────────────────────────────────────
    with st.expander("🕐 History", expanded=False):
        if st.session_state["history"]:
            for h in reversed(st.session_state["history"][-8:]):
                st.markdown(
                    f'<div class="history-card">'
                    f'<div class="ts">{h["time"]}</div>'
                    f'{h["prompt"]}<br>'
                    f'<b>{h["rows"]:,}</b> rows · <b>{h["rate"]}</b> claim rate · '
                    f'<b>{h["models"]}</b> models'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No generations yet.")

    st.divider()
    st.caption("Built with Python, Pandas, SciPy & Streamlit")


# ══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🛡️ Generative Claims</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Enter a natural-language prompt to generate realistic synthetic insurance data. '
    'Describe model allocation, fuel type, age range, risk level — the engine understands it all.</p>',
    unsafe_allow_html=True,
)

# ── Preset chips ─────────────────────────────────────────────────────────
st.markdown("**Quick presets** — click to populate the prompt:")
preset_cols = st.columns(4)
selected_preset = None
for i, preset in enumerate(PRESETS):
    col = preset_cols[i % 4]
    if col.button(preset, key=f"preset_{i}", use_container_width=True):
        selected_preset = preset

# ── Prompt input ─────────────────────────────────────────────────────────
prompt = st.text_input(
    "📝 Enter your generation prompt",
    value=selected_preset or "",
    placeholder="e.g., 2000 rows with 34 M3 cars, rest divided among others, diesel, young drivers, 10% claim rate ...",
    help="Natural language prompt. The parser understands models, fuel, segments, transmission, "
         "steering, brakes, safety features, NCAP, age ranges, claim rates, and more.",
    label_visibility="collapsed",
)

# ── Live interpretation preview ──────────────────────────────────────────
if prompt and prompt.strip():
    gen = get_generator()
    live_filters = gen._parse_prompt(prompt)
    badge_html = format_filter_badges(live_filters)
    st.markdown(
        f'<div style="margin: 0.3rem 0 0.8rem;">🔍 <b>Interpreted:</b> {badge_html}</div>',
        unsafe_allow_html=True,
    )

# ── Generate button ──────────────────────────────────────────────────────
gen_col, _, info_col = st.columns([2, 4, 2])
with gen_col:
    generate_clicked = st.button("🚀 Generate Data", type="primary", use_container_width=True)
with info_col:
    clear_clicked = st.button("🗑️ Clear Results", use_container_width=True)
    if clear_clicked:
        st.session_state["generated_df"] = None
        st.rerun()


# ── Generation logic ─────────────────────────────────────────────────────
if generate_clicked:
    gen = get_generator()
    t0 = time.time()

    with st.spinner("⏳ Generating synthetic data..."):
        try:
            if prompt and prompt.strip():
                filters = gen._parse_prompt(prompt)
                final_rows = filters.pop("n_rows", n_rows)
                final_rate = filters.pop("claim_rate_override", claim_rate)
                df = gen.generate(n_rows=final_rows, claim_rate=final_rate, seed=seed, prompt=prompt)
            else:
                df = gen.generate(n_rows=n_rows, claim_rate=claim_rate, seed=seed)

            elapsed = time.time() - t0

            st.session_state["generated_df"] = df
            st.session_state["elapsed"] = elapsed
            st.session_state["prompt_used"] = prompt or "(sidebar defaults)"

            # Add to history
            st.session_state["history"].append({
                "time": time.strftime("%H:%M:%S"),
                "prompt": prompt or "(defaults)",
                "rows": len(df),
                "rate": f"{df['claim_status'].mean():.2%}",
                "models": df["model"].nunique(),
            })

        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.get("generated_df") is not None:
    df = st.session_state["generated_df"]
    elapsed = st.session_state.get("elapsed", 0)

    st.success(
        f"✅ Generated **{len(df):,}** rows × **{len(df.columns)}** columns in **{elapsed:.1f}s**"
    )

    # ── Metrics row ──────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", f"{len(df.columns)}")
    m3.metric("Claim Rate", f"{df['claim_status'].mean():.2%}")
    m4.metric("Models", f"{df['model'].nunique()}")
    m5.metric("Fuel Types", f"{df['fuel_type'].nunique()}")
    m6.metric("Regions", f"{df['region_code'].nunique()}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    tab_preview, tab_charts, tab_compare, tab_details, tab_download = st.tabs(
        ["📊 Data Preview", "📈 Charts", "🔀 Compare with Real", "🔍 Column Inspector", "⬇️ Download"]
    )

    # ── Tab 1: Data Preview ──────────────────────────────────────
    with tab_preview:
        search_col = st.text_input(
            "Search / filter columns",
            placeholder="Type to filter columns ...",
            key="col_search",
        )
        if search_col:
            matched_cols = [c for c in df.columns if search_col.lower() in c.lower()]
            st.dataframe(df[matched_cols].head(200), use_container_width=True, height=450)
        else:
            st.dataframe(df.head(200), use_container_width=True, height=450)

        st.caption(f"Showing first {min(200, len(df))} of {len(df):,} rows")

    # ── Tab 2: Charts ─────────────────────────────────────────────
    with tab_charts:
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("#### Model Distribution")
            model_vc = df["model"].value_counts()
            st.bar_chart(model_vc, color="#0f3460")

        with chart_c2:
            st.markdown("#### Fuel Type Breakdown")
            fuel_vc = df["fuel_type"].value_counts()
            st.bar_chart(fuel_vc, color="#059669")

        chart_c3, chart_c4 = st.columns(2)

        with chart_c3:
            st.markdown("#### Customer Age Distribution")
            age_hist = pd.cut(df["customer_age"], bins=15).value_counts().sort_index()
            age_hist.index = age_hist.index.astype(str)
            st.bar_chart(age_hist, color="#7c3aed")

        with chart_c4:
            st.markdown("#### Claim Status")
            claim_vc = df["claim_status"].value_counts().rename({0: "No Claim", 1: "Claim"})
            st.bar_chart(claim_vc, color="#dc2626")

        chart_c5, chart_c6 = st.columns(2)
        with chart_c5:
            if "segment" in df.columns:
                st.markdown("#### Segment Distribution")
                st.bar_chart(df["segment"].value_counts(), color="#0891b2")

        with chart_c6:
            if "transmission_type" in df.columns:
                st.markdown("#### Transmission Type")
                st.bar_chart(df["transmission_type"].value_counts(), color="#d97706")

    # ── Tab 3: Compare with Real Data ────────────────────────────
    with tab_compare:
        st.markdown("#### Synthetic vs. Real Data Comparison")
        real_df = load_real_data_sample()

        compare_col = st.selectbox(
            "Select column to compare",
            [c for c in df.columns if c in real_df.columns and c != "policy_id"],
            key="compare_col",
        )

        if compare_col:
            comp_c1, comp_c2 = st.columns(2)

            with comp_c1:
                st.markdown('<div class="compare-header">🔵 Synthetic Data</div>', unsafe_allow_html=True)
                if pd.api.types.is_numeric_dtype(df[compare_col]):
                    desc_syn = df[compare_col].describe().round(2)
                    st.dataframe(desc_syn, use_container_width=True)
                    bins = pd.cut(df[compare_col], bins=15).value_counts().sort_index()
                    bins.index = bins.index.astype(str)
                    st.bar_chart(bins, color="#2563eb")
                else:
                    vc_syn = df[compare_col].value_counts().head(15)
                    st.bar_chart(vc_syn, color="#2563eb")

            with comp_c2:
                st.markdown('<div class="compare-header">🟢 Real Data</div>', unsafe_allow_html=True)
                if pd.api.types.is_numeric_dtype(real_df[compare_col]):
                    desc_real = real_df[compare_col].describe().round(2)
                    st.dataframe(desc_real, use_container_width=True)
                    bins_r = pd.cut(real_df[compare_col], bins=15).value_counts().sort_index()
                    bins_r.index = bins_r.index.astype(str)
                    st.bar_chart(bins_r, color="#059669")
                else:
                    vc_real = real_df[compare_col].value_counts().head(15)
                    st.bar_chart(vc_real, color="#059669")

            if pd.api.types.is_numeric_dtype(df[compare_col]):
                st.markdown("---")
                st.markdown("**Statistical Summary**")
                summary = pd.DataFrame({
                    "Synthetic": df[compare_col].describe(),
                    "Real": real_df[compare_col].describe(),
                }).round(2)
                summary["Difference %"] = (
                    ((summary["Synthetic"] - summary["Real"]) / summary["Real"].replace(0, np.nan)) * 100
                ).round(1)
                st.dataframe(summary, use_container_width=True)

    # ── Tab 4: Column Inspector ──────────────────────────────────
    with tab_details:
        st.markdown("#### Column-level Inspector")

        detail_col = st.selectbox(
            "Select a column",
            df.columns.tolist(),
            key="detail_col",
        )
        if detail_col:
            series = df[detail_col]

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Type", str(series.dtype))
            d2.metric("Unique", f"{series.nunique():,}")
            d3.metric("Nulls", f"{series.isna().sum()}")
            if pd.api.types.is_numeric_dtype(series):
                d4.metric("Mean", f"{series.mean():.2f}")
            else:
                d4.metric("Mode", str(series.mode().iloc[0]) if not series.mode().empty else "N/A")

            if pd.api.types.is_numeric_dtype(series):
                st.markdown("**Distribution**")
                hist_data = pd.cut(series, bins=20).value_counts().sort_index()
                hist_data.index = hist_data.index.astype(str)
                st.bar_chart(hist_data)

                st.markdown("**Percentiles**")
                pcts = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2)
                pcts.index = ["1%", "5%", "25%", "50%", "75%", "95%", "99%"]
                st.dataframe(pcts.to_frame("Value").T, use_container_width=True)
            else:
                st.markdown("**Value Counts**")
                st.bar_chart(series.value_counts().head(25))

    # ── Tab 5: Download ──────────────────────────────────────────
    with tab_download:
        st.markdown("#### Download Generated Data")

        dl1, dl2, dl3 = st.columns(3)

        # CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_data = csv_buf.getvalue()
        with dl1:
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name="synthetic_claims.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"~{len(csv_data) / 1024:.0f} KB")

        # JSON
        json_data = df.to_json(orient="records", indent=2)
        with dl2:
            st.download_button(
                "📥 Download JSON",
                data=json_data,
                file_name="synthetic_claims.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption(f"~{len(json_data) / 1024:.0f} KB")

        # Excel
        try:
            excel_buf = io.BytesIO()
            df.to_excel(excel_buf, index=False, engine="openpyxl")
            excel_bytes = excel_buf.getvalue()
            with dl3:
                st.download_button(
                    "📥 Download Excel",
                    data=excel_bytes,
                    file_name="synthetic_claims.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.caption(f"~{len(excel_bytes) / 1024:.0f} KB")
        except Exception:
            with dl3:
                st.caption("Excel export requires openpyxl")

        st.divider()
        st.markdown("**Generation Metadata**")
        meta = {
            "prompt": st.session_state.get("prompt_used", ""),
            "rows": len(df),
            "columns": len(df.columns),
            "claim_rate": round(df["claim_status"].mean(), 4),
            "models": df["model"].nunique(),
            "seed": seed,
            "generation_time_s": round(elapsed, 2),
        }
        st.json(meta)


# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: #adb5bd; font-size: 0.8rem; padding: 0.5rem;'>"
    "Generative Claims v2.0 &nbsp;·&nbsp; Profiler → Retrieval → Controller → Generator → Validator"
    "&nbsp;·&nbsp; Built with Python, Pandas, SciPy &amp; Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
