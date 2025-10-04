"""
MekongWatch: SAR-Based Flood Monitoring System
==============================================
Production-grade Streamlit application for analyzing Mekong Basin flood/drought dynamics
using dual-polarization SAR and multi-source remote sensing data.

Stacks
- Sentinel-1 C-band SAR (VV+VH)
- CHIRPS precipitation
- Google Earth Engine pre-processing (outputs/web_assets)
- Plotly interactive visualization

Author: MekongWatch Research Team
Contact: your-email
Version: 1.0.0
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MekongWatch | SAR Flood Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'NASA Space Apps Challenge'
    }
)

# ============================================================================
# THEME / CSS
# ============================================================================
st.markdown("""
<style>
.main-title { font-size: 2.6rem; font-weight: 800; color: #1e3a8a; text-align: center; margin: 1.2rem 0; }
.subtitle   { font-size: 1.05rem; color: #64748b; text-align: center; margin-bottom: 1.2rem; }
.section-header { color:#1e40af; border-bottom:3px solid #3b82f6; padding-bottom:0.4rem; margin:1.6rem 0 0.8rem; font-weight:700; }

.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:#fff; padding:1.1rem; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.1); }
.alert-info    { background:#e0f2fe; border-left:4px solid #0284c7; padding:1rem 1.2rem; border-radius:8px; }
.alert-warning { background:#fef3c7; border-left:4px solid #f59e0b; padding:1rem 1.2rem; border-radius:8px; }
.alert-success { background:#dcfce7; border-left:4px solid #10b981; padding:1rem 1.2rem; border-radius:8px; }

.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
.stTabs [data-baseweb="tab-list"] { gap: 1rem; background:#f8fafc; padding:0.5rem; border-radius:8px; }
.stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; padding:0.6rem 1rem; border-radius:6px; }
.stTabs [aria-selected="true"] { background:#fff; box-shadow:0 2px 4px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIG / PATH RESOLUTION
# ============================================================================

DEFAULT_EVENT_MARKERS = {
    "JINGHONG_FLOW_CUT": "2019-07-15",
    "XIAOWAN_ONLINE":    "2009-01-01",
    "NUOZHADU_ONLINE":   "2012-01-01"
}

@st.cache_data(show_spinner=False)
def resolve_assets_dir() -> Path:
    """
    Find outputs/web_assets directory produced by Notebook 07.
    Priority:
      1) MEKONG_ASSETS_DIR env
      2) CWD/outputs/web_assets
      3) script_dir/outputs/web_assets
      4) script_dir.parent/outputs/web_assets
    """
    env = os.environ.get("MEKONG_ASSETS_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    candidates = [
        Path.cwd() / "outputs" / "web_assets",
        Path(__file__).resolve().parent / "outputs" / "web_assets",
        Path(__file__).resolve().parent.parent / "outputs" / "web_assets",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd()  # fallback (will error below if missing)

ASSETS = resolve_assets_dir()
MANIFEST_PATH = ASSETS / "manifest.json"

@st.cache_data(show_spinner=False)
def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        st.error(f"❌ manifest.json not found at: {path}")
        st.info("Run Notebook 07 to generate outputs/web_assets/*.json")
        st.stop()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"❌ Failed to parse manifest: {e}")
        st.stop()

MANIFEST = load_manifest(MANIFEST_PATH)

# Baselines / Events
BASELINES = MANIFEST.get("baselines", {})
EVENTS = MANIFEST.get("events", DEFAULT_EVENT_MARKERS)
EVENTS = {k: pd.to_datetime(v) for k, v in EVENTS.items()}

# ============================================================================
# DATA LOADERS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_plotly_from_assets(fname: str) -> Optional[go.Figure]:
    f = ASSETS / fname
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return go.Figure(data)
    except Exception:
        # Some plotly JSONs are full fig spec; handle both styles
        try:
            return go.Figure(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            st.warning(f"⚠️ Could not load Plotly figure '{fname}': {e}")
            return None

@st.cache_data(show_spinner=False)
def load_csv(name: str) -> Optional[pd.DataFrame]:
    f = ASSETS / name
    if not f.exists():
        return None
    try:
        return pd.read_csv(f)
    except Exception as e:
        st.warning(f"⚠️ Failed to read CSV {name}: {e}")
        return None

@st.cache_data(show_spinner=False)
def build_combined_df() -> pd.DataFrame:
    """
    Assemble annual flood/dry & precip for both AOIs if CSVs exist.
    Produced by Note 07:
      - annual_flood_delta.csv
      - annual_flood_tonlesap.csv
      - annual_dry_delta.csv
      - annual_dry_tonlesap.csv
    """
    df_fd = load_csv("annual_flood_delta.csv")
    df_ft = load_csv("annual_flood_tonlesap.csv")
    df_dd = load_csv("annual_dry_delta.csv")
    df_dt = load_csv("annual_dry_tonlesap.csv")

    dfs = []
    if df_fd is not None:
        df_fd["aoi"] = "Mekong_Delta"
        dfs.append(df_fd)
    if df_ft is not None:
        df_ft["aoi"] = "Tonle_Sap"
        dfs.append(df_ft)

    # merge dry for each AOI
    out = []
    if df_fd is not None and df_dd is not None:
        out.append(pd.merge(df_fd, df_dd, on="year", how="outer").assign(aoi="Mekong_Delta"))
    elif df_fd is not None:
        out.append(df_fd.assign(aoi="Mekong_Delta"))

    if df_ft is not None and df_dt is not None:
        out.append(pd.merge(df_ft, df_dt, on="year", how="outer").assign(aoi="Tonle_Sap"))
    elif df_ft is not None:
        out.append(df_ft.assign(aoi="Tonle_Sap"))

    if out:
        df = pd.concat(out, ignore_index=True)
        # derived
        if {"flood_vh_km2","flood_vv_km2"}.issubset(df.columns):
            df["vh_gain_km2"] = df["flood_vh_km2"] - df["flood_vv_km2"]
        return df.sort_values(["aoi","year"])
    else:
        st.error("❌ No annual CSVs found in outputs/web_assets. Run Notebook 07.")
        st.stop()

DF = build_combined_df()
AVAILABLE_AOIS = sorted(DF["aoi"].unique().tolist())

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="main-title">🛰️ MekongWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dual-Polarization SAR Analysis of Mekong Flood & Drought (2015–2024)</div>',
            unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.header("Controls")
    aoi = st.selectbox("AOI", AVAILABLE_AOIS, index=0, help="Select study area")
    year_min, year_max = int(DF["year"].min()), int(DF["year"].max())
    year_range = st.slider("Years", min_value=year_min, max_value=year_max,
                           value=(year_min, year_max), step=1)
    show_baseline = st.checkbox("Show baselines", value=True)
    show_events = st.checkbox("Show dam/event markers", value=True)
    show_vv = st.checkbox("Show VV", value=True)
    show_vh = st.checkbox("Show VH", value=True)

SEL = DF[(DF["aoi"] == aoi) & (DF["year"].between(*year_range))].copy()

# ============================================================================
# TOP METRICS
# ============================================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Years", f"{SEL['year'].nunique()}", delta=f"{year_range[0]}–{year_range[1]}")
with col2:
    if "flood_vh_km2" in SEL:
        st.metric("Mean Flood (VH)", f"{SEL['flood_vh_km2'].mean():,.0f} km²",
                  delta=f"σ = {SEL['flood_vh_km2'].std():,.0f}")
with col3:
    if {"flood_vh_km2","precip_wet_mm"}.issubset(SEL.columns):
        if SEL.dropna(subset=["flood_vh_km2","precip_wet_mm"]).shape[0] >= 3:
            r = np.corrcoef(SEL["precip_wet_mm"], SEL["flood_vh_km2"])[0,1]
            st.metric("Wet-season correlation", f"r = {r:.3f}",
                      delta="Weak" if abs(r) < 0.5 else "Moderate")
        else:
            st.metric("Wet-season correlation", "N/A")
with col4:
    if "flood_vh_km2" in SEL:
        y_peak = SEL.loc[SEL["flood_vh_km2"].idxmax(), "year"]
        v_peak = SEL["flood_vh_km2"].max()
        st.metric("Peak Flood (VH)", f"{int(y_peak)}", delta=f"{v_peak:,.0f} km²")

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🌾 Dual-Pol", "📈 Trends", "💧 Drought 2019–20", "🔬 Methods"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.header("Executive Summary")

    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("""
**Objectives**
- Quantify flood extent variability (2015–2024)
- Assess rainfall–flood linkage (CHIRPS vs SAR)
- Recover *flooded vegetation* via VH (missed by VV)
- Document the 2019–20 drought anomaly

**Highlights**
- Dual-pol SAR finds **+15–20%** hidden inundation (VH over VV)
- SAR works in monsoon cloud cover
- Baselines from Landsat-5 (2005–2008) for pre-dam reference
""")
    with c2:
        st.markdown(f"""
<div class="alert-info">
<h4 style="margin:0;">Study Area</h4>
<p><strong>AOI:</strong> {aoi.replace('_',' ')}<br>
<strong>Years:</strong> {year_range[0]}–{year_range[1]}<br>
<strong>Data:</strong> Sentinel-1 (VV/VH), CHIRPS, Landsat-5</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Try to show prebuilt composite figure, else build a quick one.
    fig_annual = None
    annual_key = MANIFEST.get("figures", {}).get("annual_flood")
    if annual_key:
        fig_annual = load_plotly_from_assets(annual_key)

    if fig_annual:
        st.subheader("Annual Flood Extent (VV vs VH) — Both AOIs")
        st.plotly_chart(fig_annual, use_container_width=True)
    else:
        # Minimal fallback for selected AOI
        st.subheader(f"Annual Flood Extent (Selected AOI: {aoi})")
        fig = go.Figure()
        if show_vv and "flood_vv_km2" in SEL:
            fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vv_km2"],
                                     mode="lines+markers", name="VV"))
        if show_vh and "flood_vh_km2" in SEL:
            fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vh_km2"],
                                     mode="lines+markers", name="VH"))
        if show_baseline:
            base = BASELINES.get(aoi, {}).get("wet_km2")
            if base:
                fig.add_hline(y=base, line_dash="dash", line_color="red",
                              annotation_text=f"Pre-dam wet baseline: {base:,.0f} km²")
        if show_events:
            for _, t in EVENTS.items():
                fig.add_vline(x=t.year, line_dash="dot", line_color="crimson", opacity=0.5)
        fig.update_layout(height=420, title="Annual Flood (Aug–Sep)")
        fig.update_xaxes(dtick=1); fig.update_yaxes(title="km²")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: DUAL-POLARIZATION
# ============================================================================
with tab2:
    st.header("Dual-Polarization SAR: Hidden Inundation (VH over VV)")
    st.markdown("""
**Physics Recap**
- **VV**: specular reflection ⇒ open water → low backscatter (≈ −16 dB threshold)
- **VH**: volume/double-bounce ⇒ flooded vegetation under crop/forest → additional detection (≈ −22 to −25 dB)

**Why it matters**: VH reveals inundation in rice paddies/vegetated floodplains that VV often misses.
""")

    # Show VH gain bar chart from manifest if available
    vh_gain_key = MANIFEST.get("figures", {}).get("vh_gain")
    fig_vh_gain = load_plotly_from_assets(vh_gain_key) if vh_gain_key else None

    if fig_vh_gain:
        st.plotly_chart(fig_vh_gain, use_container_width=True)
    else:
        # Fallback: quick bar for selected AOI
        if "vh_gain_km2" in SEL:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=SEL["year"], y=SEL["vh_gain_km2"], name="VH−VV"))
            fig.update_layout(title=f"{aoi}: VH-only Additional Inundation", height=420)
            fig.update_xaxes(dtick=1); fig.update_yaxes(title="km²")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Notebook 05/07 to generate VH gain metrics.")

# ============================================================================
# TAB 3: TRENDS
# ============================================================================
with tab3:
    st.header(f"Decadal Trends — {aoi.replace('_',' ')}")

    fig = go.Figure()
    if show_vh and "flood_vh_km2" in SEL:
        fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vh_km2"],
                                 mode="lines+markers", name="Flood (VH)",
                                 fill="tozeroy", fillcolor="rgba(59,130,246,0.12)"))
    if show_vv and "flood_vv_km2" in SEL:
        fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vv_km2"],
                                 mode="lines+markers", name="Flood (VV)"))
    if show_baseline:
        base = BASELINES.get(aoi, {}).get("wet_km2")
        if base:
            fig.add_hline(y=base, line_dash="dash", line_color="red",
                          annotation_text=f"Pre-dam wet baseline: {base:,.0f} km²")
    if show_events:
        for label, t in EVENTS.items():
            fig.add_vline(x=t.year, line_dash="dot", line_color="crimson", opacity=0.5,
                          annotation_text=label, annotation_position="top left")

    fig.update_layout(title="Annual Maximum Flood Extent (Aug–Sep)",
                      xaxis_title="Year", yaxis_title="km²", height=470,
                      hovermode="x unified")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        # Stats
        if "flood_vh_km2" in SEL:
            stats_df = pd.DataFrame({
                "Metric": ["Mean","Median","Std","Min","Max","CV (%)"],
                "Value": [
                    f"{SEL['flood_vh_km2'].mean():,.0f} km²",
                    f"{SEL['flood_vh_km2'].median():,.0f} km²",
                    f"{SEL['flood_vh_km2'].std():,.0f} km²",
                    f"{SEL['flood_vh_km2'].min():,.0f} km²",
                    f"{SEL['flood_vh_km2'].max():,.0f} km²",
                    f"{(SEL['flood_vh_km2'].std()/max(SEL['flood_vh_km2'].mean(),1e-6)*100):.1f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    with c2:
        # Linear regression on VH flood
        if "flood_vh_km2" in SEL:
            x = SEL["year"].values
            y = SEL["flood_vh_km2"].values
            if len(x) >= 3:
                slope, intercept, r, p, se = stats.linregress(x, y)
                st.markdown(f"""
**Linear Trend (VH)**
- slope: {slope:,.1f} km²/yr
- R²: {r**2:.3f}
- p-value: {p:.4f}
- trend: {"increasing" if slope>0 else "decreasing"} ({'significant' if p<0.05 else 'ns'})
""")
            else:
                st.info("Not enough points for regression")

# ============================================================================
# TAB 4: DROUGHT 2019–20
# ============================================================================
with tab4:
    st.header("2019–2020 Drought Crisis")

    # Prebuilt bi-axis figure if present
    dry_key = MANIFEST.get("figures", {}).get("dry_biaxis")
    fig_dry = load_plotly_from_assets(dry_key) if dry_key else None

    if fig_dry:
        st.plotly_chart(fig_dry, use_container_width=True)
    else:
        # Fallback bi-axis for selected AOI
        if {"dry_vh_km2","precip_dry_mm"}.issubset(SEL.columns):
            fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=SEL["year"], y=SEL["dry_vh_km2"], name="Dry-season water (km²)"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["precip_dry_mm"],
                                     mode="lines+markers", name="Dry precip (mm)"),
                          secondary_y=True)
            if show_baseline:
                base = BASELINES.get(aoi, {}).get("dry_km2")
                if base:
                    fig.add_hline(y=base, line_dash="dash", line_color="red",
                                  annotation_text=f"Pre-dam dry baseline: {base:,.0f} km²")
            if show_events:
                for _, t in EVENTS.items():
                    fig.add_vline(x=t.year, line_dash="dot", line_color="crimson", opacity=0.5)
            fig.update_layout(title=f"{aoi.replace('_',' ')} — Dry-season water vs precip",
                              height=460)
            fig.update_xaxes(dtick=1)
            fig.update_yaxes(title_text="Water (km²)", secondary_y=False)
            fig.update_yaxes(title_text="Precip (mm)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Notebook 06/07 to generate drought series")

    st.markdown("""
**Notes**
- Lowest Mekong levels in ~100 years reported in 2019–20.
- Event markers (e.g., Jinghong flow cut in Jul 2019) align with anomalous drawdown downstream.
""")

# ============================================================================
# TAB 5: METHODS
# ============================================================================
with tab5:
    st.header("Methodology & Data")
    st.markdown("""
**Data Sources**
- COPERNICUS/S1_GRD (ESA) — VV/VH (12-day)
- UCSB-CHG/CHIRPS/DAILY — Precipitation
- LANDSAT/LT05/C02/T1_L2 — Baselines (MNDWI>0)
- NASADEM — Topography context

**SAR Classification**
```python
# Open water (VV specular)
open_water = vv_db < -16  # dB

# Flooded vegetation (VH + VV gate)
flooded_veg = (vh_db < -25) & (vv_db > -16) & (vv_db < -10)
