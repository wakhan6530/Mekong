"""
MekongWatch: SAR-Based Flood Monitoring System
==============================================
Production-grade Streamlit application for analyzing Mekong Basin flood/drought dynamics
using dual-polarization SAR and multi-source remote sensing data.

Stacks
- Sentinel-1 C-band SAR (VV+VH)
- CHIRPS precipitation
- Google Earth Engine pre-processing (outputs/web_assets)  [No runtime EE required]
- Plotly interactive visualization
- Optional map via geemap (folium backend, local GeoTIFF only)

Author: MekongWatch Research Team
Contact: your-email
Version: 1.0.0 (patched)
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Optional mapping libs (NO Earth Engine at runtime)
# -----------------------------------------------------------------------------
GEEMAP_AVAILABLE = False
STF_AVAILABLE = False
try:
    # use folium backend to avoid EE initialization
    from geemap.foliumap import Map as FMap
    GEEMAP_AVAILABLE = True
except Exception:
    GEEMAP_AVAILABLE = False

try:
    from streamlit_folium import st_folium
    STF_AVAILABLE = True
except Exception:
    STF_AVAILABLE = False

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MekongWatch | SAR Flood Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': 'NASA Space Apps Challenge'}
)

# -----------------------------------------------------------------------------
# THEME / CSS
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# CONFIG / PATH RESOLUTION
# -----------------------------------------------------------------------------
DEFAULT_EVENT_MARKERS = {
    "JINGHONG_FLOW_CUT": "2019-07-15",
    "XIAOWAN_ONLINE":    "2009-01-01",
    "NUOZHADU_ONLINE":   "2012-01-01"
}

SCRIPT_DIR = Path(__file__).resolve().parent

@st.cache_data(show_spinner=False)
def resolve_assets_dir() -> Path:
    """
    outputs/web_assets directory produced by Notebook 07.
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
        SCRIPT_DIR / "outputs" / "web_assets",
        SCRIPT_DIR.parent / "outputs" / "web_assets",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default (may not exist; we handle below)

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
BASELINES: Dict[str, Dict[str, float]] = MANIFEST.get("baselines", {})
EVENTS = MANIFEST.get("events", DEFAULT_EVENT_MARKERS)
EVENTS = {k: pd.to_datetime(v) for k, v in EVENTS.items()}
# ---------------------------------------------------------------------------
# ASSET BOOTSTRAP (auto-create missing annual CSVs from existing assets)
# ---------------------------------------------------------------------------
def _candidate_paths(filename: str) -> list[Path]:
    bases = [
        ASSETS,                           # outputs/web_assets
        ASSETS.parent,                    # outputs
        Path.cwd() / "outputs",
        Path(__file__).resolve().parent / "outputs",
        Path(__file__).resolve().parent.parent / "outputs",
        Path.cwd() / "notebooks" / "outputs",
    ]
    return [b / filename for b in bases]

def _load_csv_any(filename: str) -> Optional[pd.DataFrame]:
    for p in _candidate_paths(filename):
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def _extract_flood_from_fig():
    """Rebuild annual_flood_*.csv from fig_annual_flood.json if the fig exists."""
    fig_key = MANIFEST.get("figures", {}).get("annual_flood")
    if not fig_key:
        return
    fp = ASSETS / fig_key
    if not fp.exists():
        return
    try:
        spec = json.loads(fp.read_text(encoding="utf-8"))
        data = spec["data"] if isinstance(spec, dict) and "data" in spec else spec  # tolerate both schemas
    except Exception:
        return

    # Collect x/y by AOI & pol based on trace names created in Notebook 07
    buckets = {"Mekong_Delta": {"VV": None, "VH": None},
               "Tonle_Sap":   {"VV": None, "VH": None}}
    for tr in data:
        name = tr.get("name", "")
        x = tr.get("x", [])
        y = tr.get("y", [])
        if "Delta" in name and "VV" in name:
            buckets["Mekong_Delta"]["VV"] = (x, y)
        elif "Delta" in name and "VH" in name:
            buckets["Mekong_Delta"]["VH"] = (x, y)
        elif ("Tonlé" in name or "Tonle" in name) and "VV" in name:
            buckets["Tonle_Sap"]["VV"] = (x, y)
        elif ("Tonlé" in name or "Tonle" in name) and "VH" in name:
            buckets["Tonle_Sap"]["VH"] = (x, y)

    for aoi, comp in buckets.items():
        if comp["VV"] and comp["VH"]:
            years = pd.to_numeric(pd.Series(comp["VV"][0]), errors="coerce").dropna().astype(int)
            df = pd.DataFrame({
                "year": years,
                "flood_vv_km2": pd.Series(comp["VV"][1]).iloc[:len(years)].values,
                "flood_vh_km2": pd.Series(comp["VH"][1]).iloc[:len(years)].values,
            })
            # precip_wet_mm을 여기서 복원할 수는 없으니 NaN으로 채움(상관도 탭은 figure가 대체)
            df["precip_wet_mm"] = np.nan
            outname = "annual_flood_delta.csv" if aoi == "Mekong_Delta" else "annual_flood_tonlesap.csv"
            (ASSETS / outname).write_text(df.to_csv(index=False), encoding="utf-8")

def _extract_dry_from_note06():
    """Rebuild annual_dry_*.csv from Note06 output if present."""
    dry = _load_csv_any("dry_season_analysis_2015_2024.csv")
    if dry is None:
        return
    # normalize column names
    dry = dry.rename(columns={
        "water_extent_km2": "dry_vh_km2",
        "precip_total_mm": "precip_dry_mm",
    })
    # expected: columns include year, dry_vh_km2, precip_dry_mm, [aoi]
    for aoi_key, fname in [("Mekong_Delta", "annual_dry_delta.csv"),
                           ("Tonle_Sap",   "annual_dry_tonlesap.csv")]:
        if {"year", "dry_vh_km2"}.issubset(dry.columns):
            if "aoi" in dry.columns:
                sub = dry[dry["aoi"].str.replace(" ", "_") == aoi_key][["year", "dry_vh_km2", "precip_dry_mm"]]
            else:
                sub = dry[["year", "dry_vh_km2", "precip_dry_mm"]]
            if len(sub):
                (ASSETS / fname).write_text(sub.to_csv(index=False), encoding="utf-8")

def _ensure_annual_csvs_from_figs_and_dry():
    needed = {
        "annual_flood_delta.csv",
        "annual_flood_tonlesap.csv",
        "annual_dry_delta.csv",
        "annual_dry_tonlesap.csv",
    }
    missing = [n for n in needed if not (ASSETS / n).exists()]
    if not missing:
        return
    # Try to reconstruct from existing assets
    _extract_flood_from_fig()
    _extract_dry_from_note06()
    # log
    still_missing = [n for n in needed if not (ASSETS / n).exists()]
    if still_missing:
        st.warning(f"Bootstrap: still missing files → {', '.join(still_missing)} "
                   f"(run Notebook 07 Cell 3 to compute from GEE if you need precip_wet_mm).")

# -----------------------------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_plotly_from_assets(fname_or_key: str, allow_key=True) -> Optional[go.Figure]:
    """
    Load Plotly figure from ASSETS.
    - if allow_key=True and fname_or_key is a key in MANIFEST['figures'], resolve filename.
    - else treat as filename.
    Returns a go.Figure or None.
    """
    filename = fname_or_key
    if allow_key and isinstance(MANIFEST.get("figures", {}), dict) and fname_or_key in MANIFEST["figures"]:
        filename = MANIFEST["figures"][fname_or_key]
    f = ASSETS / str(filename)
    if not f.exists():
        return None
    try:
        spec = json.loads(f.read_text(encoding="utf-8"))
        # Handle both full spec and {"data":[...], "layout":{...}}
        return go.Figure(spec)
    except Exception as e:
        st.warning(f"⚠️ Could not load Plotly figure '{filename}': {e}")
        return None

@st.cache_data(show_spinner=False)
def load_csv_from_assets(name: str) -> Optional[pd.DataFrame]:
    f = ASSETS / name
    if not f.exists():
        return None
    try:
        return pd.read_csv(f)
    except Exception as e:
        st.warning(f"⚠️ Failed to read CSV {name}: {e}")
        return None

def _normalize_cols(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    if df is None:
        return df
    cols_lower = {c.lower(): c for c in df.columns}
    for want, aliases in mapping.items():
        if want in df.columns:
            continue
        for a in aliases:
            if a in df.columns:
                df.rename(columns={a: want}, inplace=True); break
            if a.lower() in cols_lower:
                df.rename(columns={cols_lower[a.lower()]: want}, inplace=True); break
    return df

@st.cache_data(show_spinner=False)
def build_combined_df() -> pd.DataFrame:
    """
    Assemble annual flood/dry & precip for both AOIs if CSVs exist.
    Produced by Note 07:
      - annual_flood_delta.csv    (year, flood_vv_km2, flood_vh_km2, precip_wet_mm)
      - annual_flood_tonlesap.csv
      - annual_dry_delta.csv      (year, dry_vh_km2, precip_dry_mm)
      - annual_dry_tonlesap.csv
    """
    df_fd = _normalize_cols(load_csv_from_assets("annual_flood_delta.csv"), {
        "year": ["Year","year"],
        "flood_vv_km2": ["vv_km2","flood_area_vv_km2","area_km2_vv"],
        "flood_vh_km2": ["vh_km2","flood_area_vh_km2","area_km2_vh"],
        "precip_wet_mm": ["precip_mm","precipitation_mm"]
    })
    df_ft = _normalize_cols(load_csv_from_assets("annual_flood_tonlesap.csv"), {
        "year": ["Year","year"],
        "flood_vv_km2": ["vv_km2","flood_area_vv_km2","area_km2_vv"],
        "flood_vh_km2": ["vh_km2","flood_area_vh_km2","area_km2_vh"],
        "precip_wet_mm": ["precip_mm","precipitation_mm"]
    })
    df_dd = _normalize_cols(load_csv_from_assets("annual_dry_delta.csv"), {
        "year": ["Year","year"],
        "dry_vh_km2": ["water_extent_km2","vh_km2","area_km2_vh"],
        "precip_dry_mm": ["precip_total_mm","precip_mm","precipitation_mm"]
    })
    df_dt = _normalize_cols(load_csv_from_assets("annual_dry_tonlesap.csv"), {
        "year": ["Year","year"],
        "dry_vh_km2": ["water_extent_km2","vh_km2","area_km2_vh"],
        "precip_dry_mm": ["precip_total_mm","precip_mm","precipitation_mm"]
    })

    out = []
    if df_fd is not None and df_dd is not None:
        out.append(pd.merge(df_fd, df_dd, on="year", how="outer").assign(aoi="Mekong_Delta"))
    elif df_fd is not None:
        out.append(df_fd.assign(aoi="Mekong_Delta"))

    if df_ft is not None and df_dt is not None:
        out.append(pd.merge(df_ft, df_dt, on="year", how="outer").assign(aoi="Tonle_Sap"))
    elif df_ft is not None:
        out.append(df_ft.assign(aoi="Tonle_Sap"))

    if not out:
        st.error("❌ No annual CSVs found in outputs/web_assets. Run Notebook 07.")
        st.stop()

    df = pd.concat(out, ignore_index=True)
    if {"flood_vh_km2","flood_vv_km2"}.issubset(df.columns):
        df["vh_gain_km2"] = df["flood_vh_km2"] - df["flood_vv_km2"]
    return df.sort_values(["aoi","year"])

_ensure_annual_csvs_from_figs_and_dry()


DF = build_combined_df()
AVAILABLE_AOIS = sorted(DF["aoi"].unique().tolist())

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛰️ MekongWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Dual-Polarization SAR Analysis of Mekong Flood & Drought (2015–2024)</div>',
            unsafe_allow_html=True)

# Debug expander (paths, manifest)
with st.expander("Debug: assets & manifest", expanded=False):
    st.write("ASSETS:", str(ASSETS))
    st.write("Manifest:", str(MANIFEST_PATH))
    st.write("Manifest keys:", list(MANIFEST.keys()))

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# TOP METRICS
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Years", f"{SEL['year'].nunique()}", delta=f"{year_range[0]}–{year_range[1]}")
with col2:
    if "flood_vh_km2" in SEL.columns:
        st.metric("Mean Flood (VH)", f"{SEL['flood_vh_km2'].mean():,.0f} km²",
                  delta=f"σ = {SEL['flood_vh_km2'].std():,.0f}")
with col3:
    if {"flood_vh_km2","precip_wet_mm"}.issubset(SEL.columns):
        sub = SEL.dropna(subset=["flood_vh_km2","precip_wet_mm"])
        if sub.shape[0] >= 3:
            r = np.corrcoef(sub["precip_wet_mm"], sub["flood_vh_km2"])[0,1]
            st.metric("Wet-season correlation", f"r = {r:.3f}",
                      delta="Weak" if abs(r) < 0.5 else "Moderate")
        else:
            st.metric("Wet-season correlation", "N/A")
with col4:
    if "flood_vh_km2" in SEL.columns and SEL.shape[0] > 0:
        y_peak = SEL.loc[SEL["flood_vh_km2"].idxmax(), "year"]
        v_peak = SEL["flood_vh_km2"].max()
        st.metric("Peak Flood (VH)", f"{int(y_peak)}", delta=f"{v_peak:,.0f} km²")

st.markdown("---")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🌾 Dual-Pol", "📈 Trends", "💧 Drought 2019–20", "🗺️ Map", "🔬 Methods"
])

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
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

    # Try to show prebuilt composite figure (both AOIs)
    fig_annual = load_plotly_from_assets("annual_flood", allow_key=True)
    if fig_annual is not None:
        st.subheader("Annual Flood Extent (VV vs VH) — Both AOIs")
        st.plotly_chart(fig_annual, use_container_width=True)
    else:
        # Minimal fallback for selected AOI
        st.subheader(f"Annual Flood Extent (Selected AOI: {aoi})")
        fig = go.Figure()
        if show_vv and "flood_vv_km2" in SEL.columns:
            fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vv_km2"],
                                     mode="lines+markers", name="VV"))
        if show_vh and "flood_vh_km2" in SEL.columns:
            fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vh_km2"],
                                     mode="lines+markers", name="VH"))
        if show_baseline:
            base = BASELINES.get(aoi, {}).get("wet_km2")
            if base:
                fig.add_hline(y=base, line_dash="dash", line_color="red",
                              annotation_text=f"Pre-dam wet baseline: {base:,.0f} km²")
        if show_events:
            for _, t in EVENTS.items():
                fig.add_vline(x=int(t.year), line_dash="dot", line_color="crimson", opacity=0.5)
        fig.update_layout(height=420, title="Annual Flood (Aug–Sep)")
        fig.update_xaxes(dtick=1); fig.update_yaxes(title="km²")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: DUAL-POLARIZATION
# -----------------------------------------------------------------------------
with tab2:
    st.header("Dual-Polarization SAR: Hidden Inundation (VH over VV)")
    st.markdown("""
**Physics Recap**
- **VV**: specular reflection ⇒ open water → low backscatter (≈ −16 dB threshold)
- **VH**: volume/double-bounce ⇒ flooded vegetation under crop/forest → additional detection (≈ −22 to −25 dB)

**Why it matters**: VH reveals inundation in rice paddies/vegetated floodplains that VV often misses.
""")

    fig_vh_gain = load_plotly_from_assets("vh_gain", allow_key=True)
    if fig_vh_gain is not None:
        st.plotly_chart(fig_vh_gain, use_container_width=True)
    else:
        # Fallback: quick bar for selected AOI
        if "vh_gain_km2" in SEL.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=SEL["year"], y=SEL["vh_gain_km2"], name="VH−VV"))
            fig.update_layout(title=f"{aoi}: VH-only Additional Inundation", height=420)
            fig.update_xaxes(dtick=1); fig.update_yaxes(title="km²")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Notebook 05/07 to generate VH gain metrics.")

# -----------------------------------------------------------------------------
# TAB 3: TRENDS
# -----------------------------------------------------------------------------
with tab3:
    st.header(f"Decadal Trends — {aoi.replace('_',' ')}")

    fig = go.Figure()
    if show_vh and "flood_vh_km2" in SEL.columns:
        fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vh_km2"],
                                 mode="lines+markers", name="Flood (VH)",
                                 fill="tozeroy", fillcolor="rgba(59,130,246,0.12)"))
    if show_vv and "flood_vv_km2" in SEL.columns:
        fig.add_trace(go.Scatter(x=SEL["year"], y=SEL["flood_vv_km2"],
                                 mode="lines+markers", name="Flood (VV)"))
    if show_baseline:
        base = BASELINES.get(aoi, {}).get("wet_km2")
        if base:
            fig.add_hline(y=base, line_dash="dash", line_color="red",
                          annotation_text=f"Pre-dam wet baseline: {base:,.0f} km²")
    if show_events:
        for label, t in EVENTS.items():
            fig.add_vline(x=int(t.year), line_dash="dot", line_color="crimson", opacity=0.5,
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
        if "flood_vh_km2" in SEL.columns and SEL.shape[0] > 0:
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
        if "flood_vh_km2" in SEL.columns and SEL.shape[0] >= 3:
            x = SEL["year"].values
            y = SEL["flood_vh_km2"].values
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

# -----------------------------------------------------------------------------
# TAB 4: DROUGHT 2019–20
# -----------------------------------------------------------------------------
with tab4:
    st.header("2019–2020 Drought Crisis")

    fig_dry = load_plotly_from_assets("dry_biaxis", allow_key=True)
    if fig_dry is not None:
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
                    fig.add_vline(x=int(t.year), line_dash="dot", line_color="crimson", opacity=0.5)
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

# -----------------------------------------------------------------------------
# TAB 5: MAP (geemap folium backend; LOCAL GeoTIFF ONLY)
# -----------------------------------------------------------------------------
with tab5:
    st.header("Map Viewer (local GeoTIFF; no Earth Engine at runtime)")

    # Search roots for GeoTIFFs (expected & any *.tif)
    ROOTS = [
        SCRIPT_DIR / "outputs",
        SCRIPT_DIR.parent / "outputs",
        Path.cwd() / "outputs",
    ]
    tif_env = os.environ.get("MEKONG_TIF_DIR")
    if tif_env:
        ROOTS.insert(0, Path(tif_env).expanduser().resolve())

    expected = {
        "L5 MNDWI (Wet; 2005–2008)": "mndwi_wet_2005_2008.tif",
        "L5 MNDWI (Dry; 2005–2008)": "mndwi_dry_2005_2008.tif",
        "Water Mask (Wet; 2005–2008)": "watermask_wet_2005_2008.tif",
        "Water Mask (Dry; 2005–2008)": "watermask_dry_2005_2008.tif",
        "JRC Permanent Water (seasonality=12)": "jrc_perm_seasonality12.tif",
    }

    def first_existing(paths: List[Path]) -> Optional[Path]:
        for p in paths:
            if p.exists():
                return p
        return None

    # Find expected files
    found: Dict[str, Path] = {}
    for label, fname in expected.items():
        candidates = [r / fname for r in ROOTS]
        hit = first_existing(candidates)
        if hit:
            found[label] = hit

    # Also find ANY *.tif in the roots to avoid "No options to select"
    discovered: Dict[str, Path] = {}
    for root in ROOTS:
        if root.exists():
            for p in list(root.rglob("*.tif"))[:200]:  # limit to 200
                discovered[p.name] = p

    # UI
    left, right = st.columns([1.6, 1])
    with left:
        if not GEEMAP_AVAILABLE:
            st.warning("`geemap`가 없어 맵을 비활성화합니다. 설치: `pip install geemap streamlit-folium`")
        else:
            # Select from expected (if any) OR from discovered *.tif list
            st.markdown("**Layers**")
            mode = st.radio("Source", ["Expected set", "All *.tif under outputs"], horizontal=True)
            if mode == "Expected set":
                options = list(found.keys())
                chosen = st.multiselect("표시할 레이어 선택", options=options, default=options[:2])
                opacity = st.slider("레이어 불투명도", 0.10, 1.00, 0.85, 0.05)
                if len(chosen) == 0:
                    st.info("표시할 레이어를 선택해 주세요.")
                else:
                    try:
                        try:
                            m = FMap(center=(12.1, 105.2), zoom=6, ee_initialize=False, tiles="HYBRID")
                        except Exception:
                            m = FMap(center=(12.1, 105.2), zoom=6, ee_initialize=False, basemap="HYBRID")
                        for label in chosen:
                            path = found.get(label)
                            if path:
                                try:
                                    m.add_raster(str(path), layer_name=label, opacity=opacity)
                                except Exception as e:
                                    st.warning(f"{label} 추가 실패: {e}")
                        m.add_layer_control()
                        try:
                            m.to_streamlit(height=650)
                        except Exception:
                            if STF_AVAILABLE:
                                st_folium(m, height=650)
                            else:
                                st.error("지도를 표시하려면 `streamlit-folium`이 필요합니다. pip install streamlit-folium")
                    except Exception as e:
                        st.exception(e)
            else:
                # All discovered tif files
                options2 = sorted(discovered.keys())
                chosen2 = st.multiselect("표시할 GeoTIFF 파일 선택 (outputs/*)", options=options2, default=options2[:1])
                opacity2 = st.slider("레이어 불투명도", 0.10, 1.00, 0.85, 0.05, key="op2")
                if len(chosen2) == 0:
                    st.info("출력 폴더에 GeoTIFF가 없습니다. Notebook 01에서 export 하세요.")
                else:
                    try:
                        try:
                            m = FMap(center=(12.1, 105.2), zoom=6, ee_initialize=False, tiles="HYBRID")
                        except Exception:
                            m = FMap(center=(12.1, 105.2), zoom=6, ee_initialize=False, basemap="HYBRID")
                        for name in chosen2:
                            path = discovered.get(name)
                            if path:
                                try:
                                    m.add_raster(str(path), layer_name=name, opacity=opacity2)
                                except Exception as e:
                                    st.warning(f"{name} 추가 실패: {e}")
                        m.add_layer_control()
                        try:
                            m.to_streamlit(height=650)
                        except Exception:
                            if STF_AVAILABLE:
                                st_folium(m, height=650)
                            else:
                                st.error("지도를 표시하려면 `streamlit-folium`이 필요합니다. pip install streamlit-folium")
                    except Exception as e:
                        st.exception(e)

    with right:
        missing = sorted(set(expected.keys()) - set(found.keys()))
        if missing:
            st.warning("누락 파일: " + ", ".join(missing))
            st.markdown(
                "- Notebook 01에서 **GeoTIFF export** 셀을 실행해 `outputs/` 아래에 파일을 만드세요.\n"
                "- 또는 다른 위치에 있다면 `notebooks/outputs`나 프로젝트 루트 `outputs`로 복사하세요.\n"
                "- 커스텀 경로가 있으면 환경변수 `MEKONG_TIF_DIR` 로 지정할 수 있습니다."
            )
        else:
            st.success("모든 예상 GeoTIFF를 찾았습니다.")
        with st.expander("Debug (roots & discovered)"):
            st.write("Search roots:", [str(r) for r in ROOTS])
            st.write("Expected found:", {k: str(v) for k, v in found.items()})
            st.write("Sample discovered:", list(discovered.keys())[:15])

# -----------------------------------------------------------------------------
# TAB 6: METHODS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TAB 6: METHODS
# -----------------------------------------------------------------------------
with tab6:
    st.header("Methodology & Data")

    methods_md = (
        "**Data Sources**\n"
        "- COPERNICUS/S1_GRD (ESA) — VV/VH (12-day)\n"
        "- UCSB-CHG/CHIRPS/DAILY — Precipitation\n"
        "- LANDSAT/LT05/C02/T1_L2 — Baselines (MNDWI>0)\n"
        "- NASADEM — Topography context\n\n"
        "**SAR Classification (concept)**"
    )
    st.markdown(methods_md)

    st.code(
        "# Open water (VV specular)\n"
        "open_water = vv_db < -16  # dB\n\n"
        "# Flooded vegetation (VH + VV gate)\n"
        "flooded_veg = (vh_db < -25) & (vv_db > -16) & (vv_db < -10)\n",
        language="python",
    )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("MekongWatch | NASA Space Apps Challenge — Dual-Pol SAR for flood intelligence (2015–2024)")
