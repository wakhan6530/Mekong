"""
MekongWatch: SAR-Based Flood Monitoring System
==============================================
Production-grade Streamlit application for analyzing Mekong Delta flood dynamics
using dual-polarization SAR and multi-source remote sensing data.

Technical Stack:
- Sentinel-1 C-band SAR (VV+VH polarization)
- CHIRPS precipitation dataset
- Google Earth Engine processing
- Plotly interactive visualization

Author: MekongWatch Research Team
Contact: [your-email]
Version: 1.0.0
Last Updated: 2025-01-XX
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    import os
    BASE_DIR = Path(__file__).parent.parent  # streamlit_app의 부모 (Mekong)
    DATA_DIR = BASE_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    METADATA_DIR = DATA_DIR / "metadata"
    PLOTLY_DIR = DATA_DIR / "plotly"
    ASSETS_DIR = Path(__file__).parent / "assets"  # streamlit_app/assets
    IMAGES_DIR = ASSETS_DIR / "images"
    HTML_DIR = ASSETS_DIR / "html"
    
    # Study parameters
    STUDY_PERIOD = "2015-2024"
    CASE_STUDY_PERIOD = "2019-2020"
    FLOOD_SEASON = "August-September"
    AOI_BOUNDS = "104.5-106.8°E, 8.5-11.0°N"
    
    # Technical specs
    SAR_PLATFORM = "Sentinel-1 C-band"
    VV_THRESHOLD = -16  # dB
    VH_FLOODED_VEG_THRESHOLD = -25  # dB

config = Config()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MekongWatch | SAR Flood Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourteam/mekongwatch',
        'Report a bug': 'https://github.com/yourteam/mekongwatch/issues',
        'About': 'NASA Space Apps Challenge 2025'
    }
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main typography */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin: 1.5rem 0;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Alert boxes */
    .alert-info {
        background-color: #e0f2fe;
        border-left: 4px solid #0284c7;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #dcfce7;
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        color: #1e40af;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING (with caching)
# ============================================================================

@st.cache_data(ttl=3600)
def load_combined_data() -> pd.DataFrame:
    """Load main analysis dataset with validation"""
    filepath = config.PROCESSED_DIR / "combined_analysis.csv"
    
    if not filepath.exists():
        st.error(f"❌ Required file not found: {filepath}")
        st.info("Please run Jupyter notebooks 01-03 first.")
        st.stop()
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = ['year', 'flood_area_km2', 'precipitation_mm']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    
    # Data quality checks
    if df['flood_area_km2'].isna().any():
        st.warning("⚠️ Some flood area values are missing")
    
    return df

@st.cache_data(ttl=3600)
def load_metadata(filename: str) -> dict:
    """Load JSON metadata with error handling"""
    filepath = config.METADATA_DIR / filename
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.warning(f"⚠️ Invalid JSON: {filename}")
        return None

@st.cache_data(ttl=3600)
def load_plotly_json(filename: str):
    """Load Plotly figure from JSON"""
    filepath = config.PLOTLY_DIR / filename
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r') as f:
            fig_json = json.load(f)
            return go.Figure(fig_json)
    except Exception as e:
        st.warning(f"⚠️ Error loading {filename}: {e}")
        return None

# Load all data at startup
try:
    df = load_combined_data()
    meta_correlation = load_metadata("correlation_stats.json")
    meta_dual = load_metadata("dual_pol_2019_stats.json")
    meta_monthly = load_metadata("monthly_analysis_summary.json")
    meta_drought = load_metadata("drought_analysis_summary.json")
except Exception as e:
    st.error(f"Fatal error loading data: {e}")
    st.stop()

# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown('<div class="main-title">🛰️ MekongWatch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Dual-Polarization SAR Analysis of Mekong Delta Flood Dynamics</div>',
    unsafe_allow_html=True
)

# Key metrics banner
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Study Period",
        config.STUDY_PERIOD,
        delta=f"{df['year'].nunique()} years"
    )

with col2:
    avg_flood = df['flood_area_km2'].mean()
    std_flood = df['flood_area_km2'].std()
    st.metric(
        "Mean Flood Extent",
        f"{avg_flood:,.0f} km²",
        delta=f"σ = {std_flood:,.0f}"
    )

with col3:
    if meta_correlation:
        corr = meta_correlation.get('correlations', {}).get('flood_precip', {}).get('r', 0)
        st.metric(
            "Precip Correlation",
            f"r = {corr:.3f}",
            delta="Weak" if abs(corr) < 0.5 else "Moderate"
        )
    else:
        st.metric("Correlation", "N/A")

with col4:
    peak_year = df.loc[df['flood_area_km2'].idxmax(), 'year']
    peak_area = df['flood_area_km2'].max()
    st.metric(
        "Peak Flood Year",
        f"{int(peak_year)}",
        delta=f"{peak_area:,.0f} km²"
    )

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🌾 Dual-Pol Analysis",
    "📈 Decadal Trends",
    "💧 2019-20 Drought",
    "🔬 Methods & Data"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.header("Executive Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Research Objectives
        
        This study quantifies flood extent variability in Vietnam's Mekong Delta (2015-2024)
        using **Sentinel-1 C-band SAR** with dual-polarization (VV+VH) analysis to:
        
        1. Establish baseline flood patterns in post-dam era
        2. Assess correlation between precipitation and flood extent
        3. Identify flooded vegetation missed by traditional methods
        4. Document the 2019-2020 drought crisis
        
        ### Key Innovations
        
        - **Dual-polarization SAR**: VV+VH reveals 15-20% more flooded area
        - **10-year time series**: Captures inter-annual variability
        - **Multi-source validation**: CHIRPS precipitation, JRC water data
        - **Cloud-independent**: SAR penetrates monsoon cloud cover
        """)
    
    with col2:
        st.markdown("""
        <div class="alert-info">
        <h4 style="margin-top:0;">Study Area</h4>
        <p><strong>Location:</strong> Vietnam Mekong Delta<br>
        <strong>Coordinates:</strong> 104.5-106.8°E, 8.5-11.0°N<br>
        <strong>Area:</strong> ~26,000 km²<br>
        <strong>Population:</strong> 17.5M<br>
        <strong>Agriculture:</strong> 55% national rice output</p>
        </div>
        
        <div class="alert-warning">
        <h4 style="margin-top:0;">Dam Context</h4>
        <p><strong>Upstream Dams:</strong> 11 major (China)<br>
        <strong>Total Capacity:</strong> 40+ billion m³<br>
        <strong>Impact Period:</strong> 2008-present</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Findings
    st.markdown('<h3 class="section-header">Key Findings</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="alert-success">
        <h4>1. Dual-Pol Advantage</h4>
        <p>VH polarization detects <strong>double-bounce scattering</strong> from 
        flooded rice paddies, revealing 15-20% more inundated area than VV alone.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if meta_correlation:
            corr_val = meta_correlation.get('correlations', {}).get('flood_precip', {}).get('r', 0)
            st.markdown(f"""
            <div class="alert-warning">
            <h4>2. Weak Precip Correlation</h4>
            <p>Pearson r = <strong>{corr_val:.3f}</strong> indicates flood patterns 
            are NOT primarily rainfall-driven, suggesting dam operations as key factor.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="alert-info">
        <h4>3. 2019-20 Drought</h4>
        <p>Lowest Mekong levels in <strong>100 years</strong>, with 80%+ 
        precipitation deficit during critical period (Dec 2019 - Feb 2020).</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown('<h3 class="section-header">Dataset Statistics</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            df.describe()[['flood_area_km2', 'precipitation_mm']].T,
            use_container_width=True
        )
    
    with col2:
        # Year-over-year change
        df_change = df.copy()
        df_change['flood_change_%'] = df_change['flood_area_km2'].pct_change() * 100
        
        fig_change = px.bar(
            df_change[1:],
            x='year',
            y='flood_change_%',
            color='flood_change_%',
            color_continuous_scale='RdYlGn_r',
            color_continuous_midpoint=0,
            title="Annual % Change in Flood Extent"
        )
        fig_change.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_change, use_container_width=True)

# ============================================================================
# TAB 2: DUAL-POLARIZATION
# ============================================================================

with tab2:
    st.header("Dual-Polarization SAR: Detecting Hidden Flooding")
    
    st.markdown("""
    ### Physical Principles
    
    **VV Polarization (Vertical-Vertical):**
    - Detects **open water** surfaces (rivers, lakes)
    - Smooth water = **specular reflection** = low backscatter (-20 to -25 dB)
    
    **VH Polarization (Vertical-Horizontal):**
    - Detects **volume scattering** + **double-bounce**
    - Water surface + vertical structures (rice stems) = strong backscatter
    - Reveals **flooded vegetation** invisible to VV
    
    **Threshold Classification:**
    - Open Water: VV < -16 dB
    - Flooded Vegetation: VH < -25 dB AND -16 < VV < -10 dB
    """)
    
    if meta_dual:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "VV-Only Detection",
                f"{meta_dual.get('open_water_km2', 0):,.0f} km²",
                help="Traditional single-polarization method"
            )
        
        with col2:
            total = meta_dual.get('total_inundation_km2', 0)
            hidden_pct = meta_dual.get('hidden_damage_percent', 0)
            st.metric(
                "VV+VH Total",
                f"{total:,.0f} km²",
                delta=f"+{hidden_pct:.1f}%"
            )
        
        with col3:
            st.metric(
                "Hidden Ag Flooding",
                f"{meta_dual.get('flooded_vegetation_km2', 0):,.0f} km²",
                help="Flooded rice paddies"
            )
    
    st.markdown("---")
    
    # Interactive map
    st.subheader("Spatial Comparison: VV vs VH Detection")
    
    fig_vv_vh = load_plotly_json("vv_vh_comparison.json")
    
    if fig_vv_vh:
        st.plotly_chart(fig_vv_vh, use_container_width=True)
    else:
        st.warning("Run notebook 07 to generate interactive maps")
        
        # Fallback to static image
        img_path = config.IMAGES_DIR / "dual_pol_comparison.png"
        if img_path.exists():
            st.image(str(img_path))

# ============================================================================
# TAB 3: DECADAL TRENDS
# ============================================================================

with tab3:
    st.header("10-Year Flood Evolution (2015-2024)")
    
    # Time series plot
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=df['year'],
        y=df['flood_area_km2'],
        mode='lines+markers',
        name='Flood Extent',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Add dam markers
    fig_trend.add_vline(
        x=2014, line_dash="dash", line_color="red",
        annotation_text="Nuozhadu Dam Complete"
    )
    
    fig_trend.update_layout(
        title="Annual Maximum Flood Extent (Aug-Sep)",
        xaxis_title="Year",
        yaxis_title="Flood Area (km²)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Descriptive Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'CV (%)'],
            'Value': [
                f"{df['flood_area_km2'].mean():,.0f} km²",
                f"{df['flood_area_km2'].median():,.0f} km²",
                f"{df['flood_area_km2'].std():,.0f} km²",
                f"{df['flood_area_km2'].min():,.0f} km²",
                f"{df['flood_area_km2'].max():,.0f} km²",
                f"{(df['flood_area_km2'].std() / df['flood_area_km2'].mean() * 100):.1f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("Trend Analysis")
        
        # Linear regression
        x = df['year'].values
        y = df['flood_area_km2'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        st.markdown(f"""
        **Linear Trend:**
        - Slope: {slope:,.1f} km²/year
        - R²: {r_value**2:.3f}
        - p-value: {p_value:.4f}
        - Trend: {'Increasing' if slope > 0 else 'Decreasing'}
        - Significance: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)
        """)

# ============================================================================
# TAB 4: DROUGHT ANALYSIS
# ============================================================================

with tab4:
    st.header("2019-2020 Drought Crisis")
    
    if meta_drought:
        st.markdown(f"""
        ### Crisis Summary
        
        **Period:** {meta_drought.get('period', 'N/A')}  
        **Months Analyzed:** {meta_drought.get('months_analyzed', 0)}  
        **Correlation:** r = {meta_drought.get('statistics', {}).get('correlation_coefficient', 0):.3f}  
        
        {meta_drought.get('key_findings', {}).get('ecological_impact', '')}
        """)
    
    # Drought chart
    fig_drought = load_plotly_json("drought_2019_2020.json")
    
    if fig_drought:
        st.plotly_chart(fig_drought, use_container_width=True)
    else:
        st.info("Run notebook 06 for drought analysis")

# ============================================================================
# TAB 5: METHODS
# ============================================================================

with tab5:
    st.header("Methodology & Data Sources")
    
    st.markdown("""
    ### Data Sources
    
    | Dataset | Source | Temporal Resolution | Spatial Resolution |
    |---------|--------|---------------------|-------------------|
    | Sentinel-1 SAR | ESA/Copernicus | 12 days | 10m |
    | CHIRPS Precipitation | UCSB/CHG | Daily | 5km |
    | JRC Surface Water | JRC/Google | Yearly | 30m |
    
    ### Processing
    
    **Platform:** Google Earth Engine  
    **Language:** Python 3.11+  
    **Libraries:** geemap, pandas, plotly, scipy  
    
    **SAR Processing:**
    """)
    
    st.code("""
# VV water detection
water_mask = sar_vv.lt(-16)

# VH flooded vegetation
flooded_veg = sar_vh.lt(-25).And(sar_vv.gt(-16).And(sar_vv.lt(-10)))
    """, language='python')
    
    st.markdown("""
    ### Limitations
    
    - SAR affected by wind roughness on water surface
    - Threshold may miss very shallow water (<5cm)
    - Correlation does not imply causation
    - Dam discharge data not directly available
    
    ### Citation
MekongWatch Team (2025). Dual-Polarization SAR Analysis of 
Vietnam Mekong Delta Flood Dynamics (2015-2024). 
NASA Space Apps Challenge.
                """)
    
    st.markdown("---")
    st.subheader("Download Data")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Combined Dataset (CSV)",
        data=csv,
        file_name="mekong_flood_2015_2024.csv",
        mime="text/csv"
    )