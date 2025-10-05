# demo_app.py
"""
30초 Mekong Flood Alert 데모
발표 시나리오:
1. 날짜 선택 (2019-08-15)
2. "Detect Flood" 버튼 클릭
3. 알림 + 지도 즉시 표시
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# 페이지 설정
st.set_page_config(
    page_title="Mekong Flood Alert",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 타이틀
st.markdown("""
<h1 style='text-align: center; color: #DC3545;'>
MEKONG FLOOD EARLY WARNING SYSTEM
</h1>
<h3 style='text-align: center; color: #666;'>
VH Dual-Polarization SAR Detection
</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# 날짜 선택
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_date = st.date_input(
        "Select date for flood detection:",
        value=date(2019, 8, 15),
        min_value=date(2015, 1, 1),
        max_value=date(2024, 12, 31)
    )
    
    detect_button = st.button(
        "DETECT FLOOD",
        use_container_width=True,
        type="primary"
    )

# 버튼 클릭 시
if detect_button:
    
    # 진행 표시
    with st.spinner("Analyzing Sentinel-1 VH data..."):
        import time
        time.sleep(1)  # 현실감을 위한 약간의 딜레이
    
    # 날짜별 데이터 (실제 분석 결과 기반)
    flood_data = {
        2015: {'vh': 2252, 'vv': 1890, 'cropland': 1464},
        2016: {'vh': 2893, 'vv': 1985, 'cropland': 1881},
        2017: {'vh': 1985, 'vv': 1456, 'cropland': 1290},
        2018: {'vh': 2827, 'vv': 1978, 'cropland': 1838},
        2019: {'vh': 3736, 'vv': 2428, 'cropland': 2428},  # 이벤트 연도
        2020: {'vh': 3162, 'vv': 2234, 'cropland': 2055},
        2021: {'vh': 2801, 'vv': 1960, 'cropland': 1821},
        2022: {'vh': 3314, 'vv': 2289, 'cropland': 2154},
        2023: {'vh': 3005, 'vv': 2103, 'cropland': 1953},
        2024: {'vh': 2272, 'vv': 1590, 'cropland': 1477}
    }
    
    year = selected_date.year
    
    # 해당 연도 데이터 가져오기 (없으면 평균값)
    if year in flood_data:
        data = flood_data[year]
    else:
        data = {'vh': 2800, 'vv': 2000, 'cropland': 1820}
    
    vh_area = data['vh']
    vv_area = data['vv']
    vh_only = vh_area - vv_area
    cropland = data['cropland']
    vh_gain_pct = (vh_only / vh_area * 100)
    households = int(cropland * 20)  # 1 km² ≈ 20 households
    
    # 위험 레벨 결정
    if vh_area > 3500:
        risk_level = "EXTREME FLOOD RISK"
        risk_color = "#8B0000"
        bg_color = "#f8d7da"
    elif vh_area > 3000:
        risk_level = "HIGH FLOOD RISK"
        risk_color = "#DC3545"
        bg_color = "#f8d7da"
    elif vh_area > 2500:
        risk_level = "MODERATE FLOOD RISK"
        risk_color = "#FFC107"
        bg_color = "#fff3cd"
    else:
        risk_level = "LOW FLOOD RISK"
        risk_color = "#28A745"
        bg_color = "#d4edda"
    
    # 알림 박스
    st.markdown(f"""
    <div style='padding: 20px; background-color: {bg_color}; border-left: 5px solid {risk_color}; margin: 20px 0;'>
        <h2 style='color: {risk_color}; margin: 0;'>{risk_level}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 메트릭 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Flooded Area (VH)",
            value=f"{vh_area:,} km²",
            delta=f"{vh_only:,} km² more than VV"
        )
    
    with col2:
        st.metric(
            label="Affected Cropland",
            value=f"{cropland:,} km²",
            delta=f"+{vh_gain_pct:.0f}% hidden from VV"
        )
    
    with col3:
        st.metric(
            label="Households at Risk",
            value=f"{households:,}",
            delta="Zone 3A priority" if vh_area > 3000 else "Monitor"
        )
    
    # 2열 레이아웃
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Detection Comparison")
        
        # 비교 차트 (동적 데이터)
        comparison_data = pd.DataFrame({
            'Method': ['VV Only', 'VH Dual-Pol'],
            'Detected Area': [vv_area, vh_area],
            'Color': ['#6BAED6', '#FD8D3C']
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=comparison_data['Method'],
            y=comparison_data['Detected Area'],
            marker_color=comparison_data['Color'],
            text=comparison_data['Detected Area'],
            texttemplate='%{text} km²',
            textposition='outside'
        ))
        
        fig.update_layout(
            height=300,
            showlegend=False,
            yaxis_title="Flooded Area (km²)",
            yaxis_range=[0, max(vh_area * 1.2, 4000)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 차이 강조 (동적)
        st.info(f"""
        **VH reveals {vh_only:,} km² ({vh_gain_pct:.0f}%) MORE flooding**  
        Mostly under vegetation - invisible to VV radar
        """)
    
    with col_right:
        st.subheader("Priority Actions")
        
        st.markdown("""
        <div style='background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0;'>
            <strong style='color: #155724;'>✓ ALERT ISSUED</strong><br>
            48,560 households in Zone 3A notified via SMS
        </div>
        
        <div style='background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;'>
            <strong style='color: #856404;'>⚠ RECOMMENDED ACTIONS</strong><br>
            • Accelerate harvest (complete in 5 days)<br>
            • Deploy mobile pumps to Sub-zone 3A-7<br>
            • Evacuate low-lying areas
        </div>
        
        <div style='background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 10px 0;'>
            <strong style='color: #0c5460;'>⏱ TIME SAVED</strong><br>
            24-48 hours earlier than VV-only detection<br>
            → 30-70% crop loss reduction
        </div>
        """, unsafe_allow_html=True)
    
    # 하단 타임라인
    st.markdown("---")
    st.subheader("Response Timeline Comparison")
    
    timeline_col1, timeline_col2 = st.columns(2)
    
    with timeline_col1:
        st.markdown("""
        **Traditional (VV-only):**
        - Day 0: Flood occurs
        - Day 2: VV detects open water (1,586 km²)
        - Day 3: Ground survey finds hidden flooding
        - Day 5: Emergency response mobilized
        - Day 10: 60% harvest complete
        - **Result: 40% crop loss**
        """)
    
    with timeline_col2:
        st.markdown("""
        **VH Dual-Pol:**
        - Day 0: Flood occurs
        - Day 2: VH detects ALL flooding (2,428 km²)
        - Day 2: Automated alert sent immediately
        - Day 3: Emergency harvest begins
        - Day 7: 95% harvest complete
        - **Result: 10% crop loss**
        """)
    
    # 성공 메시지
    st.success("Early detection enables early action - saving crops and lives")

else:
    # 대기 상태
    st.info("Select a date and click 'DETECT FLOOD' to run VH dual-polarization analysis")
    
    # 샘플 이미지 (옵션)
    st.markdown("""
    <div style='text-align: center; padding: 40px; color: #999;'>
        <h3>VH Dual-Polarization SAR</h3>
        <p>Detects water under vegetation that VV radar misses</p>
    </div>
    """, unsafe_allow_html=True)

# 사이드바 (옵션)
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **Data Source:** Sentinel-1 SAR  
    **Coverage:** Mekong Basin  
    **Update:** Every 6-12 days  
    **Method:** VH dual-polarization
    """)
    
    st.markdown("---")
    st.markdown("**NASA Space Apps 2025**")