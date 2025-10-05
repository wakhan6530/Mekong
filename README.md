# NASA Mekong Hackathon 2025
메콩강 유역 댐 영향 종합 분석

## 🎯 프로젝트 목표
상류 댐 건설이 하류 지역에 미치는 **환경적·경제적·사회적 영향** 통합 분석

---

## 📂 프로젝트 구조

### 1️⃣ Environmental Analysis (환경 영향)
**Path**: `1_Environmental_Analysis/`  
**작성자**: Parkspace

**분석 내용**:
- 🛰️ Sentinel-1 SAR 홍수 탐지
- 💧 CHIRPS 강수량 상관관계
- 📊 Mekong Delta & Tonle Sap 침수 변화

**주요 발견**:
- VH 편파로 30% 추가 침수 탐지
- 강수량-홍수 상관관계 약함 (r~0.3) → **댐 영향 시사**
- 2019년 Jinghong 댐 방류 사건 감지

**실행**:
```bash
cd 1_Environmental_Analysis
jupyter notebook notebooks/01_data_acquisition.ipynb