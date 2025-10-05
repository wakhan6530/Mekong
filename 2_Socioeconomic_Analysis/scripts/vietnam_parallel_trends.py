# 여기에 두 번째 코드 전체 복사


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# ======================
# 0) 설정
# ======================
DATA_PATH = "/mnt/data/vietnam_panel.csv"   # <-- 파일 경로 필요시 수정
BASELINES = [2010, 2020]                    # 기준연도
WINDOW = 5                                  

# 제외할 베트남 지역구 (집계구라서 제외했음)
EXCLUDE_PROVINCES: List[str] = [
    "WHOLE COUNTRY",
    "Northern midlands and mountain areas",
    "Red River Delta",
    "Northern Central area and Central coastal area",
    "Central Highlands",
    "Mekong River Delta",
]

# Treated 지역 리스트 
TREATED_PROVINCES: List[str] = [
    "Ha Noi", "Vinh Phuc", "Bac Ninh", "Thai Binh", "Ha Nam", "Nam Dinh",
    "Lao Cai", "Yen Bai", "Phu Tho", "Dien Bien", "Lai Chau", "Son La",
    "Kon Tum", "Gia Lai", "Dak Lak", "Dak Nong",
    "Long An", "Tien Giang", "Ben Tre", "Tra Vinh", "Vinh Long",
    "Dong Thap", "An Giang", "Kien Giang", "Can Tho", "Hau Giang",
    "Soc Trang", "Bac Lieu", "Ca Mau"
]

# 열 이름 표준화
RENAME_MAP = {
    "cultivation farm": "cultivationfarm",
    "industrial production": "industrialproduction",
    "newly established enterprises": "newlyestablishedenterprises",
    "nonfirm employees": "nonfirmemployees",
    "nonfirm estb": "nonfirmestb",
    "net immigration": "netimmigration",
    "total farm": "totalfarm",
}

# 영향을 보고자하는 Outcome variables
FOCUS_OUTCOMES = [
    "cultivationfarm","livestock","fishing","others","industrialproduction",
    "newlyestablishedenterprises","nonfirmemployees","nonfirmestb","netimmigration"
]

# 출력 폴더
OUT_DIR = Path("/mnt/data/pt_prepost_window5_plots_regen")

# ======================
# 1) 데이터 로드/정리
# ======================
df = pd.read_csv(DATA_PATH)
df = df.rename(columns=RENAME_MAP)
df.columns = [c.strip() for c in df.columns]

# 필수 열 
assert "province" in df.columns and "year" in df.columns, "province/year 열이 필요합니다."

# 기본 전처리
df["province"] = df["province"].astype(str).str.strip()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["province","year"]).copy()
df["year"] = df["year"].astype(int)

# 집계 권역 제외
df = df[~df["province"].isin(EXCLUDE_PROVINCES)].copy()

# Treated 
df["treated"] = df["province"].isin(TREATED_PROVINCES).astype(int)

# 숫자형으로 추출되도록 설정. 
id_cols = {"province","year","treated"}
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outcomes = [c for c in FOCUS_OUTCOMES if c in num_cols and c not in id_cols]

# ======================
# 2) 함수
# ======================
def yearly_group_stats(dfin: pd.DataFrame, outcome: str, treated_flag: int) -> pd.DataFrame:
    """
    특정 outcome에 대해 (treated/control)×연도별 평균/표준오차/95% CI 계산
    """
    tmp = dfin[["province","year","treated", outcome]].dropna().copy()
    tmp = tmp[tmp["treated"] == treated_flag]
    g = (tmp.groupby("year")[outcome]
           .agg(['mean','count','std'])
           .rename(columns={'mean':'ybar','count':'n','std':'sd'})
           .reset_index())
    # 표준오차 + 95% CI (집단평균의 표준오차: sd/sqrt(n))
    g["se"] = g["sd"] / np.sqrt(g["n"]).replace(0, np.nan)
    z = 1.96
    g["lci"] = g["ybar"] - z*g["se"]
    g["uci"] = g["ybar"] + z*g["se"]
    g["group"] = "Treated" if treated_flag==1 else "Control"
    return g

# ======================
# 3) 그래프 생성
# ======================
OUT_DIR.mkdir(parents=True, exist_ok=True)
index_rows = []

for y in outcomes:
    gt = yearly_group_stats(df, y, 1)
    gc = yearly_group_stats(df, y, 0)
    merged = pd.concat([gt, gc], ignore_index=True)
    if merged.empty:
        continue

    for t0 in BASELINES:
        # ±WINDOW 내 연도만 사용
        sub = merged[(merged["year"] >= t0 - WINDOW) & (merged["year"] <= t0 + WINDOW)].copy()
        if sub.empty:
            continue

        # 정규화 기준 연도: t0가 양 집단에 모두 있으면 t0, 아니면 t0-1 중 가장 가까운 과거 연도
        has_t0_t = not sub[(sub["group"]=="Treated") & (sub["year"]==t0)].empty
        has_t0_c = not sub[(sub["group"]=="Control") & (sub["year"]==t0)].empty
        if has_t0_t and has_t0_c:
            base_year = t0
        else:
            pre_years = sub[sub["year"] < t0]["year"]
            if pre_years.empty:
                # 사전연도가 없다면 건너뜀
                continue
            base_year = int(pre_years.max())

        # 기준 시점의 집단 평균
        base_t = sub[(sub["group"]=="Treated") & (sub["year"]==base_year)]["ybar"]
        base_c = sub[(sub["group"]=="Control") & (sub["year"]==base_year)]["ybar"]
        norm_ok = (not base_t.empty) and (not base_c.empty)
        bt = float(base_t.iloc[0]) if norm_ok else np.nan
        bc = float(base_c.iloc[0]) if norm_ok else np.nan

        # ----- 그래프 -----
        fig, ax = plt.subplots(figsize=(8, 4.7))
        for grp, gsub in sub.groupby("group"):
            gsub = gsub.sort_values("year")
            x = gsub["year"].to_numpy(dtype=float)
            ybar = gsub["ybar"].to_numpy(dtype=float)
            lci = gsub["lci"].to_numpy(dtype=float)
            uci = gsub["uci"].to_numpy(dtype=float)

            # 정규화(기준=100)
            if norm_ok:
                if grp == "Treated" and bt != 0:
                    ybar, lci, uci = 100*ybar/bt, 100*lci/bt, 100*uci/bt
                if grp == "Control" and bc != 0:
                    ybar, lci, uci = 100*ybar/bc, 100*lci/bc, 100*uci/bc

            # 95% CI 밴드 + 평균선
            ax.fill_between(x, lci, uci, alpha=0.12, label=f"{grp} 95% CI" if grp=="Treated" else None)
            ax.plot(x, ybar, marker='o', label=f"{grp} mean")

        # 기준연도 표시
        ax.axvline(float(t0), linestyle='--')
        # 라벨/제목
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{y} (index=100 at {base_year})" if norm_ok else f"{y} (level)")
        ax.set_title(f"Parallel Trends (±{WINDOW} yrs) — {y} around {t0}")
        ax.legend()
        fig.tight_layout()

        # 저장
        outpng = OUT_DIR / f"pt_prepost_{y}_{t0}_w{WINDOW}.png"
        fig.savefig(outpng, dpi=200, bbox_inches="tight")
        plt.close(fig)

        index_rows.append({
            "outcome": y,
            "baseline": t0,
            "plot_path": outpng.as_posix(),
            "normalized": norm_ok,
            "base_year": base_year
        })

# ======================
# 4)  CSV 저장
# ======================
pt_index = pd.DataFrame(index_rows).sort_values(["outcome","baseline"])
pt_index_path = OUT_DIR / "pt_prepost_index_regen.csv"
pt_index.to_csv(pt_index_path, index=False)

print("Saved index CSV to:", pt_index_path.as_posix())
print("Total plots:", len(pt_index))
