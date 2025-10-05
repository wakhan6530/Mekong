# 여기에 첫 번째 코드 전체 복사
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.environ.get("CAMBODIA_DID_OUT", "/mnt/data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TREAT = ["Banteay Meanchey", "Battambang", "Pursat", "Kampong Chhnang", "Kampong Thom", "Siem Reap"]
CTRL  = ["Prey Veng", "Takeo", "Svay Rieng", "Kandal", "Kampot", "Kampong Speu"]

PROD_2018_TREAT = {"Banteay Meanchey": 979.3, "Battambang": 1249.4, "Pursat": 536.2, "Kampong Chhnang": 594.8, "Kampong Thom": 864.9, "Siem Reap": 545.9}
PROD_2019_TREAT = {"Banteay Meanchey": 928.8, "Battambang": 1236.8, "Pursat": 527.5, "Kampong Chhnang": 640.3, "Kampong Thom": 869.6, "Siem Reap": 556.0}
PROD_2018_CTRL  = {"Prey Veng": 1424.2, "Takeo": 1179.9, "Svay Rieng": 548.0, "Kandal": 387.9, "Kampot": 493.7, "Kampong Speu": 354.3}
PROD_2019_CTRL  = {"Prey Veng": 1367.9, "Takeo": 1208.7, "Svay Rieng": 566.5, "Kandal": 358.6, "Kampot": 485.8, "Kampong Speu": 340.9}

AREA_2009_TREAT = {"Banteay Meanchey": 211.8, "Battambang": 265.0, "Pursat": 102.0, "Kampong Chhnang": 132.1, "Kampong Thom": 199.0, "Siem Reap": 195.1}
AREA_2011_TREAT = {"Banteay Meanchey": 219.1, "Battambang": 265.6, "Pursat": 114.8, "Kampong Chhnang": 136.3, "Kampong Thom": 199.3, "Siem Reap": 196.8}
AREA_2009_CTRL  = {"Prey Veng": 335.4, "Takeo": 260.6, "Svay Rieng": 180.5, "Kandal": 106.2, "Kampot": 133.9, "Kampong Speu": 110.8}
AREA_2011_CTRL  = {"Prey Veng": 361.3, "Takeo": 289.8, "Svay Rieng": 188.8, "Kandal": 103.5, "Kampot": 138.0, "Kampong Speu": 115.0}

YEARS_FISH = np.array([2014, 2015, 2016, 2017, 2018, 2019])
INLAND     = np.array([505.0, 487.9, 509.4, 527.8, 535.0, 478.9])
MARINE     = np.array([120.3, 120.5, 120.6, 121.0, 121.1, 122.3])
AQUA       = np.array([120.1, 143.1, 172.5, 207.4, 254.0, 307.4])

INLAND_LOTSMID = np.array([12.3, 13.4, 13.9, 16.6, 17.0, 14.3])
INLAND_FAMILY  = np.array([342.6, 339.3, 348.6, 352.5, 360.7, 326.5])
INLAND_RICEF   = np.array([150.1, 135.2, 146.8, 158.7, 157.3, 138.0])

def build_panel(d0, d1, provs, years, varname, unit, group):
    rows = []
    for p in provs:
        rows.append({"group": group, "province": p, "year": years[0], "variable": varname, "value": d0[p], "unit": unit})
        rows.append({"group": group, "province": p, "year": years[1], "variable": varname, "value": d1[p], "unit": unit})
    return pd.DataFrame(rows)

def compute_did(df, y0, y1, varname):
    d = df[df["variable"] == varname].copy()
    gy = d.groupby(["group","year"])["value"].mean().unstack()
    te, tl = float(gy.loc["Treated", y0]), float(gy.loc["Treated", y1])
    ce, cl = float(gy.loc["Control", y0]), float(gy.loc["Control", y1])
    return {"te": te, "tl": tl, "ce": ce, "cl": cl, "tchg": tl-te, "cchg": cl-ce, "did": (tl-te)-(cl-ce)}

def save_show(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# build data
df_prod_t = build_panel(PROD_2018_TREAT, PROD_2019_TREAT, TREAT, (2018, 2019), "Rice production", "thousand tons", "Treated")
df_prod_c = build_panel(PROD_2018_CTRL,  PROD_2019_CTRL,  CTRL,  (2018, 2019), "Rice production", "thousand tons", "Control")
df_area_t = build_panel(AREA_2009_TREAT, AREA_2011_TREAT, TREAT, (2009, 2011), "Harvested area", "thousand ha", "Treated")
df_area_c = build_panel(AREA_2009_CTRL,  AREA_2011_CTRL,  CTRL,  (2009, 2011), "Harvested area", "thousand ha", "Control")
df_all = pd.concat([df_prod_t, df_prod_c, df_area_t, df_area_c], ignore_index=True)

df_all.to_csv(os.path.join(OUTPUT_DIR, "cambodia_did_dataset.csv"), index=False)

did_prod = compute_did(df_all, 2018, 2019, "Rice production")
did_area = compute_did(df_all, 2009, 2011, "Harvested area")

# group means: rice production
import numpy as np
x = np.arange(2); width=0.35
plt.figure(figsize=(7.8,5.6))
plt.bar(x-width/2, [did_prod["te"], did_prod["tl"]], width, label="Treated (Tonle Sap provinces)")
plt.bar(x+width/2, [did_prod["ce"], did_prod["cl"]], width, label="Control (comparison provinces)")
plt.xticks(x, ["2018","2019"])
plt.ylabel("Rice production (thousand tons)")
plt.title("DID Setup: Rice Production — 2018 vs 2019 (Group Means)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
ymax = max([did_prod["te"], did_prod["tl"], did_prod["ce"], did_prod["cl"]])
for i,v in enumerate([did_prod["te"], did_prod["tl"]]):
    plt.text(x[i]-width/2, v + ymax*0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
for i,v in enumerate([did_prod["ce"], did_prod["cl"]]):
    plt.text(x[i]+width/2, v + ymax*0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
save_show(os.path.join(OUTPUT_DIR, "DID_rice_means_2018_2019_script.png"))

# group means: harvested area
plt.figure(figsize=(7.8,5.6))
plt.bar(x-width/2, [did_area["te"], did_area["tl"]], width, label="Treated (Tonle Sap provinces)")
plt.bar(x+width/2, [did_area["ce"], did_area["cl"]], width, label="Control (comparison provinces)")
plt.xticks(x, ["2009","2011"])
plt.ylabel("Harvested area (thousand ha)")
plt.title("DID Setup: Harvested Area — 2009 vs 2011 (Group Means)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
ymax = max([did_area["te"], did_area["tl"], did_area["ce"], did_area["cl"]])
for i,v in enumerate([did_area["te"], did_area["tl"]]):
    plt.text(x[i]-width/2, v + ymax*0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
for i,v in enumerate([did_area["ce"], did_area["cl"]]):
    plt.text(x[i]+width/2, v + ymax*0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
save_show(os.path.join(OUTPUT_DIR, "DID_area_means_2009_2011_script.png"))

# DID lines: rice
plt.figure(figsize=(7.8,5.6))
tvals = [did_prod["te"], did_prod["tl"]]
cvals = [did_prod["ce"], did_prod["cl"]]
plt.plot([2018,2019], tvals, marker="o", label="Treated")
plt.plot([2018,2019], cvals, marker="o", label="Control")
plt.title("DID Effect — Rice Production (Group Means)")
plt.xlabel("Year"); plt.ylabel("Rice production (thousand tons)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
ymin, ymax = min(tvals+cvals), max(tvals+cvals)
plt.text(2018.5, ymin + 0.05*(ymax-ymin), f"DID = (ΔTreated − ΔControl) = ({did_prod['tchg']:.1f} − {did_prod['cchg']:.1f}) = {did_prod['did']:.1f}", ha="center")
save_show(os.path.join(OUTPUT_DIR, "DID_rice_lines_2018_2019_script.png"))

# DID lines: area
plt.figure(figsize=(7.8,5.6))
tvals = [did_area["te"], did_area["tl"]]
cvals = [did_area["ce"], did_area["cl"]]
plt.plot([2009,2011], tvals, marker="o", label="Treated")
plt.plot([2009,2011], cvals, marker="o", label="Control")
plt.title("DID Effect — Harvested Area (Group Means)")
plt.xlabel("Year"); plt.ylabel("Harvested area (thousand ha)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
ymin, ymax = min(tvals+cvals), max(tvals+cvals)
plt.text(2010, ymin + 0.05*(ymax-ymin), f"DID = (ΔTreated − ΔControl) = ({did_area['tchg']:.1f} − {did_area['cchg']:.1f}) = {did_area['did']:.1f}", ha="center")
save_show(os.path.join(OUTPUT_DIR, "DID_area_lines_2009_2011_script.png"))

# Slope charts with end labels
def slope_chart_end_labels(d0, d1, title, x0, x1, unit, fname):
    plt.figure(figsize=(8.2,6.0))
    provs = sorted(list(d0.keys()), key=lambda p: d0[p])
    for p in provs:
        y0, y1 = d0[p], d1[p]
        plt.plot([0,1], [y0, y1], marker="o")
        plt.annotate(p, xy=(1, y1), xytext=(4, 4), textcoords="offset points", ha="left", va="bottom", fontsize=9)
    plt.xticks([0,1], [x0, x1])
    plt.ylabel(unit); plt.title(title)
    save_show(os.path.join(OUTPUT_DIR, fname))

slope_chart_end_labels(PROD_2018_TREAT, PROD_2019_TREAT, "Treated Provinces — Rice Production Change (2018→2019)", "2018", "2019", "thousand tons", "slope_treated_rice_2018_2019_script.png")
slope_chart_end_labels(PROD_2018_CTRL,  PROD_2019_CTRL,  "Control Provinces — Rice Production Change (2018→2019)", "2018", "2019", "thousand tons", "slope_control_rice_2018_2019_script.png")

# Fishing charts
def fish_by_source(years, inland, marine, aqua, fname):
    plt.figure(figsize=(8.6,5.8))
    plt.plot(years, inland, marker="o", label="Inland fisheries")
    plt.plot(years, marine, marker="o", label="Marine fisheries")
    plt.plot(years, aqua,   marker="o", label="Aquaculture")
    plt.title("Cambodia Fish Production by Source (2014–2019)")
    plt.xlabel("Year"); plt.ylabel("Production (thousand tonnes)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    for yvals, name in [(inland, "Inland"), (marine, "Marine"), (aqua, "Aquaculture")]:
        plt.annotate(f"{name}: {yvals[-1]:.1f}", xy=(years[-1], yvals[-1]), xytext=(6, 4), textcoords="offset points", ha="left", va="bottom", fontsize=9)
    save_show(os.path.join(OUTPUT_DIR, fname))

def fish_inland_composition(years, lotsmid, family, ricef, fname):
    plt.figure(figsize=(8.6,5.8))
    plt.plot(years, lotsmid, marker="o", label="Fishing lots & middle scale")
    plt.plot(years, family,  marker="o", label="Family fisheries")
    plt.plot(years, ricef,   marker="o", label="Rice-field fisheries")
    plt.title("Inland Fisheries Composition (2014–2019)")
    plt.xlabel("Year"); plt.ylabel("Production (thousand tonnes)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    for yvals, name in [(lotsmid, "Lots/mid"), (family, "Family"), (ricef, "Rice-field")]:
        plt.annotate(f"{name}: {yvals[-1]:.1f}", xy=(years[-1], yvals[-1]), xytext=(6, 4), textcoords="offset points", ha="left", va="bottom", fontsize=9)
    save_show(os.path.join(OUTPUT_DIR, fname))

def fish_delta_2019_2018(inland, marine, aqua, fname):
    cats = ["Inland", "Marine", "Aquaculture"]
    change = np.array([inland[-1]-inland[-2], marine[-1]-marine[-2], aqua[-1]-aqua[-2]])
    plt.figure(figsize=(8.0,5.6))
    x = np.arange(len(cats))
    plt.bar(x, change)
    plt.xticks(x, cats)
    plt.axhline(0, linewidth=0.8)
    plt.title("Change from 2018 to 2019 in Fish Production")
    plt.ylabel("Δ (thousand tonnes)")
    ymax = max(abs(change))*1.1 if max(abs(change))>0 else 1
    for i, v in enumerate(change):
        plt.text(i, v + (0.03*ymax if v>=0 else -0.07*ymax), f"{v:+.1f}", ha="center", va="bottom" if v>=0 else "top", fontsize=10)
    save_show(os.path.join(OUTPUT_DIR, fname))

fish_by_source(YEARS_FISH, INLAND, MARINE, AQUA, "fishing_by_source_2014_2019_script.png")
fish_inland_composition(YEARS_FISH, INLAND_LOTSMID, INLAND_FAMILY, INLAND_RICEF, "inland_composition_2014_2019_script.png")
fish_delta_2019_2018(INLAND, MARINE, AQUA, "fishing_change_2019_vs_2018_script.png")

with open(os.path.join(OUTPUT_DIR, "DID_summary_script.txt"), "w", encoding="utf-8") as f:
    f.write(
        f"Rice production DID: Treated Δ={did_prod['tchg']:.1f}, Control Δ={did_prod['cchg']:.1f}, DID={did_prod['did']:.1f}\n"
        f"Harvested area DID:  Treated Δ={did_area['tchg']:.1f}, Control Δ={did_area['cchg']:.1f}, DID={did_area['did']:.1f}\n"
    )