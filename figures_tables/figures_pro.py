"""Study-area map (Fig. 1) and icon-based methodological framework (Fig. 2).
Icons: Font Awesome Free 6.6 (CC BY 4.0). Basemap: Natural Earth 1:110m (public domain)."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
from shapely.geometry import Point

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
from paths import MODEL as R, FIG as F, RAW, ROOT
F.mkdir(exist_ok=True)
import fontawesomefree  # pip install fontawesomefree
FA_TTF = Path(fontawesomefree.__file__).parent / "static/fontawesomefree/webfonts/fa-solid-900.ttf"
FA = fm.FontProperties(fname=str(FA_TTF))
CODES = json.load(open(HERE / "fa_codes.json"))
SERIF = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
                     "font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 400})


def icon(name):
    return chr(int(CODES[name], 16))


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(F / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================ coordinates and sample counts
def city_table():
    sd = pd.read_csv(RAW / "sludgedata/samples.csv")
    sd["city"] = sd.city_sample.str.split(",").str[0].str.strip()
    xy = sd.groupby("city")[["sample_longitude", "sample_latitude"]].median()
    d = pd.read_csv(ROOT / "data/processed/fs_model_table.csv")
    d = d[d.outlier_flag.fillna(0) == 0]
    t = d.groupby(["city", "country", "region"]).agg(n=("ts_gL", "size"), ts=("ts_gL", "median")).reset_index()
    t = t.merge(xy, left_on="city", right_index=True, how="left").rename(
        columns={"sample_longitude": "lon", "sample_latitude": "lat"})
    return t


# ============================================================ Fig. 1 study area map
def fig_map():
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[world.name != "Antarctica"]
    t = city_table()
    ssa = t[t.region == "SSA"].copy()
    ext = t[t.region == "external"].copy()
    ssa_iso = {"Zambia", "Uganda", "Tanzania", "Burkina Faso", "Ghana", "Kenya", "Sierra Leone", "South Africa"}
    ext_c = {"India", "Vietnam", "Lebanon", "Nepal", "Indonesia", "Thailand", "Uruguay"}

    land, edge, sea = "#EDEAE4", "#B9B4AA", "#F4F8FB"
    c_ssa, c_ext = "#B2182B", "#2166AC"
    fig = plt.figure(figsize=(7.4, 4.0))
    # ---- (a) world, Robinson
    ax = fig.add_axes([0.0, 0.40, 0.56, 0.55])
    rob = "ESRI:54030"
    w = world.to_crs(rob)
    ax.set_facecolor(sea)
    w.plot(ax=ax, color=land, edgecolor=edge, lw=0.25)
    w[w.name.isin(ssa_iso)].plot(ax=ax, color="#F2C9A8", edgecolor="#9C6B4E", lw=0.35)
    w[w.name.isin(ext_c)].plot(ax=ax, color="#C9DCEC", edgecolor="#5A7FA0", lw=0.35)
    # external sites aggregated per country so dots do not stack
    ec = ext.groupby("country").agg(n=("n", "sum"), lon=("lon", "median"), lat=("lat", "median")).reset_index()
    for df, col in ((ec, c_ext), (ssa, c_ssa)):
        g = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326").to_crs(rob)
        ax.scatter(g.geometry.x, g.geometry.y, s=6 + 2.2 * np.sqrt(g.n) ** 1.6, color=col, alpha=0.85,
                   edgecolor="white", lw=0.5, zorder=4)
    for _, r in ec.iterrows():
        p = gpd.GeoSeries([Point(r.lon, r.lat)], crs="EPSG:4326").to_crs(rob).iloc[0]
        dx, dy = {"Uruguay": (6e5, -9e5), "Lebanon": (-1.9e6, 6e5), "Nepal": (-1.1e6, 9e5),
                  "India": (-0.9e6, -2.0e6), "Vietnam": (6e5, 5e5), "Thailand": (5e5, -1.0e6),
                  "Indonesia": (5e5, -1.0e6)}.get(r.country, (5e5, 5e5))
        ax.text(p.x + dx, p.y + dy, f"{r.country} ({r.n})", fontsize=6.2, color="#1B3A57")
    # Africa frame
    ab = gpd.GeoSeries.from_xy([-20, 52], [-36, 22], crs="EPSG:4326").to_crs(rob)
    ax.add_patch(Rectangle((ab.x[0], ab.y[0]), ab.x[1] - ab.x[0], ab.y[1] - ab.y[0], fill=False, ec="#333333",
                           lw=0.8, ls="--", zorder=5))
    ax.text(ab.x[0], ab.y[1] + 4e5, "(b)", fontsize=7, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#999999"); s.set_lw(0.5)
    ax.set_title("(a) Data sources: 1,390 samples, 33 cities, 15 countries", fontsize=8.5, loc="left", weight="bold")
    # legend (a)
    h = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=c_ssa, mec="white", label="Sub-Saharan Africa: training and leave-one-city-out testing (8 cities)"),
         plt.Line2D([], [], marker="o", ls="", ms=6, mfc=c_ext, mec="white", label="External test sites, aggregated by country (25 sites)")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, -0.01), frameon=False, fontsize=6.6, handletextpad=0.3)

    # ---- (b) Africa close-up, Lambert azimuthal equal area centred on Africa
    axb = fig.add_axes([0.57, 0.02, 0.43, 0.93])
    laea = "+proj=laea +lat_0=0 +lon_0=20 +datum=WGS84 +units=m"
    af = world[world.continent == "Africa"].to_crs(laea)
    others = world[world.continent.isin(["Asia", "Europe"])].to_crs(laea)
    axb.set_facecolor(sea)
    others.plot(ax=axb, color="#F3F1EC", edgecolor=edge, lw=0.2)
    af.plot(ax=axb, color=land, edgecolor=edge, lw=0.3)
    af[af.name.isin(ssa_iso)].plot(ax=axb, color="#F2C9A8", edgecolor="#9C6B4E", lw=0.45)
    g = gpd.GeoDataFrame(ssa, geometry=[Point(xy) for xy in zip(ssa.lon, ssa.lat)], crs="EPSG:4326").to_crs(laea)
    norm = LogNorm(vmin=5, vmax=200)
    cmap = plt.get_cmap("YlOrBr")
    sc = axb.scatter(g.geometry.x, g.geometry.y, s=14 + 3.2 * g.n ** 0.85, c=g.ts, cmap=cmap, norm=norm,
                     edgecolor="#3B2A1A", lw=0.7, zorder=5)
    off = {"Lusaka": (-2.9e6, -1.2e6), "Kampala": (-2.7e6, 6.0e5), "Dar es Salaam": (1.5e5, -1.25e6),
           "Ouagadougou": (-1.6e6, 1.05e6), "Kumasi": (0.45e6, -1.5e6), "Naivasha": (6.5e5, 7.5e5),
           "Freetown": (-0.25e6, -2.2e6), "Pietermaritzburg": (4.5e5, -3.5e5)}
    for _, r in g.iterrows():
        dx, dy = off.get(r.city, (3e5, 3e5))
        axb.annotate(f"{r.city}\n$n$ = {r.n}, TS = {r.ts:.0f} g L$^{{-1}}$" if r.city != "Pietermaritzburg" else f"{r.city}\n$n$ = {r.n}",
                     xy=(r.geometry.x, r.geometry.y), xytext=(r.geometry.x + dx, r.geometry.y + dy),
                     fontsize=6.0, color="#222222", arrowprops=dict(arrowstyle="-", lw=0.45, color="#555555", shrinkA=0, shrinkB=4), zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    x0, x1 = -4.6e6, 4.4e6
    y0, y1 = -4.3e6, 4.3e6
    axb.set_xlim(x0, x1); axb.set_ylim(y0, y1)
    axb.set_xticks([]); axb.set_yticks([])
    for s in axb.spines.values():
        s.set_edgecolor("#999999"); s.set_lw(0.5)
    axb.set_title("(b) Sub-Saharan African cities", fontsize=8.5, loc="left", weight="bold")
    # scale bar 1000 km
    sx, sy = x0 + 0.5e6, y0 + 0.45e6
    axb.add_patch(Rectangle((sx, sy), 1e6, 0.7e5, fc="#333333", ec="#333333", lw=0.4))
    axb.add_patch(Rectangle((sx + 1e6, sy), 1e6, 0.7e5, fc="white", ec="#333333", lw=0.4))
    for k, lab in ((0, "0"), (1e6, "1000"), (2e6, "2000 km")):
        axb.text(sx + k, sy + 1.6e5, lab, ha="center", fontsize=5.8)
    # north arrow
    nx, ny = x1 - 0.55e6, y1 - 1.25e6
    axb.add_patch(Polygon([[nx, ny + 6e5], [nx - 1.7e5, ny], [nx, ny + 1.6e5]], closed=True, fc="#333333", ec="#333333", lw=0.4))
    axb.add_patch(Polygon([[nx, ny + 6e5], [nx + 1.7e5, ny], [nx, ny + 1.6e5]], closed=True, fc="white", ec="#333333", lw=0.4))
    axb.text(nx, ny + 6.8e5, "N", ha="center", fontsize=7, weight="bold")
    # colour bar + size legend
    cax = fig.add_axes([0.035, 0.17, 0.24, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[5, 10, 20, 50, 100, 200])
    cb.ax.set_xticklabels(["5", "10", "20", "50", "100", "200"], fontsize=6.3)
    cb.set_label("(b) Median total solids, g L$^{-1}$ (log scale)", fontsize=6.6)
    sax = fig.add_axes([0.31, 0.11, 0.24, 0.12]); sax.axis("off")
    for xx, n in zip([0.08, 0.38, 0.78], [30, 150, 450]):
        sax.scatter(xx, 0.62, s=14 + 3.2 * n ** 0.85, color="white", edgecolor="#3B2A1A", lw=0.7)
        sax.text(xx, -0.32, f"$n$ = {n}", ha="center", fontsize=6.2)
    sax.set_xlim(-0.05, 1.0); sax.set_ylim(-0.4, 1.2)
    sax.text(0.47, 1.12, "(b) Samples per city", ha="center", fontsize=6.6)
    fig.text(0.0, -0.005, "Basemap: Natural Earth (public domain). Projections: (a) Robinson; (b) Lambert azimuthal equal-area.",
             fontsize=5.6, color="#666666")
    save(fig, "fig1_study_area_map")


# ============================================================ Fig. 2 methodological framework
def fig_framework():
    fig = plt.figure(figsize=(7.4, 4.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
    stages = [
        ("1", "Open data", "#1F4E79", "#E8EFF7", [
            ("database", "Seven open sources", "Eawag, openwashdata, GFSD"),
            ("toilet", "1,390 samples", "pit latrines, septic tanks"),
            ("earth-africa", "33 cities, 15 countries", "8 SSA cities + 25 sites"),
            ("flask-vial", "Laboratory targets", "TS, VS, COD, NH$_4^+$-N")]),
        ("2", "Harmonisation", "#2E7D6B", "#E6F3EF", [
            ("filter", "QC and de-duplication", "units, outliers, duplicates"),
            ("clipboard-list", "14 survey predictors", "containment, users, water, age"),
            ("vial", "Field probes", "pH, electrical conductivity"),
            ("cloud-sun-rain", "City covariates", "climate, density, groundwater")]),
        ("3", "Models and validation", "#A65E1F", "#FBF0E4", [
            ("diagram-project", "Seven models", "baselines, RF, XGBoost, GP"),
            ("house", "Within-city random CV", "as in prior single-city work"),
            ("city", "Leave-one-city-out", ""),
            ("globe", "External testing", "India, Vietnam, Lebanon, other")]),
        ("4", "Planning outputs", "#6A4C93", "#F1ECF7", [
            ("magnifying-glass-chart", "City offsets", "0.5–7× systematic bias"),
            ("shield-halved", "Conformal intervals", "90% coverage in new cities"),
            ("chart-column", "SHAP drivers", "toilet type, sludge age, pit"),
            ("list-check", "Local sampling rule", "k = 0–40 samples per city")]),
    ]
    W, gap, x = 22.6, 2.2, 1.0
    top, bot = 56.5, 9.5
    for si, (num, title, dark, light, items) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x, bot), W, top - bot, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=light, ec=dark, lw=0.9))
        ax.add_patch(FancyBboxPatch((x, top - 6.2), W, 6.2, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=dark, ec=dark, lw=0.9))
        ax.add_patch(Rectangle((x, top - 6.2), W, 2.0, fc=dark, ec=dark, lw=0))
        ax.add_patch(Circle((x + 3.2, top - 3.1), 2.05, fc="white", ec="none"))
        ax.text(x + 3.2, top - 3.18, num, ha="center", va="center", fontsize=9, weight="bold", color=dark)
        ax.text(x + 6.2, top - 3.1, title, ha="left", va="center", fontsize=8.0, weight="bold", color="white")
        y = top - 11.6
        for name, head, sub in items:
            ax.add_patch(Circle((x + 4.0, y), 3.05, fc="white", ec=dark, lw=0.9))
            ax.text(x + 4.0, y - 0.05, icon(name), fontproperties=FA, fontsize=10.5, ha="center", va="center", color=dark)
            ax.text(x + 8.1, y + 1.05, head, ha="left", va="center", fontsize=6.9, weight="bold", color="#1A1A1A")
            ax.text(x + 8.1, y - 1.65, sub, ha="left", va="center", fontsize=5.9, color="#444444")
            y -= 9.6
        if si < 3:
            cx = x + W + gap / 2
            ax.add_patch(Polygon([[cx - 0.9, 34.5], [cx + 1.1, 33], [cx - 0.9, 31.5]], closed=True, fc="#555555", ec="none"))
        x += W + gap
    # leave-one-city-out schematic inside stage 3
    sx0 = 1.0 + 2 * (W + gap) + 8.1
    cities = ["LUN", "KLA", "DAR", "OUA", "KMS", "NVS", "FNA"]
    for i, c in enumerate(cities):
        test = i == 2
        ax.add_patch(FancyBboxPatch((sx0 + i * 1.75, 24.05 - 0.8), 1.5, 1.6, boxstyle="round,pad=0,rounding_size=0.35",
                                    fc="#A65E1F" if test else "#D9C3AC", ec="none"))
    ax.text(sx0 + 7 * 1.75 + 0.2, 24.05, "test", fontsize=5.4, va="center", color="#A65E1F", style="italic")
    # bottom band: link to treatment and resource recovery
    ax.add_patch(FancyBboxPatch((1.0, 1.0), 98.0, 6.4, boxstyle="round,pad=0,rounding_size=1.4", fc="#F6F6F4", ec="#888888", lw=0.7))
    for bx, (nm, txt) in zip([3.0, 41.0, 71.0], [("recycle", "Treatment sizing and resource recovery"),
                                   ("ruler-combined", "Design for distributions, not single values"),
                                   ("arrows-rotate", "Report leave-one-city-out performance")]):
        ax.text(bx, 4.2, icon(nm), fontproperties=FA, fontsize=9.5, va="center", color="#555555")
        ax.text(bx + 3.6, 4.2, txt, fontsize=6.8, va="center", color="#222222")
    ax.text(99, 59.6, "Icons: Font Awesome Free (CC BY 4.0)", ha="right", fontsize=5.2, color="#888888")
    ax.text(1.0, 59.6, "Do fecal sludge predictions transfer to a new city?", ha="left", fontsize=8.8, weight="bold", color="#1A1A1A")
    save(fig, "fig2_framework")


if __name__ == "__main__":
    fig_map(); fig_framework()
    print("ok")
