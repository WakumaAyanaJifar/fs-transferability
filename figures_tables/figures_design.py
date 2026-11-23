"""Fig. 2 study design: framed, four-stage methods diagram (methods only, no results).
Colours follow the rest of the paper: SSA crimson / external blue (Fig. 1),
within-city blue / leave-one-city-out vermillion / green (Figs 3-5)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import figures_pro as fp  # city coordinates and sample counts

HERE = Path(__file__).parent
from paths import FIG as F
SERIF = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
                     "font.size": 7, "pdf.fonttype": 42})

# ---- paper palette
NAVY = "#1F3A5F"          # frame and stage headers
INK = "#111111"           # all text
SSA, EXT = "#B2182B", "#2166AC"            # Fig. 1
WITHIN, LOCO, GREEN = "#0072B2", "#D55E00", "#009E73"   # Figs 3-5
LAND, SSA_FILL = "#EDEAE4", "#F2C9A8"
TRAIN, IDLE = "#C9D3DE", "#F1F1F1"
TINT = "#F3F6FA"

W, H = 100, 74
fig = plt.figure(figsize=(7.48, 7.48 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def inset(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


def txt(x, y, s, **k):
    k.setdefault("color", INK); k.setdefault("fontsize", 6.6); k.setdefault("va", "center")
    return ax.text(x, y, s, **k)


def panel(x0, y0, x1, y1, num, head):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec=NAVY, lw=0.9, zorder=1))
    ax.add_patch(FancyBboxPatch((x0, y1 - 4.2), x1 - x0, 4.2, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=NAVY, ec=NAVY, lw=0.9, zorder=2))
    ax.add_patch(Rectangle((x0, y1 - 4.2), x1 - x0, 1.5, fc=NAVY, ec="none", zorder=2))
    ax.add_patch(Circle((x0 + 2.6, y1 - 2.1), 1.45, fc="white", ec="none", zorder=3))
    txt(x0 + 2.6, y1 - 2.1, num, ha="center", fontsize=7.6, weight="bold", color=NAVY, zorder=4)
    txt(x0 + 4.9, y1 - 2.1, head, fontsize=8.2, weight="bold", color="white", zorder=4)


def arrow(p0, p1, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=conn, arrowstyle="-|>", mutation_scale=11,
                                 lw=1.4, color=NAVY, zorder=5))


def dot(x, y, c, r=0.75):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=0.4, zorder=4))


# ================================================================ outer frame + question
ax.add_patch(FancyBboxPatch((0.6, 0.6), W - 1.2, H - 1.2, boxstyle="round,pad=0,rounding_size=1.6",
                            fc="white", ec=NAVY, lw=1.3))
ax.add_patch(FancyBboxPatch((2.2, 66.2), W - 4.4, 6.0, boxstyle="round,pad=0,rounding_size=1.0",
                            fc=TINT, ec="none"))
txt(4.0, 70.4, "Research question", fontsize=7.0, weight="bold", color=NAVY)
txt(4.0, 67.9, "Can survey-based machine-learning models predict fecal sludge characteristics "
               "in a city where no samples have been taken?", fontsize=8.0, style="italic")

L0, L1, R0, R1 = 2.2, 48.6, 51.4, 97.8
T0, T1 = 35.0, 64.2          # top row
B0, B1 = 2.2, 31.0           # bottom row

# ================================================================ 1 data
panel(L0, T0, L1, T1, "1", "Open data")
t = fp.city_table()
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
laea = "+proj=laea +lat_0=5 +lon_0=20 +datum=WGS84 +units=m"
af = world[world.continent == "Africa"].to_crs(laea)
m = inset(L0 + 0.8, T0 + 1.0, L0 + 20.5, T1 - 4.6)
af.plot(ax=m, color=LAND, edgecolor="#B9B4AA", lw=0.25)
ssa_iso = {"Zambia", "Uganda", "Tanzania", "Burkina Faso", "Ghana", "Kenya", "Sierra Leone", "South Africa"}
af[af.name.isin(ssa_iso)].plot(ax=m, color=SSA_FILL, edgecolor="#9C6B4E", lw=0.35)
s = t[t.region == "SSA"]
pts = gpd.GeoSeries(gpd.points_from_xy(s.lon, s.lat), crs=4326).to_crs(laea)
m.scatter(pts.x, pts.y, s=6 + 0.5 * s.n.values, color=SSA, edgecolor="white", lw=0.5, zorder=5)
m.set_xlim(-4.3e6, 4.6e6); m.set_ylim(-4.3e6, 4.4e6); m.set_aspect("equal"); m.axis("off")

x = L0 + 21.5
dot(x + 0.5, 58.0, SSA); txt(x + 1.8, 58.0, "Sub-Saharan Africa: 8 cities, 1,030 samples", weight="bold")
txt(x + 1.8, 56.0, "training and leave-one-city-out testing", fontsize=6.3)
dot(x + 0.5, 53.6, EXT); txt(x + 1.8, 53.6, "External: 25 sites, 7 countries, 360 samples", weight="bold")
txt(x + 1.8, 51.6, "India, Vietnam, Lebanon, Nepal, Indonesia, ...", fontsize=6.3)
txt(x, 48.6, "Seven open sources", weight="bold")
src = [("Ward et al. (2021)", 465), ("Global FS Database", 269), ("Englund et al. (2020)", 250),
       ("Andriessen et al. (2023)", 203), ("Prasad et al. (2021)", 164), ("Ward et al. (2023)", 20),
       ("Shaw et al. (2026)", 19)]
for i, (nm, n) in enumerate(src):
    yy = 46.5 - i * 1.55
    txt(x + 0.4, yy, nm, fontsize=6.2)
    ax.add_patch(Rectangle((x + 14.6, yy - 0.45), 0.0135 * n, 0.9, fc=NAVY, ec="none", zorder=3))
    txt(x + 15.0 + 0.0135 * n, yy, f"{n}", fontsize=6.0)

# ================================================================ 2 harmonisation
panel(R0, T0, R1, T1, "2", "Harmonisation")
groups = [("Survey\npredictors", 14, WITHIN), ("pH,\nEC", 2, "#E69F00"), ("City\ncovariates", 7, "#CC79A7"),
          ("Targets", 4, SSA)]
cw, ch, nrow = 0.58, 1.05, 14
gx, gy = R0 + 2.4, T0 + 2.4
rng = np.random.default_rng(7)
cx = gx
for gi, (gname, ncol, col) in enumerate(groups):
    miss = rng.random((nrow, ncol)) < (0.18 if gi < 2 else 0.0)
    for j in range(ncol):
        for r in range(nrow):
            fc = "white" if miss[r, j] else col
            a = 1.0 if miss[r, j] else (0.30 if r % 2 else 0.42)
            ax.add_patch(Rectangle((cx + j * cw, gy + r * ch), cw * 0.92, ch * 0.88, fc=fc, alpha=a,
                                   ec="#BBBBBB" if miss[r, j] else "none", lw=0.3))
    ax.add_patch(Rectangle((cx, gy + nrow * ch + 0.25), ncol * cw - 0.05, 0.6, fc=col, ec="none"))
    txt(cx + ncol * cw / 2, gy + nrow * ch + 2.4, gname, ha="center", fontsize=6.1, linespacing=1.0)
    cx += ncol * cw + 1.8
txt(gx - 0.7, gy + nrow * ch / 2, "1,390 samples", rotation=90, ha="center", fontsize=6.1)
sx = cx + 1.6
txt(sx, 58.2, "Steps", weight="bold")
steps = ["Units harmonised to g L$^{-1}$\n(NH$_4^+$-N in mg L$^{-1}$)", "Author-flagged outliers removed",
         "Overlapping studies de-duplicated", "Containment, toilet and emptying\ncoded consistently",
         "Targets log$_{10}$-transformed", "White cells: missing values"]
yy = 55.6
for i, st in enumerate(steps):
    if i < 5:
        ax.add_patch(Circle((sx + 0.55, yy), 0.75, fc=NAVY, ec="none"))
        txt(sx + 0.55, yy, str(i + 1), ha="center", fontsize=5.6, color="white", weight="bold")
    else:
        ax.add_patch(Rectangle((sx + 0.1, yy - 0.5), 0.9, 1.0, fc="white", ec="#888888", lw=0.4))
    txt(sx + 1.9, yy, st, fontsize=6.2, linespacing=1.3)
    yy -= 3.4 if "\n" in st else 2.4
txt(sx, 38.4, "Targets: TS, VS, COD, NH$_4^+$-N", fontsize=6.4, weight="bold")

# ================================================================ 3 validation
panel(L0, B0, L1, B1, "3", "Models and validation")
designs = [("Within-city CV", "random 5-fold × 5", WITHIN, "within"),
           ("Leave-one-city-out", "main test, one city at a time", LOCO, "loco"),
           ("External test", "train SSA, test outside", EXT, "ext")]
bx0, bw, bh, gap = L0 + 18.0, 2.15, 4.2, 0.45
cities = ["Lus", "Kam", "Dar", "Oua", "Nai", "Kum", "Fre", "Pie"]
for k, c in enumerate(cities):
    txt(bx0 + k * (bw + gap) + bw / 2, B1 - 5.6, c, ha="center", fontsize=5.6)
xe = bx0 + 8 * (bw + gap) + 1.0
txt(xe + bw / 2, B1 - 5.6, "Ext.", ha="center", fontsize=5.6)
for r, (nm, sub, col, kind) in enumerate(designs):
    y = B1 - 10.6 - r * 5.6
    txt(L0 + 1.6, y + bh / 2 + 0.9, nm, fontsize=6.6, weight="bold")
    txt(L0 + 1.6, y + bh / 2 - 1.1, sub, fontsize=6.1)
    for k in range(9):
        x0 = bx0 + k * (bw + gap) if k < 8 else xe
        for u in range(8):                       # 2 x 4 sub-squares per city block
            ux, uy = x0 + (u % 2) * bw / 2, y + (u // 2) * bh / 4
            if kind == "within":
                st = ("test" if ((k == 1 and u in (1, 6)) or (k == 0 and u in (2, 7))) else "train") if k in (0, 1) else "idle"
            elif kind == "loco":
                st = "idle" if k == 8 else ("test" if k == 1 else "train")
            else:
                st = "test" if k == 8 else "train"
            fc = {"train": TRAIN, "test": col, "idle": IDLE}[st]
            ax.add_patch(Rectangle((ux + 0.06, uy + 0.06), bw / 2 - 0.12, bh / 4 - 0.12, fc=fc, ec="none"))
    if kind == "loco":
        txt(bx0 + 8 * (bw + gap) - 0.2, y - 0.75, "each city held out in turn", ha="right", fontsize=5.8,
            color=LOCO, weight="bold")
ax.plot([xe - 0.75, xe - 0.75], [B1 - 22.2, B1 - 6.6], lw=0.6, color=NAVY, ls=(0, (2, 2)))
yl = B0 + 5.2
for i, (lab, fc) in enumerate([("training", TRAIN), ("test", "#555555"), ("not used", IDLE)]):
    xx = L0 + 1.6 + i * 7.6
    if lab == "test":
        for j, c in enumerate([WITHIN, LOCO, EXT]):
            ax.add_patch(Rectangle((xx + j * 0.9, yl - 0.5), 0.8, 1.0, fc=c, ec="none"))
        txt(xx + 3.0, yl, lab, fontsize=6.1)
    else:
        ax.add_patch(Rectangle((xx, yl - 0.5), 1.0, 1.0, fc=fc, ec="#999999" if lab == "not used" else "none", lw=0.3))
        txt(xx + 1.5, yl, lab, fontsize=6.1)
txt(L0 + 1.6, B0 + 2.2, "Models: B0 pooled median · B1 containment rule · ridge · random forest · XGBoost ·\ncity-weighted XGBoost · Gaussian process · linear mixed model (city intercept)",
    fontsize=5.9)

# ================================================================ 4 outputs
panel(R0, B0, R1, B1, "4", "Outputs evaluated")
cells = [("City offset", "bias in a\nnew city"), ("Prediction\nintervals", "split-conformal,\n90% coverage"),
         ("Drivers", "SHAP\nattribution"), ("Local\nsampling rule", "samples\nneeded, $k$")]
cwid, chgt = (R1 - R0 - 3.0) / 2, (B1 - B0 - 6.4) / 2
for i, (h, sub) in enumerate(cells):
    c0 = R0 + 1.0 + (i % 2) * (cwid + 1.0)
    r0 = B1 - 5.4 - (i // 2 + 1) * chgt
    ax.add_patch(FancyBboxPatch((c0, r0 + 0.4), cwid, chgt - 0.8, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=TINT, ec="none"))
    txt(c0 + 1.0, r0 + chgt - 1.4, h, fontsize=6.8, weight="bold", va="top", linespacing=1.05)
    txt(c0 + 1.0, r0 + chgt - 1.4 - (4.6 if "\n" in h else 2.6), sub, fontsize=6.1, va="top", linespacing=1.1)
    g = inset(c0 + cwid * 0.47, r0 + 1.3, c0 + cwid - 0.9, r0 + chgt - 1.3)
    g.set_facecolor("none")
    for sp in g.spines.values():
        sp.set_visible(False)
    g.set_xticks([]); g.set_yticks([])
    xx = np.linspace(-3, 6, 200)
    if i == 0:
        g.fill_between(xx, np.exp(-0.5 * (xx - 0) ** 2), color=GREEN, alpha=0.35, lw=0)
        g.plot(xx, np.exp(-0.5 * (xx - 0) ** 2), color=GREEN, lw=0.9)
        g.plot(xx, np.exp(-0.5 * (xx - 2.6) ** 2), color=LOCO, lw=0.9, ls=(0, (3, 1.5)))
        g.annotate("", xy=(2.6, 1.08), xytext=(0, 1.08), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK))
        g.set_ylim(0, 1.3); g.set_xlim(-2.6, 5.2)
        g.axhline(0, color=INK, lw=0.6)
    elif i == 1:
        p = np.array([0.8, 1.6, 2.3, 3.2]); g.plot([0.3, 3.7], [0.3, 3.7], color=INK, lw=0.5, ls=(0, (2, 2)))
        g.errorbar(p, p + np.array([0.15, -0.2, 0.25, -0.1]), yerr=0.75, fmt="o", ms=2.6, color=GREEN,
                   elinewidth=0.9, capsize=1.3)
        g.set_xlim(0.2, 3.8); g.set_ylim(-0.4, 4.4)
        g.spines["left"].set_visible(True); g.spines["bottom"].set_visible(True)
    elif i == 2:
        g.barh(range(5), [0.45, 0.6, 0.72, 0.85, 1.0], color=GREEN, height=0.62)
        g.axvline(0, color=INK, lw=0.6); g.set_xlim(0, 1.1)
    else:
        k = np.linspace(0, 15, 60); g.plot(k, 1 - 0.55 * np.exp(-k / 3.2), color=GREEN, lw=1.1)
        g.plot([0, 3, 5, 10, 15], 1 - 0.55 * np.exp(-np.array([0, 3, 5, 10, 15]) / 3.2), "o", ms=2.6, color=GREEN)
        g.set_ylim(0.35, 1.08); g.set_xlim(-1, 16)
        g.spines["left"].set_visible(True); g.spines["bottom"].set_visible(True)
        g.set_xlabel("$k$", fontsize=6.2, labelpad=0)
    for sp in ("left", "bottom"):
        g.spines[sp].set_color(INK); g.spines[sp].set_linewidth(0.6)

# ================================================================ flow arrows
arrow((L1 + 0.15, 50.0), (R0 - 0.15, 50.0))
ym = (T0 + B1) / 2
ax.plot([(R0 + R1) / 2, (R0 + R1) / 2, (L0 + L1) / 2], [T0 - 0.1, ym, ym], color=NAVY, lw=1.4, solid_capstyle="butt", zorder=5)
arrow(((L0 + L1) / 2, ym + 0.05), ((L0 + L1) / 2, B1 + 0.1))
arrow((L1 + 0.15, 16.0), (R0 - 0.15, 16.0))

for ext in ("pdf", "png"):
    fig.savefig(F / f"fig2_study_design.{ext}", dpi=300 if ext == "png" else None)
print("ok")
