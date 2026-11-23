"""Figures added in revision: campaign heterogeneity (main text) and sample flow (supplement)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
from paths import REV as RV, FIG as F
SERIF = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix", "font.size": 8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
INK, NAVY = "#111111", "#1F3A5F"
PT = {"in containment": "#009E73", "truck at discharge": "#E69F00", "not recorded": "#8C8C8C"}


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(F / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def point(s):
    return "in containment" if s.startswith("in containment") else ("truck at discharge" if s.startswith("truck") else "not recorded")


def fig_campaigns():
    s = pd.read_csv(RV / "R2_source_by_city.csv")
    s["pt"] = s.sampling_point.map(point)
    v = pd.read_csv(RV / "R3_variance_decomposition.csv")
    order = s.groupby("city").TS_median.median().sort_values().index.tolist()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.42})
    for i, c in enumerate(order):
        g = s[s.city == c]
        if len(g) > 1:
            a.plot([g.TS_median.min(), g.TS_median.max()], [i, i], color="#B5B5B5", lw=1.4, zorder=1)
        for _, r in g.iterrows():
            a.scatter(r.TS_median, i, s=14 + 1.1 * r.n, color=PT[r.pt], edgecolor="white", lw=0.6, zorder=3)
        if len(g) > 1:
            ratio = g.TS_median.max() / g.TS_median.min()
            a.text(330, i, f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×", va="center", ha="left", fontsize=7.4, color=INK)
    a.set_xscale("log"); a.set_xlim(1.2, 300)
    a.set_yticks(range(len(order))); a.set_yticklabels(order, color=INK)
    a.set_xlabel("Median TS of each study, g L$^{-1}$ (log scale)", color=INK)
    a.text(330, len(order) - 0.35, "max/min", fontsize=7, ha="left", va="bottom", color=INK)
    a.grid(axis="x", which="major", lw=0.3, color="#E3E3E3"); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    h = [Line2D([], [], marker="o", ls="", color=c, ms=5.5, label=k) for k, c in PT.items()]
    h += [Line2D([], [], marker="o", ls="", color="#BBBBBB", ms=ms, label=f"n = {n}") for n, ms in ((20, 4.2), (150, 8.5), (450, 13))]
    a.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=6.8,
             handletextpad=0.2, columnspacing=1.0, title="Sampling point and study size", title_fontsize=7.2)
    a.set_title("(a)", loc="left", fontsize=9, weight="bold", pad=34, x=-0.32)

    vv = v[v.fixed_effects == "containment"].set_index("target").loc[["ts_gL", "vs_gL", "cod_gL"]]
    labs = ["TS", "VS", "COD"]
    parts = [("pct_city", "Between cities", NAVY), ("pct_source_within_city", "Between studies\nwithin a city", "#D55E00"),
             ("pct_residual", "Between samples\nwithin a study", "#C9D3DE")]
    left = np.zeros(3)
    for col, lab, c in parts:
        b.barh(range(3)[::-1], vv[col].values, left=left, color=c, height=0.6, label=lab)
        for j, val in enumerate(vv[col].values):
            b.text(left[j] + val / 2, 2 - j, f"{val:.0f}%", ha="center", va="center", fontsize=7.4,
                   color="white" if c != "#C9D3DE" else INK, weight="bold")
        left += vv[col].values
    b.set_yticks(range(3)[::-1]); b.set_yticklabels(labs, color=INK); b.set_xlim(0, 100)
    b.set_xlabel("Share of variance in log$_{10}$ concentration (%)", color=INK)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=3, frameon=False, fontsize=6.6, handlelength=1.0,
             columnspacing=0.8, handletextpad=0.3)
    b.set_title("(b)", loc="left", fontsize=9, weight="bold", pad=34, x=-0.16)
    save(fig, "fig_campaign_heterogeneity")


def fig_flow():
    fl = pd.read_csv(RV / "R1_sample_flow.csv").set_index("stage").n
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, title, body, fc="white", ec=NAVY):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.3, body, ha="center", va="top", fontsize=7.2, color=INK, linespacing=1.3)

    def arr(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=NAVY))

    box(30, 51, 40, 10, f"Harmonised samples: n = {fl.iloc[0]:,}", "7 open sources, 33 cities, 15 countries\n(duplicates and treatment-plant samples removed)", fc="#F3F6FA")
    box(72, 38, 26, 10, f"External test set: n = {fl.iloc[1]}", "25 sites outside SSA\nnot used for training")
    box(30, 38, 40, 10, f"Sub-Saharan Africa: n = {fl.iloc[2]:,}", "8 cities")
    box(2, 38, 26, 10, f"Excluded: n = {fl.iloc[3]}", "outliers flagged by the\noriginal authors (Kampala)", ec="#B2182B")
    box(30, 25, 40, 10, f"SSA analytical samples: n = {fl.iloc[4]:,}", "Fig. 1 and Table 1 (samples)")
    arr((50, 51), (50, 48.2)); arr((70, 56), (85, 48.2)); arr((50, 38), (50, 35.2)); arr((30, 43), (28.2, 43))
    tg = [("TS", "ts_gL"), ("VS", "vs_gL"), ("COD", "cod_gL"), ("NH$_4^+$-N", "nh4_mgL")]
    cities = {"ts_gL": 6, "vs_gL": 5, "cod_gL": 6, "nh4_mgL": 3}
    for i, (lab, t) in enumerate(tg):
        x = 2 + i * 24.5
        meas = fl[f"  with measured {t}"]; loco = fl[f"  with {t} and >=50% survey predictors (LOCO set)"]
        box(x, 2, 22.5, 17, lab, f"measured: n = {meas}\nwith ≥50% survey\npredictors: n = {loco}\nheld-out cities: {cities[t]}")
        arr((50, 25), (x + 11.25, 19.2))
    ax.text(50, 22.6, "target-specific sets (Table 1 counts)", ha="center", fontsize=7, style="italic", color=INK,
            bbox=dict(fc="white", ec="none", pad=0.5))
    save(fig, "figS1_sample_flow")


if __name__ == "__main__":
    fig_campaigns(); fig_flow(); print("ok")


def fig_calibration():
    g = pd.read_csv(RV / "R8_local_calibration_gain.csv")
    g13 = pd.read_csv(RV / "R13_local_calibration_gain.csv")          # matched random totals 6 and 20
    g = pd.concat([g, g13[g13.k.isin([6, 20])]]).sort_values(["target", "design", "k"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    styles = {"random (total k)": ("#0072B2", "Random draws", 1),
              "stratified (k per containment type)": ("#009E73", "k per containment type (pit, non-pit)", 2)}
    for ax, (t, lab) in zip(axes, [("ts_gL", "(a) TS"), ("vs_gL", "(b) VS"), ("cod_gL", "(c) COD")]):
        ax.axhline(0, color=INK, lw=0.6)
        for des, (col, name, mult) in styles.items():
            s = g[(g.target == t) & (g.design == des)].sort_values("k")
            x = s.k * mult
            ax.fill_between(x, s.gain_p10, s.gain_p90, color=col, alpha=0.16, lw=0)
            ax.plot(x, s.gain_mean, "-o", color=col, ms=3.2, lw=1.2,
                    label=f"{name} ({int(s.cities.min())}–{int(s.cities.max())} cities)" if s.cities.min() != s.cities.max()
                    else f"{name} ({int(s.cities.min())} cities)")
            for _, r in s[(s.k * mult).isin([10, 20])].iterrows():
                yv = r.gain_p90 + 1.5 if mult == 2 else r.gain_p10 - 4.5
                ax.text(r.k * mult, yv, f"{r.share_draws_improved:.0f}%", ha="center", fontsize=6.0, color=col)
        ax.set_title(lab, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("Local samples analysed (total)", color=INK)
        ax.set_xticks([0, 5, 10, 15, 20, 30]); ax.set_xlim(1, 31)
        ax.grid(axis="y", lw=0.3, color="#E3E3E3")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Gain in % within a factor of 2\n(percentage points vs no local data)", color=INK)
    axes[0].set_ylim(-22, 48)
    h = [Line2D([], [], color="#0072B2", marker="o", ms=3.2, lw=1.2, label="Random draws regardless of containment type"),
         Line2D([], [], color="#009E73", marker="o", ms=3.2, lw=1.2, label="Equal numbers from pit latrines and other containments, each with its own shift")]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7)
    save(fig, "fig5_local_calibration")


if __name__ == "__main__":
    fig_calibration()
