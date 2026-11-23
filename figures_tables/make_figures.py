"""Publication figures for the manuscript. Reads results/ and writes figures/ (PDF + PNG)."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
from paths import MODEL as R, REV, FIG as F, ROOT
F.mkdir(exist_ok=True)
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]  # Okabe-Ito
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes"], "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 400, "pdf.fonttype": 42})
TLAB = {"ts_gL": "TS", "vs_gL": "VS", "cod_gL": "COD", "nh4_mgL": "NH$_4^+$-N"}
SSA_CITIES = ["Lusaka", "Kumasi", "Naivasha", "Kampala", "Freetown", "Ouagadougou", "Dar es Salaam"]


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(F / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1 study design
def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 104); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, title, lines, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.2, weight="bold")
        for i, l in enumerate(lines):
            ax.text(x + w / 2, y + h - 5.4 - i * 3.0, l, ha="center", va="top", fontsize=5.9)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))

    box(0.5, 2, 24, 46, "Open data (7 sources)", ["Ward et al. 2021", "(Lusaka)", "Andriessen et al. 2023",
        "(7 countries)", "Englund et al. 2020", "(Kampala, Hanoi)", "Ward et al. 2023;", "Shaw et al. 2024",
        "Prasad et al. 2021 (Sircilla)", "Global Faecal Sludge Database", "(Dar es Salaam, Ouagadougou,",
        "other sites)", "", "1,390 samples, 33 cities"], "#EAF2FA")
    box(28.5, 2, 22, 46, "Harmonisation", ["Common units (g/L, mg/L)", "QC rules", "Duplicate removal", "",
        "Predictors:", "14 containment and", "household variables", "Field pH and EC", "City climate covariates", "",
        "Targets (log$_{10}$):", "TS, VS, COD, NH$_4^+$-N"], "#F4F4F4")
    box(54.5, 26.5, 23, 21.5, "Validation designs", ["Within-city random 5-fold", "Leave-one-city-out",
        "(7 SSA cities)", "External tests (India,", "Vietnam, Lebanon, others)"], "#FFF4E0")
    box(54.5, 2, 23, 21.5, "Models", ["Pooled median", "Containment-type rule", "Ridge, random forest",
        "XGBoost (\u00b1 city-balanced)", "Gaussian process"], "#FFF4E0")
    box(81.5, 26.5, 22, 21.5, "Diagnostics", ["City offsets", "Conformal 90% intervals", "SHAP drivers"], "#E8F6EF")
    box(81.5, 2, 22, 21.5, "Planning rule", ["Local calibration with", "k = 0\u201340 samples", "in a new city"], "#E8F6EF")
    arrow(25.2, 25, 28.2, 25); arrow(50.8, 37, 54.2, 37); arrow(50.8, 13, 54.2, 13)
    arrow(77.8, 37, 81.2, 37); arrow(77.8, 13, 81.2, 13); arrow(92.5, 26.2, 92.5, 23.8)
    save(fig, "fig1_study_design")


# ---------------------------------------------------------------- Fig 2 city distributions
def fig2(d):
    s = d[(d.region == "SSA") & d.city.isin(SSA_CITIES) & (d.outlier_flag == 0)].copy()
    order = s.groupby("city").ts_gL.median().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    cols = {"pit latrine": OI[3], "septic tank": OI[0], "other": "#888888"}
    for ax, t in zip(axes, ["ts_gL", "cod_gL"]):
        for i, c in enumerate(order):
            g = s[(s.city == c) & s[t].notna() & (s[t] > 0)]
            if not len(g):
                continue
            ax.boxplot(g[t], positions=[i], widths=0.55, showfliers=False, whis=(5, 95),
                       medianprops=dict(color="k", lw=1), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            rng = np.random.default_rng(i)
            for ct, col in cols.items():
                gg = g[g.containment == ct]
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(gg)), gg[t], s=3.5, alpha=0.45, color=col, lw=0,
                           label=ct if i == 0 else None, zorder=3)
            ax.text(i, 0.06 if t == "ts_gL" else 0.06, f"n={len(g)}", ha="center", fontsize=5.8, color="#444444")
        ax.set_yscale("log"); ax.set_ylim(0.04, 2000)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylabel(f"{TLAB[t]} (g L$^{{-1}}$)")
        ax.set_title(f"({'a' if t == 'ts_gL' else 'b'}) {TLAB[t]} by city")
    h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=3.5, label=k) for k, c in cols.items()]
    axes[0].legend(handles=h, loc="upper right", frameon=False, title="Containment", title_fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig2_city_distributions")


# ---------------------------------------------------------------- Fig 3 transfer gap + offsets
def fig3():
    p = pd.read_csv(R / "loco_metrics_pooled.csv")
    w = pd.read_csv(R / "within_city_cv_metrics.csv")
    a = pd.read_csv(R / "loco_predictions.csv")
    targets = ["ts_gL", "vs_gL", "cod_gL", "nh4_mgL"]
    model = "random_forest"
    within = w[(w.features == "survey") & (w.model == model)].groupby("target").R2_log.mean()
    loco = p[(p.features == "survey") & (p.model == model)].set_index("target").R2_log
    x = a[(a.model == model) & (a.features == "survey")].copy()
    x["y"], x["p"] = np.log10(x.observed), np.log10(x.predicted)
    x["off"] = (x.p - x.y).groupby([x.target, x.test_city]).transform("median")

    def r2(g, col):
        return 1 - ((g.y - g[col]) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()
    x["p_adj"] = x.p - x.off
    adj = x.groupby("target").apply(lambda g: r2(g, "p_adj"))
    sc = pd.read_csv(REV / "R11_transfer_scenarios.csv")
    op = sc[sc.scenario.str.startswith("operational")].set_index("target")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    wd = 0.2
    series = [("Within-city random CV", within, OI[0], None),
              ("Leave-one-city-out, uncalibrated", loco, OI[3], None),
              ("Leave-one-city-out + 5 local samples", op.R2_log, OI[2], None),
              ("Leave-one-city-out + oracle offset (diagnostic)", adj, "white", "////")]
    for j, (lab, ser, col, hatch) in enumerate(series):
        vals = [ser.get(t, np.nan) for t in targets]
        xs = np.arange(4) + (j - 1.5) * wd
        ax.bar(xs, vals, wd, color=col, label=lab, hatch=hatch, edgecolor="#555555" if hatch else col, lw=0.6)
        if j == 2:
            lo_ = [op.R2_p10.get(t) for t in targets]; hi_ = [op.R2_p90.get(t) for t in targets]
            ax.errorbar(xs, vals, yerr=[np.array(vals) - lo_, np.array(hi_) - vals], fmt="none", ecolor="#222222",
                        elinewidth=0.7, capsize=1.6)
        for i, v in enumerate(vals):
            if j != 2:
                ax.text(xs[i], v + (0.02 if v >= 0 else -0.07), f"{v:.2f}", ha="center", fontsize=5.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels([TLAB[t] for t in targets])
    ax.set_ylabel("R$^2$ (log$_{10}$ scale)"); ax.set_ylim(-0.42, 1.0)
    ax.set_title("(a) Random forest, survey predictors"); ax.legend(frameon=False, loc="upper left", fontsize=6.0)
    # offsets heatmap
    ax = axes[1]
    off = (10 ** x.groupby(["test_city", "target"]).off.median()).unstack()[targets]
    cities = off.index.tolist()
    lo = np.log10(off.values)
    im = ax.imshow(lo, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(lo.shape[0]):
        for j in range(lo.shape[1]):
            v = off.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}×", ha="center", va="center", fontsize=6.2,
                        color="white" if abs(lo[i, j]) > 0.6 else "black")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=6.2, color="#888888")
    ax.set_xticks(range(4)); ax.set_xticklabels([TLAB[t] for t in targets])
    ax.set_yticks(range(len(cities))); ax.set_yticklabels(cities)
    ax.set_title("(b) City offset: median predicted / observed")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(["0.1×", "0.3×", "1×", "3×", "10×"], fontsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig3_transfer_gap_offsets")


# ---------------------------------------------------------------- Fig 4 SHAP importance
NICE = {"flush_toilet": "Flush toilet (vs dry)", "log_sludge_age": "Sludge age", "log_ec": "Electrical conductivity",
        "ph": "pH", "is_pit": "Pit latrine", "is_septic": "Septic tank", "is_other": "Other containment",
        "log_volume": "Containment volume", "log_users": "Number of users", "household": "Household (vs other)",
        "water_connection": "Piped water on premises", "watertight": "Watertight lining", "never_emptied": "Never emptied",
        "greywater": "Greywater input", "solid_waste": "Solid waste present", "additives": "Additives used"}


def fig4():
    f = pd.read_csv(R / "feature_importance.csv")
    targets = ["ts_gL", "vs_gL", "cod_gL", "nh4_mgL"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), sharex=True)
    for k, (ax, t) in enumerate(zip(axes, targets)):
        g = f[f.target == t].sort_values("importance", ascending=True).tail(8)
        col = [OI[5] if n in ("ph", "log_ec") else OI[0] for n in g.feature]
        ax.barh([NICE.get(n, n) for n in g.feature], g.importance, color=col)
        ax.set_title(f"({'abcd'[k]}) {TLAB[t]}")
        ax.tick_params(axis="y", labelsize=6.2)
        ax.set_xlabel("mean |SHAP| (log$_{10}$ units)")
    h = [plt.Rectangle((0, 0), 1, 1, color=OI[0], label="Survey predictor"),
         plt.Rectangle((0, 0), 1, 1, color=OI[5], label="Field probe (pH, EC)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig4_shap_importance")


# ---------------------------------------------------------------- Fig 5 local calibration
def fig5():
    spec = importlib.util.spec_from_file_location("m", ROOT / "04_model.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    agg = pd.read_csv(R / "local_calibration.csv")
    a = m._consistent(agg, 15)
    g = a.groupby(["target", "strategy", "k_local"])[["within_factor2_pct", "RMSE_log10"]].mean().reset_index()
    styles = [("pooled_random_forest+shrunk_offset", "Random forest (other cities) + shrunk local offset", OI[0], "-", "o"),
              ("pooled_B1_containment_rule+offset", "Containment rule (other cities) + local offset", OI[2], "-", "^"),
              ("local_containment_rule_only", "Local samples only: median per containment type", OI[3], "--", "s"),
              ("local_median_only", "Local samples only: overall median", "#777777", ":", "x")]
    targets = ["ts_gL", "vs_gL", "cod_gL", "nh4_mgL"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7), sharey=True)
    for k, (ax, t) in enumerate(zip(axes, targets)):
        n_c = a[a.target == t].test_city.nunique()
        for strat, lab, col, ls, mk in styles:
            s = g[(g.target == t) & (g.strategy == strat)].sort_values("k_local")
            ax.plot(s.k_local, s.within_factor2_pct, ls=ls, marker=mk, ms=3, lw=1, color=col, label=lab)
        ax.set_title(f"({'abcd'[k]}) {TLAB[t]} ({n_c} cities)")
        ax.set_xticks([0, 3, 5, 10, 15]); ax.set_xlabel("Local samples in new city, $k$")
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    axes[0].set_ylabel("Predictions within a factor of 2 (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig5_local_calibration")


# ---------------------------------------------------------------- Fig S1 observed vs predicted
def figS1():
    a = pd.read_csv(R / "loco_predictions.csv")
    b = a[(a.model == "random_forest") & (a.features == "survey")]
    cities = sorted(b.test_city.unique()); colors = {c: OI[i % 8] for i, c in enumerate(cities)}
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
    for k, (ax, t) in enumerate(zip(axes, ["ts_gL", "vs_gL", "cod_gL", "nh4_mgL"])):
        g = b[b.target == t]
        for c, gc in g.groupby("test_city"):
            ax.scatter(gc.observed, gc.predicted, s=3, alpha=0.5, color=colors[c], lw=0, label=c)
        lo, hi = g[["observed", "predicted"]].min().min(), g[["observed", "predicted"]].max().max()
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.6); ax.plot([lo, hi], [2 * lo, 2 * hi], "k:", lw=0.5); ax.plot([lo, hi], [lo / 2, hi / 2], "k:", lw=0.5)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"({'abcd'[k]}) {TLAB[t]}")
        ax.set_xlabel("Observed"); ax.set_ylabel("Predicted (held-out city)" if k == 0 else "")
    h = [plt.Line2D([], [], marker="o", ls="", color=colors[c], ms=3.5, label=c) for c in cities]
    fig.legend(handles=h, loc="lower center", ncol=len(cities), frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "figS1_observed_vs_predicted")


if __name__ == "__main__":
    d = pd.read_csv(ROOT / "data/processed/fs_model_table.csv")
    fig2(d); fig3(); fig4(); fig5(); figS1()
    print("figures written:", sorted(p.name for p in F.glob("*.pdf")))
