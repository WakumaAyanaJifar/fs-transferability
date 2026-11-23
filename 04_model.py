"""
04_model.py
Cross-city prediction of fecal sludge TS, VS, COD (and NH4-N) from low-cost containment/household data.

Validation designs
  1. Leave-one-city-out (LOCO) over the Sub-Saharan African cities  -> main result (transfer to a new city)
  2. Random 5-fold CV within Lusaka and Kampala                      -> what earlier single-city studies report
  3. External test: train on all SSA cities, predict India, Lebanon, Vietnam

Feature sets
  survey        : containment and household questionnaire data only (available in every city)
  survey_field  : survey + field pH and EC (cheap probes; missing for most Kampala samples and Sircilla)

Models
  B0_pooled_median    : median of training data (no inputs)
  B1_containment_rule : median per containment type (what practitioners do now)
  ridge, random_forest, xgboost, xgboost_city_balanced, gaussian_process (as in Englund et al. 2020)

Uncertainty: 90% split-conformal intervals for xgboost, calibrated on inner leave-one-city-out residuals.
Interpretation: SHAP values for xgboost trained on all SSA cities.
Local calibration: accuracy in a new city after measuring k = 0-40 local samples (offset correction).

Run:  python 04_model.py            (about 5-15 minutes)
      python 04_model.py --quick    (fewer models/repeats, for a fast check)
      python 04_model.py --calibration-only   (only the local-calibration analysis)
Outputs in outputs/model/
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
TABLE = ROOT / "data" / "processed" / "fs_model_table.csv"
OUT = ROOT / "outputs" / "model"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_TEST = 10          # a city is held out only if it has at least this many samples for the target
ALPHA = 0.10           # 90% prediction intervals

TARGETS = {"ts_gL": "TS (g/L)", "vs_gL": "VS (g/L)", "cod_gL": "COD (g/L)", "nh4_mgL": "NH4-N (mg/L)"}
SURVEY = ["is_pit", "is_septic", "is_other", "watertight", "log_volume", "log_users", "household",
          "flush_toilet", "greywater", "water_connection", "solid_waste", "additives",
          "never_emptied", "log_sludge_age"]
FIELD = ["ph", "log_ec"]
CITY = ["city_tavg_c", "city_precip_mm", "city_rel_hum_pct", "city_elevation_m", "log_city_pop_density", "city_groundwater"]
FEATURE_SETS = {"survey": SURVEY, "survey_field": SURVEY + FIELD, "survey_city": SURVEY + CITY}


# ------------------------------------------------------------------ data
def load():
    d = pd.read_csv(TABLE)
    d = d[d["outlier_flag"].fillna(0) == 0].copy()
    c = d["containment"]
    for k, v in {"is_pit": "pit latrine", "is_septic": "septic tank", "is_other": "other"}.items():
        d[k] = np.where(c.isna(), np.nan, (c == v).astype(float))
    d["log_volume"] = np.log1p(d["volume_m3"].clip(lower=0))
    d["log_users"] = np.log1p(d["users"].clip(lower=0))
    d["log_sludge_age"] = np.log1p(d["sludge_age_yr"].clip(lower=0))
    d["log_ec"] = np.log1p(d["ec_mScm"].clip(lower=0))
    if "city_pop_density_km2" in d:
        d["log_city_pop_density"] = np.log10(d["city_pop_density_km2"].clip(lower=1))
    for c in CITY:
        if c not in d:
            d[c] = np.nan
    return d


# ------------------------------------------------------------------ models
class MedianBaseline:
    def fit(self, X, y, **kw):
        self.m = np.median(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class ContainmentRule:
    """Median of log-target per containment type; pooled median when type unknown."""
    def fit(self, X, y, **kw):
        self.all = np.median(y)
        self.m = {}
        for col in ("is_pit", "is_septic", "is_other"):
            mask = X[col].values == 1
            self.m[col] = np.median(y[mask]) if mask.sum() >= 3 else self.all
        return self

    def predict(self, X):
        p = np.full(len(X), self.all)
        for col, v in self.m.items():
            p[X[col].values == 1] = v
        return p


def imputed(model, scale=False):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def make_models(quick):
    m = {
        "B0_pooled_median": lambda: MedianBaseline(),
        "B1_containment_rule": lambda: ContainmentRule(),
        "ridge": lambda: imputed(RidgeCV(alphas=np.logspace(-3, 3, 13)), scale=True),
        "random_forest": lambda: imputed(RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                                                max_features=0.5, random_state=SEED, n_jobs=-1)),
        "xgboost": lambda: XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=5,
                                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                        random_state=SEED, n_jobs=-1),
    }
    if not quick:
        m["xgboost_city_balanced"] = m["xgboost"]  # same model, fitted with weights 1/n_city
        m["gaussian_process"] = lambda: imputed(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(1)) + WhiteKernel(0.1),
            normalize_y=True, random_state=SEED), scale=True)
    return m


def fit_predict(name, factory, Xtr, ytr, Xte, city_tr):
    model = factory()
    kw = {}
    if name == "xgboost_city_balanced":
        w = 1.0 / city_tr.map(city_tr.value_counts())
        kw["sample_weight"] = (w / w.mean()).values
    model.fit(Xtr, ytr, **kw)
    return model.predict(Xte), model


def metrics(y_log, p_log):
    """Metrics on log10 scale plus interpretable original-scale ones."""
    y, p = 10 ** y_log, 10 ** p_log
    ratio = p / y
    return {
        "n": len(y_log),
        "R2_log": r2_score(y_log, p_log) if len(y_log) > 2 else np.nan,
        "RMSE_log10": float(np.sqrt(mean_squared_error(y_log, p_log))),
        "bias_log10": float(np.mean(p_log - y_log)),
        "MdAPE_pct": float(np.median(np.abs(p - y) / y) * 100),
        "within_factor2_pct": float(np.mean((ratio >= 0.5) & (ratio <= 2)) * 100),
    }


# ------------------------------------------------------------------ validation designs
def target_frame(d, target, feats, region="SSA"):
    sub = d[(d["region"] == region) & d[target].notna() & (d[target] > 0)].copy()
    sub = sub[sub[feats].notna().mean(axis=1) >= 0.5]  # need at least half the inputs
    return sub


def loco(d, models, quick):
    rows, preds = [], []
    for target in TARGETS:
        for fs_name, feats in FEATURE_SETS.items():
            sub = target_frame(d, target, feats)
            cities = [c for c, n in sub["city"].value_counts().items() if n >= MIN_TEST]
            if len(cities) < 2:
                continue
            y = np.log10(sub[target].values)
            for city in cities:
                te = (sub["city"] == city).values
                if fs_name == "survey_field" and sub.loc[te, FIELD].notna().mean().min() < 0.5:
                    continue  # field data mostly missing in this city
                if fs_name == "survey_city" and sub.loc[te, CITY].notna().mean().min() < 0.5:
                    continue
                for name, factory in models.items():
                    p, _ = fit_predict(name, factory, sub.loc[~te, feats], y[~te], sub.loc[te, feats],
                                       sub.loc[~te, "city"])
                    rows.append({"target": target, "features": fs_name, "model": name, "test_city": city,
                                 "n_train": int((~te).sum()), **metrics(y[te], p)})
                    preds.append(pd.DataFrame({"target": target, "features": fs_name, "model": name,
                                               "test_city": city, "sample_id": sub.loc[te, "sample_id"].values,
                                               "dataset": sub.loc[te, "dataset"].values,
                                               "observed": 10 ** y[te], "predicted": 10 ** p}))
            print(f"  LOCO {target:8s} {fs_name:13s} cities={cities}")
    per_city = pd.DataFrame(rows)
    allp = pd.concat(preds, ignore_index=True)
    pooled = (allp.groupby(["target", "features", "model"])
              .apply(lambda g: pd.Series(metrics(np.log10(g.observed.values), np.log10(g.predicted.values))))
              .reset_index())
    return per_city, pooled, allp


def within_city_cv(d, models, repeats):
    """Random K-fold inside one city: the optimistic design used by single-city studies."""
    rows = []
    for target in TARGETS:
        for fs_name, feats in FEATURE_SETS.items():
            sub = target_frame(d, target, feats)
            for city in ("Lusaka", "Kampala"):
                c = sub[sub["city"] == city]
                if len(c) < 50 or (fs_name == "survey_field" and c[FIELD].notna().mean().min() < 0.5):
                    continue
                y = np.log10(c[target].values)
                for name, factory in models.items():
                    if name in ("gaussian_process", "xgboost_city_balanced"):
                        continue
                    yy, pp = [], []
                    for r in range(repeats):
                        for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(c):
                            p, _ = fit_predict(name, factory, c.iloc[tr][feats], y[tr], c.iloc[te][feats],
                                               c.iloc[tr]["city"])
                            yy.append(y[te]); pp.append(p)
                    rows.append({"target": target, "features": fs_name, "model": name, "city": city,
                                 "design": f"random 5-fold x{repeats}",
                                 **metrics(np.concatenate(yy), np.concatenate(pp))})
    return pd.DataFrame(rows)


def conformal_loco(d):
    """90% intervals for xgboost: half-width = quantile of |residual| from inner LOCO on the training cities."""
    rows = []
    factory = make_models(True)["xgboost"]
    for target in TARGETS:
        for fs_name, feats in FEATURE_SETS.items():
            sub = target_frame(d, target, feats)
            cities = [c for c, n in sub["city"].value_counts().items() if n >= MIN_TEST]
            if len(cities) < 3:
                continue
            y = np.log10(sub[target].values)
            for city in cities:
                te = (sub["city"] == city).values
                if fs_name == "survey_field" and sub.loc[te, FIELD].notna().mean().min() < 0.5:
                    continue
                tr_cities = [c for c in cities if c != city]
                res = []
                for ic in tr_cities:  # inner leave-one-city-out on training cities
                    itr = (~te) & (sub["city"] != ic).values
                    ite = (sub["city"] == ic).values
                    p, _ = fit_predict("xgboost", factory, sub.loc[itr, feats], y[itr], sub.loc[ite, feats], None)
                    res.append(np.abs(y[ite] - p))
                res = np.concatenate(res)
                q = np.quantile(res, min(1.0, (1 - ALPHA) * (1 + 1 / len(res))))
                p, _ = fit_predict("xgboost", factory, sub.loc[~te, feats], y[~te], sub.loc[te, feats], None)
                cover = np.mean(np.abs(y[te] - p) <= q)
                rows.append({"target": target, "features": fs_name, "test_city": city, "n": int(te.sum()),
                             "halfwidth_log10": q, "interval_factor": 10 ** q,
                             "coverage_pct": cover * 100, "nominal_pct": (1 - ALPHA) * 100})
    return pd.DataFrame(rows)


def external(d, models):
    rows, preds = [], []
    sd = d["dataset"].astype(str).str.startswith("sludgedata")
    groups = {"Sircilla (India)": d["city"] == "Sircilla",
              "Hanoi (Vietnam)": d["city"] == "Hanoi",
              "Volaser India": (d["dataset"] == "volaser_7countries") & (d["country"] == "India"),
              "Volaser Lebanon": d["country"] == "Lebanon",
              "Global FS Database (India, Nepal, Indonesia, Thailand, Uruguay)": sd & (d["region"] == "external")}
    for target in TARGETS:
        for fs_name, feats in FEATURE_SETS.items():
            tr = target_frame(d, target, feats, "SSA")
            ytr = np.log10(tr[target].values)
            for gname, gmask in groups.items():
                te = d[gmask & d[target].notna() & (d[target] > 0)]
                te = te[te[feats].notna().mean(axis=1) >= 0.5]
                if len(te) < MIN_TEST or (fs_name == "survey_field" and te[FIELD].notna().mean().min() < 0.5) \
                        or (fs_name == "survey_city" and te[CITY].notna().mean().min() < 0.5):
                    continue
                yte = np.log10(te[target].values)
                for name, factory in models.items():
                    p, _ = fit_predict(name, factory, tr[feats], ytr, te[feats], tr["city"])
                    rows.append({"target": target, "features": fs_name, "model": name, "test_set": gname,
                                 **metrics(yte, p)})
    return pd.DataFrame(rows)


def importance(d):
    """SHAP (if installed) for xgboost trained on all SSA cities; permutation importance otherwise."""
    rows = []
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        shap = None
    from sklearn.inspection import permutation_importance
    factory = make_models(True)["xgboost"]
    for target in TARGETS:
        feats = FEATURE_SETS["survey_field"]
        sub = target_frame(d, target, feats)
        y = np.log10(sub[target].values)
        _, model = fit_predict("xgboost", factory, sub[feats], y, sub[feats].iloc[:1], None)
        if shap is not None:
            sv = shap.TreeExplainer(model).shap_values(sub[feats])
            imp = np.abs(sv).mean(axis=0)
            plt.figure()
            shap.summary_plot(sv, sub[feats], show=False, max_display=16)
            plt.title(f"SHAP: {TARGETS[target]} (log10), all SSA cities")
            plt.tight_layout(); plt.savefig(OUT / f"shap_{target}.png", dpi=200); plt.close()
            method = "mean |SHAP| (log10 units)"
        else:
            r = permutation_importance(model, sub[feats], y, n_repeats=20, random_state=SEED)
            imp, method = r.importances_mean, "permutation (R2 drop)"
        rows += [{"target": target, "feature": f, "importance": v, "method": method} for f, v in zip(feats, imp)]
    return pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False])


def figures(pooled, allp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    best = allp[(allp.model == "xgboost") & (allp.features == "survey")]
    palette = plt.get_cmap("tab10").colors
    colors = {c: palette[i] for i, c in enumerate(sorted(best["test_city"].unique()))}  # same colour per city in every panel
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.4 * len(TARGETS), 4.4))
    for ax, (t, lab) in zip(axes, TARGETS.items()):
        g = best[best.target == t]
        for city, gc in g.groupby("test_city"):
            ax.scatter(gc.observed, gc.predicted, s=12, alpha=0.6, color=colors[city], label=f"{city} (n={len(gc)})")
        if len(g):
            lo, hi = g[["observed", "predicted"]].min().min(), g[["observed", "predicted"]].max().max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.plot([lo, hi], [2 * lo, 2 * hi], "k:", lw=0.8); ax.plot([lo, hi], [lo / 2, hi / 2], "k:", lw=0.8)
            ax.legend(fontsize=7, loc="upper left")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{lab}: held-out city")
        ax.set_xlabel("observed"); ax.set_ylabel("predicted (XGBoost, survey inputs)")
    plt.tight_layout(); plt.savefig(OUT / "loco_observed_vs_predicted.png", dpi=200); plt.close()


KS = [0, 3, 5, 10, 15, 20, 40]
K_ALL = 15          # largest k available in every city (summary over all cities uses k <= 15)
CAL_REPEATS = 100
CAL_MODELS = ["B1_containment_rule", "random_forest", "xgboost"]


def local_calibration(d, quick):
    """How many local samples does a new city need?
    For each held-out city: fit on the other cities once; then draw k local samples (CAL_REPEATS times),
    shift predictions by the median offset (observed - predicted, log10) in those k samples,
    and score on the city's remaining samples. Compared with using only the k local samples
    (their median, or their median per containment type) and no model at all."""
    rng = np.random.default_rng(SEED)
    repeats = 20 if quick else CAL_REPEATS
    models = make_models(True)
    feats = FEATURE_SETS["survey"]
    rows = []
    for target in TARGETS:
        sub = target_frame(d, target, feats)
        cities = [c for c, n in sub["city"].value_counts().items() if n >= 20]
        y_all = np.log10(sub[target].values)
        for city in cities:
            te = (sub["city"] == city).values
            Xte, yte = sub.loc[te, feats], y_all[te]
            pit = Xte["is_pit"].values
            n = len(yte)
            # shrinkage constant k0 = within-city variance / between-city variance of the log target,
            # estimated on the training cities only (empirical-Bayes partial pooling of the city offset)
            tr_df = pd.DataFrame({"city": sub.loc[~te, "city"].values, "y": y_all[~te]})
            within = tr_df.groupby("city")["y"].var().mean()
            between = max(tr_df.groupby("city")["y"].median().var(), 1e-6)
            k0 = within / between
            pooled = {}
            for name in CAL_MODELS:
                p, _ = fit_predict(name, models[name], sub.loc[~te, feats], y_all[~te], Xte, sub.loc[~te, "city"])
                pooled[name] = p
            for k in KS:
                if k > n - 10:
                    continue
                for r in range(repeats if k > 0 else 1):
                    idx = rng.permutation(n)
                    cal, ev = idx[:k], idx[k:]
                    res = {}
                    for name, p in pooled.items():
                        off = np.median(yte[cal] - p[cal]) if k > 0 else 0.0
                        res[f"pooled_{name}+offset"] = p[ev] + off
                        if name in ("random_forest", "B1_containment_rule"):
                            res[f"pooled_{name}+shrunk_offset"] = p[ev] + off * k / (k + k0)
                    if k > 0:
                        res["local_median_only"] = np.full(len(ev), np.median(yte[cal]))
                        lm = np.full(len(ev), np.median(yte[cal]))
                        for v in (0.0, 1.0):  # local median per containment type (pit vs not), if >= 2 local samples
                            m = pit[cal] == v
                            if m.sum() >= 2:
                                lm[pit[ev] == v] = np.median(yte[cal][m])
                        res["local_containment_rule_only"] = lm
                    for strat, pe in res.items():
                        rows.append({"target": target, "test_city": city, "k_local": k, "repeat": r,
                                     "strategy": strat, **metrics(yte[ev], pe)})
            print(f"  calibration {target:8s} {city}")
    raw = pd.DataFrame(rows)
    agg = (raw.groupby(["target", "test_city", "strategy", "k_local"])
           .agg(n_eval=("n", "median"), R2_log=("R2_log", "mean"), RMSE_log10=("RMSE_log10", "mean"),
                within_factor2_pct=("within_factor2_pct", "mean"),
                f2_p10=("within_factor2_pct", lambda x: np.percentile(x, 10)),
                f2_p90=("within_factor2_pct", lambda x: np.percentile(x, 90)),
                abs_bias_log10=("bias_log10", lambda x: np.mean(np.abs(x))))
           .reset_index())
    return agg


def calibration_figure(agg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    styles = {"pooled_xgboost+offset": ("XGBoost (other cities) + local offset", "-", "o"),
              "pooled_random_forest+offset": ("Random forest (other cities) + local offset", "-", "s"),
              "pooled_B1_containment_rule+offset": ("Containment rule (other cities) + local offset", "-", "^"),
              "pooled_random_forest+shrunk_offset": ("Random forest + shrunk local offset (partial pooling)", "-", "D"),
              "local_containment_rule_only": ("Local samples only: median per containment", "--", "v"),
              "local_median_only": ("Local samples only: median", ":", "x")}
    city_avg = _consistent(agg, K_ALL).groupby(["target", "strategy", "k_local"])[["within_factor2_pct", "RMSE_log10"]].mean().reset_index()
    fig, axes = plt.subplots(2, len(TARGETS), figsize=(4.4 * len(TARGETS), 8), sharex=True)
    for j, (t, lab) in enumerate(TARGETS.items()):
        g = city_avg[city_avg.target == t]
        for strat, (name, ls, mk) in styles.items():
            s = g[g.strategy == strat].sort_values("k_local")
            if len(s):
                axes[0, j].plot(s.k_local, s.within_factor2_pct, ls=ls, marker=mk, label=name)
                axes[1, j].plot(s.k_local, s.RMSE_log10, ls=ls, marker=mk, label=name)
        n_c = _consistent(agg, K_ALL).query("target == @t").test_city.nunique()
        axes[0, j].set_title(f"{lab}: new city (mean of {n_c} held-out cities)")
        axes[0, j].set_ylabel("% predictions within factor 2")
        axes[1, j].set_ylabel("RMSE (log10)")
        axes[1, j].set_xlabel("local samples measured in the new city (k)")
        axes[1, j].set_xticks([k for k in KS if k <= K_ALL])
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "local_calibration.png", dpi=200); plt.close()


def _consistent(agg, kmax, min_cities=None):
    """Keep only cities that have every k <= kmax, so each column averages the same cities."""
    a = agg[agg.k_local <= kmax]
    need = len([k for k in KS if k <= kmax])
    ok = a.groupby(["target", "test_city"])["k_local"].nunique().reset_index()
    ok = ok[ok.k_local == need][["target", "test_city"]]
    return a.merge(ok, on=["target", "test_city"])


def summarize_calibration(agg):
    print("\n=== Local calibration: % within factor 2 by local samples k ===")
    a = _consistent(agg, K_ALL)
    print(f"-- all cities with k <= {K_ALL} (cities: " +
          "; ".join(f"{t}: {', '.join(sorted(g.test_city.unique()))}" for t, g in a.groupby('target')) + ")")
    print(a.groupby(["target", "strategy", "k_local"])["within_factor2_pct"].mean().round(0).unstack("k_local").to_string())
    b = _consistent(agg[agg.test_city.isin(["Lusaka", "Kampala"])], max(KS))
    if len(b):
        print("-- Lusaka and Kampala only, k up to 40")
        print(b.groupby(["target", "strategy", "k_local"])["within_factor2_pct"].mean().round(0).unstack("k_local").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--calibration-only", action="store_true", help="run only the local-calibration analysis")
    args = ap.parse_args()
    d = load()
    if args.calibration_only:
        print("Local calibration analysis ...")
        agg = local_calibration(d, args.quick)
        agg.to_csv(OUT / "local_calibration.csv", index=False)
        calibration_figure(agg)
        summarize_calibration(agg)
        return
    models = make_models(args.quick)
    print(f"Samples: {len(d)} ({(d.region == 'SSA').sum()} SSA). Models: {list(models)}")

    print("\n[1/5] Leave-one-city-out ...")
    per_city, pooled, allp = loco(d, models, args.quick)
    per_city.to_csv(OUT / "loco_metrics_by_city.csv", index=False)
    pooled.to_csv(OUT / "loco_metrics_pooled.csv", index=False)
    allp.to_csv(OUT / "loco_predictions.csv", index=False)

    print("[2/5] Within-city random CV (comparison) ...")
    wc = within_city_cv(d, models, repeats=2 if args.quick else 5)
    wc.to_csv(OUT / "within_city_cv_metrics.csv", index=False)

    print("[3/5] External test (India, Lebanon, Vietnam) ...")
    ex = external(d, models)
    ex.to_csv(OUT / "external_test_metrics.csv", index=False)

    print("[4/5] Conformal prediction intervals ...")
    cf = conformal_loco(d)
    cf.to_csv(OUT / "conformal_intervals_loco.csv", index=False)

    print("[5/5] Feature importance (SHAP) and figures ...")
    imp = importance(d)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    figures(pooled, allp)

    print("[6/6] Local calibration: how many samples does a new city need? ...")
    agg = local_calibration(d, args.quick)
    agg.to_csv(OUT / "local_calibration.csv", index=False)
    calibration_figure(agg)

    pd.set_option("display.width", 220, "display.max_rows", 200)
    cols = ["n", "R2_log", "RMSE_log10", "MdAPE_pct", "within_factor2_pct"]
    print("\n=== LOCO, pooled over held-out cities ===")
    print(pooled.set_index(["target", "features", "model"])[cols].round(2).to_string())
    print("\n=== Within-city random CV (optimistic design) ===")
    print(wc.set_index(["target", "features", "city", "model"])[cols].round(2).to_string())
    print("\n=== External test ===")
    print(ex.set_index(["target", "features", "test_set", "model"])[cols].round(2).to_string())
    print("\n=== 90% conformal intervals (xgboost, LOCO) ===")
    print(cf.round(2).to_string(index=False))
    print("\n=== Top predictors ===")
    print(imp.groupby("target").head(6).round(3).to_string(index=False))
    summarize_calibration(agg)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
