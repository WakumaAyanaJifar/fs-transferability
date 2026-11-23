"""Additional analyses requested in peer review (round 1).
Writes outputs/revision/*.csv. Reuses the data preparation and models of 04_model.py.

R1  sample flow and target-specific counts            (MC1)
R2  source x city table, sampling point, establishment (MC5)
R3  variance decomposition: city vs source-within-city (MC5)
R4  LOCO sensitivity: in-situ household samples only; leave-one-source-out (MC5)
R5  linear mixed model (random city intercept) under LOCO (MC8)
R6  hyperparameter sensitivity of RF and XGBoost under LOCO (MC8)
R7  per-city LOCO R2 with bootstrap CIs; pooled R2 with city-cluster bootstrap (MC2)
R8  local calibration: random vs containment-stratified draws, with uncertainty (MC3)
R9  SHAP consistency across cities (MC9)
R10 engineering sensitivity: drying-bed area and methane potential (MC10)
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m4", ROOT / "04_model.py")
m4 = importlib.util.module_from_spec(spec); sys.modules["m4"] = m4; spec.loader.exec_module(m4)
OUT = ROOT / "outputs" / "revision"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(m4.SEED)
SURVEY = m4.FEATURE_SETS["survey"]

# Sampling point and establishment type, from the source publications / data packages
SOURCE_INFO = {
    "ward2021_lusaka": ("Ward et al. (2021)", "in containment (pit sampler / core sampler, composite over depth)", "households and non-households"),
    "volaser_7countries": ("Andriessen et al. (2023)", "in containment (pit sampler at 3 depths / core sampler, composite)", "households and non-households"),
    "englund2020_kampala": ("Englund et al. (2020)", "truck, at discharge (Bugolobi WWTP)", "households and non-households"),
    "ward2022_kampala_naivasha": ("Ward et al. (2023)", "in containment (Kampala: multi-stage sampler, top/middle/bottom composite); truck, at discharge (Naivasha: containment fully emptied, start/middle/end composite)", "households, offices, public toilets, hotel, campsite, place of worship; sites selected for diversity"),
    "fecalcanuga_kampala": ("Shaw et al. (2026)", "in containment (3 depths, averaged)", "households"),
    "sludgedata_bassan_2013": ("Bassan et al. (2013) via GFSD", "truck, at discharge (composite of 4 sub-samples)", "not recorded"),
    "sludgedata_esanju": ("GFSD laboratory 22", "not recorded", "restaurants and hotels"),
    "sludgedata_marwa": ("GFSD laboratory 24", "not recorded", "restaurants and hotels"),
    "sludgedata_wash": ("GFSD laboratory 37", "not recorded", "not recorded"),
}
TRUCK = {"englund2020_kampala", "sludgedata_bassan_2013"}
NON_HH = {"sludgedata_esanju", "sludgedata_marwa"}
UNKNOWN = {"sludgedata_wash"}
# Ward et al. (2023, J. Environ. Manage. 326, 116801, Section 2.1): Kampala sampled in containment, Naivasha from the truck
CITY_POINT = {("ward2022_kampala_naivasha", "Kampala"): "in containment (multi-stage sampler, top/middle/bottom composite)",
              ("ward2022_kampala_naivasha", "Naivasha"): "truck, at discharge (containment fully emptied; start/middle/end composite)"}
TRUCK_CITY = {("ward2022_kampala_naivasha", "Naivasha")}


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ------------------------------------------------------------------ R1 sample flow
def sample_flow():
    raw = pd.read_csv(m4.TABLE)
    rows = [("Harmonised samples, all sources", len(raw)),
            ("  outside Sub-Saharan Africa (external test only)", int((raw.region != "SSA").sum())),
            ("  Sub-Saharan Africa", int((raw.region == "SSA").sum())),
            ("  SSA, author-flagged outliers removed (Englund Kampala)", int(((raw.region == "SSA") & (raw.outlier_flag == 1)).sum())),
            ("SSA analytical samples", int(((raw.region == "SSA") & (raw.outlier_flag.fillna(0) == 0)).sum()))]
    d = m4.load(); s = d[d.region == "SSA"]
    for t in m4.TARGETS:
        rows.append((f"  with measured {t}", int(s[t].notna().sum())))
        sub = m4.target_frame(d, t, SURVEY)
        rows.append((f"  with {t} and >=50% survey predictors (LOCO set)", len(sub)))
    pd.DataFrame(rows, columns=["stage", "n"]).to_csv(OUT / "R1_sample_flow.csv", index=False)
    by_city = s.groupby("city").agg(samples=("sample_id", "size"), TS=("ts_gL", "count"), VS=("vs_gL", "count"),
                                    COD=("cod_gL", "count"), NH4=("nh4_mgL", "count"))
    for t in m4.TARGETS:
        by_city[f"LOCO_{t}"] = m4.target_frame(d, t, SURVEY).city.value_counts()
    by_city.fillna(0).astype(int).to_csv(OUT / "R1_counts_by_city.csv")
    miss = s.groupby("city")[SURVEY + m4.FIELD].apply(lambda g: g.notna().mean() * 100).round(0)
    miss.to_csv(OUT / "R1_predictor_availability_pct_by_city.csv")
    print(pd.DataFrame(rows), "\n", by_city)


# ------------------------------------------------------------------ R2 source x city
def source_table(d):
    s = d[d.region == "SSA"]
    g = s.groupby(["city", "dataset"]).agg(n=("sample_id", "size"), n_TS=("ts_gL", "count"),
                                           TS_median=("ts_gL", "median"), COD_median=("cod_gL", "median"),
                                           pit_pct=("is_pit", lambda x: 100 * np.nanmean(x))).reset_index()
    g["source"] = g.dataset.map(lambda k: SOURCE_INFO.get(k, (k,))[0])
    g["sampling_point"] = [CITY_POINT.get((k, c), SOURCE_INFO[k][1]) for k, c in zip(g.dataset, g.city)]
    g["establishments"] = g.dataset.map(lambda k: SOURCE_INFO[k][2])
    g.round(1).to_csv(OUT / "R2_source_by_city.csv", index=False)
    print(g[["city", "source", "n", "TS_median", "sampling_point"]])


# ------------------------------------------------------------------ R3 variance decomposition
def variance_decomposition(d):
    import statsmodels.formula.api as smf
    rows = []
    s = d[d.region == "SSA"].copy()
    s["src"] = s.city + "|" + s.dataset
    for t in ["ts_gL", "vs_gL", "cod_gL"]:
        x = s[s[t] > 0].copy(); x["y"] = np.log10(x[t])
        x = x[x.is_pit.notna()]
        for adj in ("none", "containment"):
            form = "y ~ 1" if adj == "none" else "y ~ is_pit + is_other"
            m = smf.mixedlm(form, x, groups="city", re_formula="1", vc_formula={"src": "0 + C(src)"}).fit(reml=True)
            vc, vg, ve = float(m.vcomp[0]), float(m.cov_re.iloc[0, 0]), float(m.scale)
            tot = vc + vg + ve
            rows.append({"target": t, "fixed_effects": adj, "n": len(x), "cities": x.city.nunique(), "sources": x.src.nunique(),
                         "var_city": vg, "var_source_within_city": vc, "var_residual": ve,
                         "pct_city": 100 * vg / tot, "pct_source_within_city": 100 * vc / tot, "pct_residual": 100 * ve / tot})
    r = pd.DataFrame(rows); r.round(3).to_csv(OUT / "R3_variance_decomposition.csv", index=False); print(r.round(2))


# ------------------------------------------------------------------ helpers for LOCO
def loco_run(sub, target, model_names, groups_col="city", min_n=10):
    models = m4.make_models(False)
    y = np.log10(sub[target].values)
    out = []
    for g, n in sub[groups_col].value_counts().items():
        if n < min_n:
            continue
        te = (sub[groups_col] == g).values
        for name in model_names:
            p, _ = m4.fit_predict(name, models[name], sub.loc[~te, SURVEY], y[~te], sub.loc[te, SURVEY], sub.loc[~te, "city"])
            out.append(pd.DataFrame({"group": g, "model": name, "y": y[te], "p": p, "city": sub.loc[te, "city"].values}))
    return pd.concat(out)


def summarise(pr, label):
    rows = []
    for (mname), g in pr.groupby("model"):
        rows.append({"scenario": label, "model": mname, "n": len(g), "groups": g.group.nunique(), "R2_log": r2(g.y.values, g.p.values),
                     "within_factor2_pct": 100 * np.mean(np.abs(g.p - g.y) <= np.log10(2))})
    return rows


# ------------------------------------------------------------------ R4 sensitivity to source / sampling point
def source_sensitivity(d):
    rows, offs = [], []
    for t in ["ts_gL", "cod_gL"]:
        full = m4.target_frame(d, t, SURVEY)
        truck_city = pd.Series([(k, c) in TRUCK_CITY for k, c in zip(full.dataset, full.city)], index=full.index)
        insitu = full[~full.dataset.isin(TRUCK | NON_HH | UNKNOWN) & ~truck_city]
        for label, sub in [("all sources", full), ("in-containment household/mixed sources only", insitu)]:
            pr = loco_run(sub, t, ["B1_containment_rule", "random_forest"])
            rows += [dict(r, target=t) for r in summarise(pr, f"LOCO, {label}")]
            for (c, mn), g in pr[pr.model == "random_forest"].groupby(["group", "model"]):
                offs.append({"target": t, "scenario": label, "city": c, "n": len(g),
                             "offset_pred_over_obs": 10 ** np.median(g.p - g.y)})
        # leave-one-source-out: the held-out source's city may still be represented by other sources
        sub = full.copy(); sub["srcg"] = sub.city + " | " + sub.dataset.map(lambda k: SOURCE_INFO.get(k, (k,))[0])
        pr = loco_run(sub, t, ["random_forest"], groups_col="srcg")
        rows += [dict(r, target=t) for r in summarise(pr, "leave-one-source-out")]
        for gname, g in pr.groupby("group"):
            c = gname.split(" | ")[0]
            offs.append({"target": t, "scenario": "leave-one-source-out", "city": gname, "n": len(g),
                         "offset_pred_over_obs": 10 ** np.median(g.p - g.y),
                         "city_seen_in_training": int((full.city == c).sum() > len(g))})
    pd.DataFrame(rows).round(3).to_csv(OUT / "R4_source_sensitivity_metrics.csv", index=False)
    pd.DataFrame(offs).round(3).to_csv(OUT / "R4_source_sensitivity_offsets.csv", index=False)
    print(pd.DataFrame(rows).round(2)); print(pd.DataFrame(offs).round(2))


# ------------------------------------------------------------------ R5 linear mixed model
def mixed_model(d):
    """Linear mixed models with a random city intercept. For a held-out (unseen) city the random effect is
    unknown and set to zero, i.e. the prediction is the fixed-effect part. Two specifications:
    LMM-containment: containment type only; LMM-survey: all survey predictors, median-imputed on training cities."""
    import statsmodels.api as sm
    rows = []
    specs = {"LMM-containment": ["is_pit", "is_other"], "LMM-survey": SURVEY}
    for t in m4.TARGETS:
        sub = m4.target_frame(d, t, SURVEY)
        cities = [c for c, n in sub.city.value_counts().items() if n >= m4.MIN_TEST]
        if len(cities) < 3:
            continue
        y = np.log10(sub[t].values)
        for spec, cols in specs.items():
            allp, ally = [], []
            for c in cities:
                te = (sub.city == c).values
                X = sub[cols].fillna(sub.loc[~te, cols].median())
                keep = []
                for col in cols:
                    M = np.column_stack([np.ones((~te).sum()), X.loc[~te, keep + [col]].values])
                    if np.linalg.matrix_rank(M) == M.shape[1]:
                        keep.append(col)
                Xtr = sm.add_constant(X.loc[~te, keep], has_constant="add")
                Xte = sm.add_constant(X.loc[te, keep], has_constant="add")
                mdl = sm.MixedLM(y[~te], Xtr, groups=sub.city.values[~te])
                try:
                    fit = mdl.fit(reml=True, method="bfgs")
                except Exception:
                    fit = mdl.fit(reml=True, method="powell")
                p = Xte.values @ fit.fe_params.values
                allp.append(p); ally.append(y[te])
                rows.append({"target": t, "model": spec, "test_city": c, "n": int(te.sum()), "R2_log": r2(y[te], p),
                             "offset_pred_over_obs": 10 ** np.median(p - y[te]),
                             "sd_city_intercept_log10": float(np.sqrt(max(fit.cov_re.iloc[0, 0], 0)))})
            P, Y = np.concatenate(allp), np.concatenate(ally)
            rows.append({"target": t, "model": spec, "test_city": "POOLED", "n": len(Y), "R2_log": r2(Y, P),
                         "within_factor2_pct": 100 * np.mean(np.abs(P - Y) <= np.log10(2))})
    r = pd.DataFrame(rows); r.round(3).to_csv(OUT / "R5_mixed_model_loco.csv", index=False)
    print(r[r.test_city == "POOLED"].round(3)); print(r[r.target == "ts_gL"].round(2))


# ------------------------------------------------------------------ R6 hyperparameter sensitivity
def hyper_sensitivity(d):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    grid = []
    for leaf in (1, 5, 20):
        for mf in (0.33, 0.5, 1.0):
            grid.append(("random_forest", f"min_leaf={leaf}, max_features={mf}",
                         lambda leaf=leaf, mf=mf: m4.imputed(RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, max_features=mf,
                                                                               random_state=m4.SEED, n_jobs=-1))))
    for depth in (2, 3, 6):
        for lr, n in ((0.03, 400), (0.1, 200)):
            grid.append(("xgboost", f"max_depth={depth}, lr={lr}, rounds={n}",
                         lambda depth=depth, lr=lr, n=n: XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=5,
                                                                      subsample=0.8, colsample_bytree=0.8, random_state=m4.SEED, n_jobs=-1)))
    rows = []
    for t in ["ts_gL", "cod_gL"]:
        sub = m4.target_frame(d, t, SURVEY); y = np.log10(sub[t].values)
        cities = [c for c, n in sub.city.value_counts().items() if n >= m4.MIN_TEST]
        for fam, lab, fac in grid:
            P, Y = [], []
            for c in cities:
                te = (sub.city == c).values
                mdl = fac(); mdl.fit(sub.loc[~te, SURVEY], y[~te]); P.append(mdl.predict(sub.loc[te, SURVEY])); Y.append(y[te])
            P, Y = np.concatenate(P), np.concatenate(Y)
            rows.append({"target": t, "model": fam, "setting": lab, "R2_log": r2(Y, P)})
            print(t, fam, lab, round(rows[-1]["R2_log"], 3))
    pd.DataFrame(rows).round(3).to_csv(OUT / "R6_hyperparameter_sensitivity.csv", index=False)


# ------------------------------------------------------------------ R7 uncertainty of LOCO metrics
def loco_uncertainty():
    pr = pd.read_csv(m4.OUT / "loco_predictions.csv")
    pr = pr[(pr.features == "survey") & pr.model.isin(["B0_pooled_median", "B1_containment_rule", "random_forest", "xgboost"])]
    pr["y"], pr["p"] = np.log10(pr.observed), np.log10(pr.predicted)
    rows = []
    for (t, mname), g in pr.groupby(["target", "model"]):
        for c, gc in g.groupby("test_city"):
            bs = [r2(*(lambda i: (gc.y.values[i], gc.p.values[i]))(RNG.integers(0, len(gc), len(gc)))) for _ in range(1000)]
            rows.append({"target": t, "model": mname, "test_city": c, "n": len(gc), "R2_log": r2(gc.y.values, gc.p.values),
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
        cities = g.test_city.unique(); groups = {c: g[g.test_city == c] for c in cities}
        bs = []
        for _ in range(2000):
            pick = RNG.choice(cities, len(cities), replace=True)
            gg = pd.concat([groups[c] for c in pick])
            bs.append(r2(gg.y.values, gg.p.values))
        rows.append({"target": t, "model": mname, "test_city": "POOLED (city-cluster bootstrap)", "n": len(g),
                     "R2_log": r2(g.y.values, g.p.values), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
    r = pd.DataFrame(rows); r.round(3).to_csv(OUT / "R7_loco_r2_confidence_intervals.csv", index=False)
    print(r[r.model == "random_forest"].round(2))


# ------------------------------------------------------------------ R8 local calibration, stratified
def local_calibration_stratified(d, repeats=200, ks=(0, 3, 5, 10, 15), tag="R8", designs=("random (total k)", "stratified (k per containment type)")):
    models = m4.make_models(True)
    rows = []
    for t in m4.TARGETS:
        sub = m4.target_frame(d, t, SURVEY); y = np.log10(sub[t].values)
        for c, n in sub.city.value_counts().items():
            if n < 20:
                continue
            te = (sub.city == c).values
            p, _ = m4.fit_predict("random_forest", models["random_forest"], sub.loc[~te, SURVEY], y[~te], sub.loc[te, SURVEY], None)
            yt = y[te]; pit = sub.loc[te, "is_pit"].values
            strata = {"pit": np.where(pit == 1)[0], "non-pit": np.where(pit == 0)[0]}
            for k in ks:
                for design in designs:
                    if design.startswith("strat") and (k == 0 or min(len(v) for v in strata.values()) < k + 5):
                        continue
                    if design.startswith("random") and k > n - 10:
                        continue
                    for r in range(repeats if k else 1):
                        if design.startswith("random"):
                            idx = RNG.permutation(n); cal = idx[:k]
                            off = np.full(n, np.median(yt[cal] - p[cal]) if k else 0.0)
                        else:
                            cal, off = [], np.zeros(n)
                            for nm, ix in strata.items():
                                pick = RNG.choice(ix, k, replace=False); cal += list(pick)
                                mask = np.isin(np.arange(n), ix)
                                off[mask] = np.median(yt[pick] - p[pick])
                            other = ~np.isin(np.arange(n), np.concatenate(list(strata.values())))
                            off[other] = np.median(yt[cal] - p[cal])
                            cal = np.array(cal)
                        ev = np.setdiff1d(np.arange(n), cal)
                        pe = p[ev] + off[ev]
                        rows.append({"target": t, "city": c, "design": design, "k": k, "k_total": len(cal), "repeat": r,
                                     "f2": 100 * np.mean(np.abs(pe - yt[ev]) <= np.log10(2)),
                                     "abs_bias": abs(np.median(pe - yt[ev])), "R2": r2(yt[ev], pe)})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / f"{tag}_local_calibration_raw.csv.gz", index=False)
    agg = raw.groupby(["target", "city", "design", "k"]).agg(k_total=("k_total", "median"), f2_mean=("f2", "mean"),
                                                             f2_p10=("f2", lambda x: np.percentile(x, 10)),
                                                             f2_p90=("f2", lambda x: np.percentile(x, 90)),
                                                             abs_bias=("abs_bias", "mean")).reset_index()
    agg.round(2).to_csv(OUT / f"{tag}_local_calibration_by_city.csv", index=False)
    # gain over k=0 per draw, summarised across cities
    base = raw[raw.k == 0].groupby(["target", "city"]).f2.mean().rename("f2_k0")
    g = raw[raw.k > 0].merge(base, on=["target", "city"]); g["gain"] = g.f2 - g.f2_k0
    s = g.groupby(["target", "design", "k"]).agg(cities=("city", "nunique"), gain_mean=("gain", "mean"),
                                                 gain_p10=("gain", lambda x: np.percentile(x, 10)),
                                                 gain_p90=("gain", lambda x: np.percentile(x, 90)),
                                                 share_draws_improved=("gain", lambda x: 100 * np.mean(x > 0))).reset_index()
    s.round(1).to_csv(OUT / f"{tag}_local_calibration_gain.csv", index=False); print(s.round(1))


# ------------------------------------------------------------------ R9 SHAP consistency
def shap_by_city(d):
    import shap
    rows, corr = [], []
    fac = m4.make_models(True)["xgboost"]; feats = m4.FEATURE_SETS["survey_field"]
    for t in ["ts_gL", "cod_gL", "vs_gL"]:
        sub = m4.target_frame(d, t, feats); y = np.log10(sub[t].values)
        _, mdl = m4.fit_predict("xgboost", fac, sub[feats], y, sub[feats].iloc[:1], None)
        sv = pd.DataFrame(np.abs(shap.TreeExplainer(mdl).shap_values(sub[feats])), columns=feats, index=sub.index)
        overall = sv.mean()
        for c, n in sub.city.value_counts().items():
            if n < 20:
                continue
            imp = sv[sub.city == c].mean()
            rows += [{"target": t, "city": c, "feature": f, "mean_abs_shap": v} for f, v in imp.items()]
            corr.append({"target": t, "comparison": f"city subset: {c}", "n": n,
                         "spearman_vs_overall": spearmanr(imp, overall).correlation,
                         "top3": ", ".join(imp.sort_values(ascending=False).index[:3])})
        # refit without the largest city
        big = sub.city.value_counts().index[0]
        keep = (sub.city != big).values
        _, m2 = m4.fit_predict("xgboost", fac, sub.loc[keep, feats], y[keep], sub[feats].iloc[:1], None)
        imp2 = pd.Series(np.abs(shap.TreeExplainer(m2).shap_values(sub.loc[keep, feats])).mean(0), index=feats)
        corr.append({"target": t, "comparison": f"model refitted without {big}", "n": int(keep.sum()),
                     "spearman_vs_overall": spearmanr(imp2, overall).correlation,
                     "top3": ", ".join(imp2.sort_values(ascending=False).index[:3])})
        corr.append({"target": t, "comparison": "overall", "n": len(sub), "spearman_vs_overall": 1.0,
                     "top3": ", ".join(overall.sort_values(ascending=False).index[:3])})
        cm = sub[feats].corr(method="spearman").abs()
        cm = cm.where(~np.eye(len(cm), dtype=bool), 0.0)
        hi = cm.stack().sort_values(ascending=False)
        hi = hi[hi > 0.5]
        for (a, b), v in hi.iloc[::2].items():
            corr.append({"target": t, "comparison": f"predictor correlation {a} ~ {b}", "n": len(sub), "spearman_vs_overall": v, "top3": ""})
    pd.DataFrame(rows).round(4).to_csv(OUT / "R9_shap_by_city.csv", index=False)
    r = pd.DataFrame(corr); r.round(3).to_csv(OUT / "R9_shap_consistency.csv", index=False); print(r.round(2))


# ------------------------------------------------------------------ R10 engineering sensitivity
def engineering():
    """Drying-bed area and methane potential implied by observed vs predicted characteristics.
    Area A = Q * TS / SLR, with Q = 50 m3/d of fecal sludge and SLR = 200 kg TS m-2 yr-1 (unplanted drying beds;
    Strande et al., 2014). Methane potential = Q * COD * 0.35 m3 CH4 per kg COD (theoretical, STP) -- an upper bound
    used only to show proportional error. Scenarios: observed city median; uncalibrated LOCO random-forest prediction
    (median); after local calibration with k = 5 random samples (median over draws); conformal 90% interval."""
    Q, SLR, YCH4 = 50.0, 200.0, 0.35
    pr = pd.read_csv(m4.OUT / "loco_predictions.csv")
    pr = pr[(pr.features == "survey") & (pr.model == "random_forest")]
    conf = pd.read_csv(m4.OUT / "conformal_intervals_loco.csv"); conf = conf[conf.features == "survey"]
    raw = pd.read_csv(OUT / "R8_local_calibration_raw.csv.gz") if (OUT / "R8_local_calibration_raw.csv.gz").exists() else None
    rows = []
    for t, unit in (("ts_gL", "drying-bed area, m2"), ("cod_gL", "methane potential, m3 CH4/d")):
        for c, g in pr[pr.target == t].groupby("test_city"):
            obs, pred = np.median(g.observed), np.median(g.predicted)
            f = (lambda x: Q * x * 365 / SLR) if t == "ts_gL" else (lambda x: Q * x * YCH4)
            cf = conf[(conf.target == t) & (conf.test_city == c)]
            fac = float(cf.interval_factor.iloc[0]) if len(cf) else np.nan
            yl, pl = np.log10(g.observed.values), np.log10(g.predicted.values)
            errs = []
            for _ in range(500):                          # k = 5 random local samples, median offset
                idx = RNG.permutation(len(yl)); cal, ev = idx[:5], idx[5:]
                pc = 10 ** np.median(pl[ev] + np.median(yl[cal] - pl[cal]))
                errs.append(100 * (pc / np.median(10 ** yl[ev]) - 1))
            rows.append({"quantity": unit, "city": c, "observed_median": obs, "design_value_observed": f(obs),
                         "error_pct_k5_median": np.median(errs), "error_pct_k5_p10": np.percentile(errs, 10),
                         "error_pct_k5_p90": np.percentile(errs, 90),
                         "predicted_median_uncalibrated": pred, "design_value_uncalibrated": f(pred),
                         "error_pct_uncalibrated": 100 * (pred / obs - 1),
                         "design_value_90pct_low": f(pred / fac), "design_value_90pct_high": f(pred * fac)})
    r = pd.DataFrame(rows); r.round(2).to_csv(OUT / "R10_engineering_sensitivity.csv", index=False); print(r.round(1))


if __name__ == "__main__" and "R11" not in sys.argv and "R13" not in sys.argv:
    which = sys.argv[1:] or ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
    d = m4.load()
    for k in which:
        print(f"\n===== {k}")
        {"R1": sample_flow, "R2": lambda: source_table(d), "R3": lambda: variance_decomposition(d),
         "R4": lambda: source_sensitivity(d), "R5": lambda: mixed_model(d), "R6": lambda: hyper_sensitivity(d),
         "R7": loco_uncertainty, "R8": lambda: local_calibration_stratified(d), "R9": lambda: shap_by_city(d),
         "R10": engineering}[k]()


# ------------------------------------------------------------------ R11 three transfer scenarios (pooled R2)
def transfer_scenarios(k=5, repeats=200):
    """Pooled LOCO R2 for random forest (survey predictors) under: uncalibrated; operational local calibration
    with k random local samples (scored on the remaining samples, mean over draws); oracle offset (the held-out
    city's own median residual, i.e. a diagnostic that uses the test data)."""
    pr = pd.read_csv(m4.OUT / "loco_predictions.csv")
    pr = pr[(pr.features == "survey") & (pr.model == "random_forest")]
    rows = []
    for t, g in pr.groupby("target"):
        g = g.assign(y=np.log10(g.observed), p=np.log10(g.predicted))
        orc = g.p + g.groupby("test_city").apply(lambda x: np.median(x.y - x.p)).reindex(g.test_city).values
        f2 = lambda y, p: 100 * np.mean(np.abs(p - y) <= np.log10(2))
        rows.append({"target": t, "scenario": "uncalibrated", "R2_log": r2(g.y.values, g.p.values), "f2": f2(g.y, g.p)})
        r2s, f2s = [], []
        for _ in range(repeats):
            Y, P = [], []
            for c, gc in g.groupby("test_city"):
                idx = RNG.permutation(len(gc)); cal, ev = idx[:k], idx[k:]
                off = np.median(gc.y.values[cal] - gc.p.values[cal])
                Y.append(gc.y.values[ev]); P.append(gc.p.values[ev] + off)
            Y, P = np.concatenate(Y), np.concatenate(P); r2s.append(r2(Y, P)); f2s.append(f2(Y, P))
        rows.append({"target": t, "scenario": f"operational, k={k} local samples", "R2_log": np.mean(r2s),
                     "R2_p10": np.percentile(r2s, 10), "R2_p90": np.percentile(r2s, 90), "f2": np.mean(f2s)})
        rows.append({"target": t, "scenario": "oracle offset (diagnostic)", "R2_log": r2(g.y.values, orc.values), "f2": f2(g.y, orc)})
    r = pd.DataFrame(rows); r.round(3).to_csv(OUT / "R11_transfer_scenarios.csv", index=False); print(r.round(2))


if __name__ == "__main__" and "R11" in sys.argv:
    transfer_scenarios()

if __name__ == "__main__" and "R13" in sys.argv:
    # matched comparison: random draws with the same totals as the stratified design (6, 10, 20)
    local_calibration_stratified(m4.load(), ks=(0, 6, 10, 20), tag="R13", designs=("random (total k)",))
