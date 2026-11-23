"""Write LaTeX supplementary tables from result CSVs (no hand-typed numbers)."""
from pathlib import Path
import numpy as np, pandas as pd
import sys; sys.path.insert(0, str(Path(__file__).parent))
from paths import REV as RV, MODEL as RS, TAB as T
TL = {"ts_gL": "TS", "vs_gL": "VS", "cod_gL": "COD", "nh4_mgL": r"\NHfour"}
def w(name, s): (T / name).write_text(s); print("wrote", name)
def f(x, d=2):
    return "--" if pd.isna(x) else (f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}")

# S1 source x city
s = pd.read_csv(RV / "R2_source_by_city.csv")
pt = {"in containment": "In containment", "truck": "Truck, at discharge", "not stated": "Not stated (to be confirmed)", "not recorded": "Not recorded"}
rows = []
for _, r in s.sort_values(["city", "n"], ascending=[True, False]).iterrows():
    p = next(v for k, v in pt.items() if r.sampling_point.startswith(k))
    rows.append(f"{r.city} & {r.source} & {r.n} & {r.n_TS} & {f(r.TS_median,1)} & {f(r.COD_median,1)} & {f(r.pit_pct,0)} & {p} & {r.establishments} \\\\")
w("tabS1_sources.tex", "\n".join(rows))

# S2 conformal (full run)
c = pd.read_csv(RS / "conformal_intervals_loco.csv"); c = c[c.features == "survey"]
rows = []
for t in TL:
    for _, r in c[c.target == t].iterrows():
        rows.append(f"{TL[t]} & {r.test_city} & {r.n} & {r.halfwidth_log10:.2f} & {r.interval_factor:.1f} & {r.coverage_pct:.0f} \\\\")
w("tabS2_conformal.tex", "\n".join(rows))

# S3 external
e = pd.read_csv(RS / "external_test_metrics.csv"); e = e[e.features == "survey"]
ms = ["B0_pooled_median", "B1_containment_rule", "ridge", "random_forest", "xgboost", "gaussian_process"]
rows = []
for (t, g), x in e.groupby(["target", "test_set"], sort=False):
    x = x.set_index("model")
    cells = " & ".join(f"{f(x.R2_log.get(m))} ({x.within_factor2_pct.get(m, np.nan):.0f})" if m in x.index else "--" for m in ms)
    g2 = g.replace("Global FS Database (India, Nepal, Indonesia, Thailand, Uruguay)", "GFSD sites$^{a}$")
    rows.append(f"{TL[t]} & {g2} & {int(x.n.iloc[0])} & {cells} \\\\")
w("tabS3_external.tex", "\n".join(rows))

# S4 hyperparameters
h = pd.read_csv(RV / "R6_hyperparameter_sensitivity.csv")
rows = [f"{TL[r.target]} & {'Random forest' if r.model=='random_forest' else 'XGBoost'} & {r.setting.replace('_', ' ')} & {f(r.R2_log)} \\\\" for _, r in h.iterrows()]
w("tabS4_hyper.tex", "\n".join(rows))

# S5 per-city CIs + LMM
ci = pd.read_csv(RV / "R7_loco_r2_confidence_intervals.csv"); ci = ci[ci.model == "random_forest"]
lm = pd.read_csv(RV / "R5_mixed_model_loco.csv")
rows = []
for t in TL:
    for _, r in ci[ci.target == t].iterrows():
        city = "Pooled$^{b}$" if r.test_city.startswith("POOLED") else r.test_city
        key = "POOLED" if r.test_city.startswith("POOLED") else r.test_city
        l1 = lm[(lm.target == t) & (lm.model == "LMM-containment") & (lm.test_city == key)].R2_log
        l2 = lm[(lm.target == t) & (lm.model == "LMM-survey") & (lm.test_city == key)].R2_log
        rows.append(f"{TL[t]} & {city} & {r.n} & {f(r.R2_log)} [{f(r.ci_lo)}, {f(r.ci_hi)}] & {f(l1.iloc[0]) if len(l1) else '--'} & {f(l2.iloc[0]) if len(l2) else '--'} \\\\")
w("tabS5_ci_lmm.tex", "\n".join(rows))

# S6 SHAP consistency
sh = pd.read_csv(RV / "R9_shap_consistency.csv")
nice = {"log_ec": "EC", "flush_toilet": "flush toilet", "log_sludge_age": "sludge age", "ph": "pH", "household": "household",
        "water_connection": "piped water", "never_emptied": "never emptied", "watertight": "watertight", "is_pit": "pit"}
rows = []
for _, r in sh[~sh.comparison.str.startswith("predictor")].iterrows():
    top = ", ".join(nice.get(x.strip(), x.strip()) for x in str(r.top3).split(","))
    rows.append(f"{TL[r.target]} & {r.comparison.replace('city subset: ', 'Samples from ').replace('model refitted without', 'Model refitted without')} & {r.n} & {r.spearman_vs_overall:.2f} & {top} \\\\")
w("tabS6_shap.tex", "\n".join(rows))

# Table 3 engineering
en = pd.read_csv(RV / "R10_engineering_sensitivity.csv")
rows = []
for q, lab in (("drying-bed area, m2", "area"), ("methane potential, m3 CH4/d", "ch4")):
    for _, r in en[en.quantity == q].iterrows():
        rows.append((lab, r.city, r.design_value_observed, r.design_value_uncalibrated, r.error_pct_uncalibrated,
                     r.error_pct_k5_median, r.error_pct_k5_p10, r.error_pct_k5_p90))
df = pd.DataFrame(rows, columns=["q", "city", "obs", "unc", "eu", "e5", "e5lo", "e5hi"])
a, m = df[df.q == "area"].set_index("city"), df[df.q == "ch4"].set_index("city")
pc = lambda x: f"$+${x:.0f}" if x > 0 else f"$-${abs(x):.0f}"
out = []
for c_ in a.index:
    out.append(f"{c_} & {a.obs[c_]:,.0f} & {a.unc[c_]:,.0f} & {pc(a.eu[c_])} & {pc(a.e5[c_])} [{pc(a.e5lo[c_])}, {pc(a.e5hi[c_])}] & "
               f"{m.obs[c_]:,.0f} & {pc(m.eu[c_])} & {pc(m.e5[c_])} \\\\")
w("tab3_engineering.tex", "\n".join(out))

# S2 predictor availability (% of analytical samples with the predictor recorded)
av = pd.read_csv(RV / "R1_predictor_availability_pct_by_city.csv").set_index("city").T
nice = {"is_pit": "Containment type", "watertight": "Watertight lining", "log_volume": "Containment volume",
        "log_users": "Number of users", "household": "Household use", "flush_toilet": "Flush toilet",
        "greywater": "Greywater input", "water_connection": "Piped water on premises", "solid_waste": "Solid waste present",
        "additives": "Additives used", "never_emptied": "Ever emptied", "log_sludge_age": "Sludge age",
        "ph": "pH (field probe)", "log_ec": "EC (field probe)"}
av = av.drop(index=["is_septic", "is_other"]).rename(index=nice)
cols = ["Lusaka", "Kampala", "Dar es Salaam", "Ouagadougou", "Kumasi", "Naivasha", "Freetown", "Pietermaritzburg"]
w("tabS2_availability.tex", "\n".join(f"{i} & " + " & ".join(f"{v:.0f}" for v in av.loc[i, cols]) + " \\\\" for i in av.index))

# ---- audit revision (round 2) ----
# S5 conformal, corrected definitions: q = half-width (log10); factor 10^q applies in each direction; upper/lower = 10^(2q)
c = pd.read_csv(RS / "conformal_intervals_loco.csv"); c = c[c.features == "survey"]
rows = []
for t in TL:
    for _, r in c[c.target == t].iterrows():
        rows.append(f"{TL[t]} & {r.test_city} & {r.n} & {r.halfwidth_log10:.2f} & {10**r.halfwidth_log10:.0f} & {10**(2*r.halfwidth_log10):,.0f} & {r.coverage_pct:.0f} \\\\")
w("tabS2_conformal.tex", "\n".join(rows))

# S8 city-level performance
pr = pd.read_csv(RS / "loco_predictions.csv"); pr = pr[pr.features == "survey"].copy()
pr["y"], pr["p"] = np.log10(pr.observed), np.log10(pr.predicted)
r2_ = lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)
f2_ = lambda y, p: 100 * np.mean(np.abs(p - y) <= np.log10(2))
ML = {"B0_pooled_median": "Pooled median (B0)", "B1_containment_rule": "Containment rule (B1)", "random_forest": "Random forest", "xgboost": "XGBoost"}
rows = []
for t in TL:
    g = pr[(pr.target == t) & (pr.model == "random_forest")]
    for cty, gc in g.groupby("test_city"):
        rows.append(f"{TL[t]} & {cty} & {len(gc)} & {f(r2_(gc.y, gc.p))} & {np.sqrt(np.mean((gc.y-gc.p)**2)):.2f} & {f2_(gc.y, gc.p):.0f} & {f(np.mean(gc.p-gc.y))} \\\\")
    rows.append("\\midrule")
w("tabS8_city_rf.tex", "\n".join(rows[:-1]))
rows = []
for t in TL:
    for m, lab in ML.items():
        g = pr[(pr.target == t) & (pr.model == m)]
        pc = pd.DataFrame([dict(R2=r2_(gc.y, gc.p), F2=f2_(gc.y, gc.p)) for _, gc in g.groupby("test_city")])
        nl = g[g.test_city != "Lusaka"]
        rows.append(f"{TL[t]} & {lab} & {g.test_city.nunique()} & {f(r2_(g.y, g.p))} & {f(pc.R2.mean())} & {f(pc.R2.median())} & {f(r2_(nl.y, nl.p))} & {f2_(g.y, g.p):.0f} & {pc.F2.mean():.0f} \\\\")
    rows.append("\\midrule")
w("tabS8_summary.tex", "\n".join(rows[:-1]))

# S9 matched stratified vs random
mt = pd.read_csv(RV / "R13_stratified_vs_random_matched.csv")
rows = []
for _, r in mt.iterrows():
    rows.append(f"{TL[r.target]} & {int(r.total)} & {'Stratified' if r.design=='stratified' else 'Random'} & {int(r.cities)} & {f(r.gain,1)} & [{f(r.p10,1)}, {f(r.p90,1)}] & {r.improved:.0f} \\\\")
w("tabS9_matched.tex", "\n".join(rows))

# Table 3 with the medians actually used (LOCO evaluation subset)
en = pd.read_csv(RV / "R10_engineering_sensitivity.csv")
a, m = en[en.quantity.str.startswith("drying")].set_index("city"), en[en.quantity.str.startswith("methane")].set_index("city")
pc = lambda x: f"$+${x:.0f}" if x > 0 else f"$-${abs(x):.0f}"
out = []
for c_ in a.index:
    out.append(f"{c_} & {a.observed_median[c_]:.1f} & {a.design_value_observed[c_]:,.0f} & {pc(a.error_pct_uncalibrated[c_])} & {pc(a.error_pct_k5_median[c_])} [{pc(a.error_pct_k5_p10[c_])}, {pc(a.error_pct_k5_p90[c_])}] & "
               f"{m.observed_median[c_]:.1f} & {m.design_value_observed[c_]:,.0f} & {pc(m.error_pct_uncalibrated[c_])} & {pc(m.error_pct_k5_median[c_])} \\\\")
w("tab3_engineering.tex", "\n".join(out))
