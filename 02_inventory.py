"""
02_inventory.py
Read every downloaded dataset, put the key variables in common units, and report:
  outputs/standardized_samples.csv  one row per sample, common columns and units
  outputs/inventory_counts.csv      samples per dataset x city x target
  outputs/possible_duplicates.csv   sample pairs across datasets that look identical
  outputs/sircilla_columns.txt      sheet/column list for Sircilla (structure not yet mapped)

Common units: TS, VS, COD in g/L; NH4-N in mg/L; EC in mS/cm; CST in s.
Run after 01_download_data.py:   python 02_inventory.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

COLS = ["dataset", "city", "country", "sample_id", "containment", "ph", "ec_mScm",
        "ts_gL", "ts_gL_source", "vs_gL", "cod_gL", "nh4_mgL", "cst_s"]
SSA = {"Zambia", "Uganda", "Ghana", "Sierra Leone", "Kenya"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def containment(x):
    x = str(x).lower()
    if "septic" in x:
        return "septic tank"
    if "pit" in x or "latrine" in x:
        return "pit latrine"
    if x in ("nan", "", "none"):
        return np.nan
    return "other"


def frame(**kw):
    df = pd.DataFrame(kw)
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    return df[COLS]


def ward2021():
    w = pd.read_csv(RAW / "ward2021_lusaka/mastersheet_final_ward2020.csv", encoding="latin-1", low_memory=False)
    ts = num(w["ts_gL"])
    # ts_% is available for more samples; g/L = %wt x 10 x density (density assumed 1 g/mL if missing)
    ts_from_pct = num(w["ts_%"]) * 10 * num(w["density"]).fillna(1.0)
    src = np.where(ts.notna(), "measured g/L", np.where(ts_from_pct.notna(), "from %wt", None))
    ts_all = ts.fillna(ts_from_pct)
    return frame(dataset="ward2021_lusaka", city="Lusaka", country="Zambia", sample_id=w["sample_id"],
                 containment=w["oss_type_simplified"].map(containment), ph=num(w["ph"]), ec_mScm=num(w["ec"]),
                 ts_gL=ts_all, ts_gL_source=src, vs_gL=ts_all * num(w["vs_%ts"]) / 100,
                 cod_gL=num(w["cod"]), nh4_mgL=num(w["nh4_mg"]), cst_s=num(w["cst"]))


def lusaka_microbial():
    f = RAW / "lusaka_microbial/metadata.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f, encoding="latin-1", low_memory=False)
    ts = num(m.get("ts_gL"))
    ts_from_pct = num(m.get("ts")) * 10 * num(m.get("density")).fillna(1.0)  # 'ts' column is TS in %wt
    src = np.where(ts.notna(), "measured g/L", np.where(ts_from_pct.notna(), "from %wt", None))
    ts_all = ts.fillna(ts_from_pct)
    vsts = num(m["vs_ts"]) if "vs_ts" in m else np.nan
    return frame(dataset="lusaka_microbial", city="Lusaka", country="Zambia", sample_id=m["sampleID"],
                 containment=m["oss_type_simplified"].map(containment), ph=num(m["ph"]), ec_mScm=num(m["ec"]),
                 ts_gL=ts_all, ts_gL_source=src, vs_gL=ts_all * vsts / 100,
                 cod_gL=num(m["cod_mg"]) / 1000, nh4_mgL=num(m["nh4_mg"]), cst_s=num(m["cst"]))


def volaser():
    v = pd.read_csv(RAW / "volaser_7countries/complete_dataset_volaser.csv", encoding="latin-1", low_memory=False)
    v = v[v["country"].isin(SSA | {"India", "Lebanon"})]
    return frame(dataset="volaser_7countries", city=v["city"].str.strip(), country=v["country"], sample_id=v["sample_id"],
                 containment=v["cont_type"].map(containment), ph=num(v["ph"]), ec_mScm=num(v["ec"]),
                 ts_gL=num(v["ts_gl"]), ts_gL_source="measured g/L", vs_gL=num(v["vs_gl"]),
                 cod_gL=num(v["cod"]) / 1000)


def englund_kampala():
    k = pd.read_excel(RAW / "englund2020_kampala_hanoi/kampala_data.xlsx").iloc[1:]  # row 1 holds units
    return frame(dataset="englund2020_kampala", city="Kampala", country="Uganda", sample_id=k["code"],
                 containment=k["CoTyp"].map(containment), ph=num(k["pH"]),  # no EC; column 'C' is colour
                 ts_gL=num(k["TS g"]), ts_gL_source="measured g/L", vs_gL=num(k["VS g"]),
                 cod_gL=num(k["COD g"]), nh4_mgL=num(k["NH4N mg"]))


def englund_hanoi():
    x = pd.ExcelFile(RAW / "englund2020_kampala_hanoi/hanoi_data.xlsx")
    parts = []
    for sh in x.sheet_names:
        d = x.parse(sh, header=1).iloc[1:]  # header in row 2, units in row 3
        d.columns = [str(c).strip() for c in d.columns]
        d = d.rename(columns={d.columns[0]: "code"})
        d = d[d["code"].notna()]
        g = lambda c: num(d[c]) if c in d else np.nan
        parts.append(frame(dataset="englund2020_hanoi", city="Hanoi", country="Vietnam",
                           sample_id=sh.strip() + ":" + d["code"].astype(str), containment="septic tank",
                           ph=g("pH"), ec_mScm=g("EC") / 1000, ts_gL=g("TS"), ts_gL_source="measured g/L",
                           vs_gL=g("VS"), cod_gL=g("COD") / 1000, nh4_mgL=g("N-NH4")))
    return pd.concat(parts)


def ward2022():
    p = pd.read_csv(RAW / "ward2022_kampala_naivasha/data_field_samples.csv", encoding="latin-1")
    country = p["City"].map({"Kampala": "Uganda", "Naivasha": "Kenya"})
    cod = num(p["COD_gL"]) if "COD_gL" in p else np.nan
    return frame(dataset="ward2022_kampala_naivasha", city=p["City"], country=country, sample_id=p.iloc[:, 0],
                 containment=p["Containment_type"].map(containment), ph=num(p["pH"]), ec_mScm=num(p["EC_mScm"]),
                 ts_gL=num(p["TS_gL"]), ts_gL_source="measured g/L", vs_gL=num(p["VS_gL"]), cod_gL=cod,
                 nh4_mgL=num(p["NH4N_mgL"]), cst_s=num(p["CST_s"]))


def fecalcanuga():
    p = pd.read_csv(RAW / "fecalcanuga/phys_chem_parameter.csv")
    s = pd.read_csv(RAW / "fecalcanuga/household_survey.csv")[["sample_id", "local_area_name", "containment"]]
    p = p[p["date"].astype(str).str.startswith("2024")]  # 2024 = Kampala campaign; 2023 = Canada
    a = p.groupby("sample_id")[["pH", "EC", "ts", "vs", "COD", "ammonia"]].mean().reset_index()  # mean of depths
    a = a.merge(s, on="sample_id", how="left")
    return frame(dataset="fecalcanuga_kampala", city="Kampala", country="Uganda", sample_id=a["sample_id"],
                 containment=a["containment"].map(containment), ph=a["pH"], ec_mScm=a["EC"] / 1000,
                 ts_gL=a["ts"], ts_gL_source="measured g/L (mean of depths)", vs_gL=a["vs"],
                 cod_gL=a["COD"] / 1000, nh4_mgL=a["ammonia"])


def sam_multicity():
    s = pd.read_csv(RAW / "sam2023_feces_to_fs/fecal_sludge_samples.csv", encoding="latin-1")
    return frame(dataset="sam2023_multicity", city=s["city"], country=s["country"], sample_id=s.index.astype(str),
                 ph=num(s["ph"]), ec_mScm=num(s["conductivity"]), ts_gL=num(s["TS (g/L)"]),
                 ts_gL_source="measured g/L", vs_gL=num(s["VS  (g/L)"]), cst_s=num(s["CST"]))


def sircilla_structure():
    f = RAW / "sircilla_india/sircilla_data.xlsx"
    if not f.exists():
        return
    x = pd.ExcelFile(f)
    with open(OUT / "sircilla_columns.txt", "w", encoding="utf-8") as fh:
        for sh in x.sheet_names:
            d = x.parse(sh, nrows=5)
            fh.write(f"## {sh}  ({len(x.parse(sh))} rows)\n{list(d.columns)}\n\n")


def duplicates(df, tol=0.005):
    """Flag cross-dataset pairs in the same city whose pH (within 0.02), TS and EC (within 0.5%) all match.
    Re-used samples should match almost exactly; review every flagged pair by hand."""
    rows = []
    d = df.dropna(subset=["ph", "ts_gL"])
    for city, g in d.groupby("city"):
        sets = g["dataset"].unique()
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                A, B = g[g.dataset == a], g[g.dataset == b]
                for _, r in A.iterrows():
                    close = B[(abs(B.ph - r.ph) <= 0.02) &
                              (abs(B.ts_gL - r.ts_gL) <= tol * max(abs(r.ts_gL), 0.1))]
                    if pd.notna(r.ec_mScm):
                        close = close[close.ec_mScm.isna() | (abs(close.ec_mScm - r.ec_mScm) <= 0.01 + tol * abs(r.ec_mScm))]
                    for _, s in close.iterrows():
                        rows.append({"city": city, "dataset_a": a, "id_a": r.sample_id, "dataset_b": b,
                                     "id_b": s.sample_id, "ph_a": r.ph, "ph_b": s.ph, "ts_a": r.ts_gL, "ts_b": s.ts_gL,
                                     "ec_a": r.ec_mScm, "ec_b": s.ec_mScm})
    return pd.DataFrame(rows)


def main():
    loaders = [ward2021, lusaka_microbial, volaser, englund_kampala, englund_hanoi, ward2022, fecalcanuga, sam_multicity]
    parts = []
    for fn in loaders:
        try:
            d = fn()
            if d is None:
                print(f"[skip] {fn.__name__}: file not downloaded")
                continue
            parts.append(d)
            print(f"[ok]   {fn.__name__}: {len(d)} rows")
        except Exception as e:
            print(f"[!!]   {fn.__name__}: {type(e).__name__}: {e}")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT / "standardized_samples.csv", index=False)

    targets = ["ts_gL", "vs_gL", "cod_gL", "nh4_mgL", "cst_s", "ec_mScm", "ph"]
    counts = df.groupby(["dataset", "country", "city"])[targets].count()
    counts = counts[counts.sum(axis=1) > 0].reset_index()
    counts.to_csv(OUT / "inventory_counts.csv", index=False)

    ssa = df[df.country.isin(SSA)]
    print("\n=== Samples per city in Sub-Saharan Africa (all datasets, before removing duplicates) ===")
    print(ssa.groupby("city")[targets].count().to_string())
    print("\n=== Outside Africa (external test candidates) ===")
    print(df[~df.country.isin(SSA)].groupby("country")[targets].count().to_string())
    print("\n=== Median values by city (g/L, mg/L) - sanity check on units ===")
    print(df.groupby("city")[["ts_gL", "vs_gL", "cod_gL", "nh4_mgL"]].median().round(1).to_string())

    dup = duplicates(df)
    dup.to_csv(OUT / "possible_duplicates.csv", index=False)
    print(f"\n=== Possible duplicate samples across datasets: {len(dup)} pairs ===")
    if len(dup):
        print(dup.groupby(["city", "dataset_a", "dataset_b"]).size().to_string())
    sircilla_structure()
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
