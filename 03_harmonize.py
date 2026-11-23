"""
03_harmonize.py
Build the modelling table: one row per containment sample, common targets and predictors,
duplicates removed. Run after 01_download_data.py:   python 03_harmonize.py

Outputs
  data/processed/fs_model_table.csv     modelling table (all cities, incl. external test sets)
  outputs/predictor_coverage.csv        % of samples with each predictor, by dataset
  outputs/target_coverage.csv           samples with each target, by city
  outputs/data_dictionary.csv           definition, unit and coding of every column

Units: TS, VS, COD g/L; NH4-N mg/L; EC mS/cm; CST s; volume m3; ages and times in years.
Binary predictors: 1 = yes, 0 = no, blank = unknown / not recorded.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

TARGETS = ["ts_gL", "vs_gL", "cod_gL", "nh4_mgL", "cst_s"]
PREDICTORS = ["containment", "watertight", "volume_m3", "users", "household", "flush_toilet",
              "greywater", "water_connection", "solid_waste", "additives", "age_yr",
              "never_emptied", "years_since_emptied", "sludge_age_yr", "ph", "ec_mScm"]
META = ["dataset", "campaign_year", "city", "country", "region", "split_role", "sample_id", "outlier_flag"]
CITY_COV = ["city_tavg_c", "city_precip_mm", "city_rel_hum_pct", "city_elevation_m", "city_pop_density_km2",
            "city_groundwater", "city_climate_zone"]
SD = RAW / "sludgedata"
SSA = {"Zambia", "Uganda", "Ghana", "Sierra Leone", "Kenya", "Tanzania", "Burkina Faso", "South Africa"}

DICTIONARY = {
    "dataset": "Source dataset folder (see README for DOI)",
    "campaign_year": "Year(s) of the sampling campaign",
    "city": "City of sampling", "country": "Country",
    "region": "SSA = Sub-Saharan Africa; external = outside SSA",
    "split_role": "train_test = used in leave-one-city-out; external_test = only used for out-of-region testing",
    "sample_id": "Sample identifier as in the source file",
    "outlier_flag": "1 if the source authors flagged the sample as an outlier (Englund Kampala 'out' = YES)",
    "ts_gL": "Total solids, g/L", "vs_gL": "Volatile solids, g/L", "cod_gL": "Chemical oxygen demand, g/L",
    "nh4_mgL": "Ammonium nitrogen, mg/L", "cst_s": "Capillary suction time, s",
    "ts_gL_source": "'measured g/L' or 'from %wt' (= %wt x 10 x density; density 1 g/mL if not measured)",
    "containment": "pit latrine | septic tank | other (cesspit, holding tank)",
    "watertight": "1 = fully lined / sealed / watertight; 0 = partially lined, unlined or not sealed",
    "volume_m3": "Containment volume, m3",
    "users": "Number of people using the toilet(s) feeding the containment",
    "household": "1 = household (single or multiple); 0 = public, school, commercial, institutional",
    "flush_toilet": "1 = pour-flush or cistern-flush toilet; 0 = dry or urine-diverting only",
    "greywater": "1 = bathing, laundry or kitchen water also enters; 0 = toilet only",
    "water_connection": "1 = piped water connection on premises",
    "solid_waste": "1 = solid waste (incl. hygiene products) present in containment",
    "additives": "1 = anything added to the containment (chemicals, bio-additives, ash, soap ...)",
    "age_yr": "Containment age, years (categories converted to midpoints: <1 = 0.5, 1-5 = 3, 6-10 = 8, >10 = 15)",
    "never_emptied": "1 = containment never emptied since construction",
    "years_since_emptied": "Years since last emptying (Englund Kampala: reported emptying interval)",
    "sludge_age_yr": "Years the sampled sludge has been accumulating = years_since_emptied, or age_yr if never emptied",
    "ph": "pH of sample", "ec_mScm": "Electrical conductivity, mS/cm",
    "city_tavg_c": "City mean annual air temperature, deg C (Global Faecal Sludge Database)",
    "city_precip_mm": "City mean annual precipitation, mm/yr (Global Faecal Sludge Database)",
    "city_rel_hum_pct": "City mean relative humidity, % (Global Faecal Sludge Database)",
    "city_elevation_m": "City elevation, m (Global Faecal Sludge Database)",
    "city_pop_density_km2": "City population density, people/km2 (Global Faecal Sludge Database)",
    "city_groundwater": "City groundwater level: 0 = low, 1 = medium, 2 = high (Global Faecal Sludge Database)",
    "city_climate_zone": "Climate zone (Global Faecal Sludge Database)",
}


# ---------------------------------------------------------------- helpers
def num(s):
    return pd.to_numeric(s, errors="coerce")


def yesno(s, yes=("yes", "true", "1", "y"), no=("no", "false", "2", "n", "0")):
    s = s.astype(str).str.strip().str.lower()
    return np.where(s.isin(yes), 1.0, np.where(s.isin(no), 0.0, np.nan))


def cont(x):
    x = str(x).lower()
    if "septic" in x:
        return "septic tank"
    if "pit" in x or "latrine" in x:
        return "pit latrine"
    if any(k in x for k in ("cess", "holding", "leach")):
        return "other"
    return np.nan


def age_category(x):
    x = str(x).lower()
    if "less than 1" in x:
        return 0.5
    if "1 - 5" in x or "1-5" in x:
        return 3.0
    if "6 - 10" in x and "more than 10" in x:
        return 10.0
    if "6 - 10" in x or "6-10" in x:
        return 8.0
    if "more than 10" in x:
        return 15.0
    return np.nan


def greywater(x):
    x = str(x).lower()
    if x in ("nan", "", "none") or "know" in x:
        return np.nan
    return 1.0 if any(k in x for k in ("bath", "laund", "kitch", "wash")) else (1.0 if x.strip() in ("2", "3", "1+2") else 0.0)


def flush(x):
    x = str(x).lower()
    if "flush" in x or x.strip() == "wet" or x.strip() == "both":
        return 1.0
    if "dry" in x or "urine" in x:
        return 0.0
    return np.nan


def additives_any(x):
    x = str(x).strip().lower()
    if x in ("nan", "", "i don't know", "i don't know", "i don't know "):
        return np.nan
    if x in ("no", "nothing", "nil", "not yet", "none", "false", "0"):
        return 0.0
    return 1.0


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1, "about": None}


def text_to_years(x):
    """'Three months ago' -> 0.25; 'One week ago' -> 0.019; 'Today' -> 0; numbers pass through as years."""
    s = str(x).strip().lower()
    if s in ("nan", "") or "know" in s:
        return np.nan
    if "today" in s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass
    m = re.search(r"(\d+(?:\.\d+)?|" + "|".join(k for k, v in WORDS.items() if v) + r")\s*(day|week|month|year)", s)
    if not m:
        return np.nan
    n = WORDS.get(m.group(1), None) if not m.group(1)[0].isdigit() else float(m.group(1))
    return n * {"day": 1 / 365, "week": 7 / 365, "month": 1 / 12, "year": 1}[m.group(2)]


def assemble(d):
    for c in META + TARGETS + ["ts_gL_source"] + PREDICTORS:
        if c not in d:
            d[c] = np.nan
    # sludge age: time since emptying, or containment age if never emptied
    d["sludge_age_yr"] = np.where(d["never_emptied"] == 1, d["age_yr"], d["years_since_emptied"])
    d["region"] = np.where(d["country"].isin(SSA), "SSA", "external")
    d["split_role"] = np.where(d["region"] == "SSA", "train_test", "external_test")
    return d[META + TARGETS + ["ts_gL_source"] + PREDICTORS]


# ---------------------------------------------------------------- loaders
def ward2021():
    w = pd.read_csv(RAW / "ward2021_lusaka/mastersheet_final_ward2020.csv", encoding="latin-1", low_memory=False)
    ts = num(w["ts_gL"])
    ts_pct = num(w["ts_%"]) * 10 * num(w["density"]).fillna(1.0)
    oss = w["oss_type"].astype(str).str.lower()
    le = w["last_emptied"].astype(str).str.strip()
    d = pd.DataFrame({
        "dataset": "ward2021_lusaka", "campaign_year": "2019", "city": "Lusaka", "country": "Zambia",
        "sample_id": w["sample_id"],
        "ts_gL": ts.fillna(ts_pct), "ts_gL_source": np.where(ts.notna(), "measured g/L", np.where(ts_pct.notna(), "from %wt", None)),
        "vs_gL": ts.fillna(ts_pct) * num(w["vs_%ts"]) / 100, "cod_gL": num(w["cod"]),
        "nh4_mgL": num(w["nh4_mg"]), "cst_s": num(w["cst"]),
        "containment": w["oss_type"].map(cont),
        "watertight": np.where(oss.str.contains("completely sealed"), 1.0,
                               np.where(oss.str.contains("not sealed|unlined"), 0.0, np.nan)),
        "volume_m3": num(w["oss_volume"]), "users": num(w["nr_users"]),
        "household": yesno(w["origin_simplified"], yes=("household",), no=("non-household",)),
        "flush_toilet": w["toilet_type_simplified"].map(flush), "greywater": w["ww_type"].map(greywater),
        "water_connection": yesno(w["water_connection"]), "solid_waste": yesno(w["solid_waste"]),
        "additives": w["additives"].map(additives_any), "age_yr": num(w["oss_age"]),
        "never_emptied": np.where(le.str.contains("Never", case=False), 1.0, np.where(num(le).notna(), 0.0, np.nan)),
        "years_since_emptied": num(le), "ph": num(w["ph"]), "ec_mScm": num(w["ec"]),
    })
    return assemble(d)


def volaser():
    v = pd.read_csv(RAW / "volaser_7countries/complete_dataset_volaser.csv", encoding="latin-1", low_memory=False)
    v = v[v["country"].isin(SSA | {"India", "Lebanon"})].copy()
    lining = v["lining"].astype(str).str.lower()
    never = (v["last_emptied"].astype(str).str.contains("never", case=False) |
             v["fully_emptied"].astype(str).str.contains("never|nev", case=False))
    yse = num(v["last_emptied_y"])
    users = num(v["nr_users"]).fillna(num(v["pt_users"]))
    d = pd.DataFrame({
        "dataset": "volaser_7countries", "campaign_year": "2021", "city": v["city"].str.strip(), "country": v["country"],
        "sample_id": v["sample_id"],
        "ts_gL": num(v["ts_gl"]), "ts_gL_source": "measured g/L", "vs_gL": num(v["vs_gl"]), "cod_gL": num(v["cod"]) / 1000,
        "containment": v["cont_type"].map(cont),
        "watertight": np.where(lining.str.contains("fully"), 1.0, np.where(lining.str.contains("partial|unlined"), 0.0, np.nan)),
        "volume_m3": num(v["oss_volume"]).fillna(num(v["cont_vol_1"])), "users": users,
        "household": np.where(v["origin"].isna(), np.nan, (v["origin"] == "Household").astype(float)),
        "flush_toilet": v["toilet"].map(flush), "greywater": v["ww"].map(greywater),
        "water_connection": yesno(v["water_connect"]), "solid_waste": yesno(v["sw_yn"]),
        "additives": yesno(v["additives_yn"]), "age_yr": num(v["cont_age_mid"]),
        "never_emptied": np.where(never, 1.0, np.where(yse.notna(), 0.0, np.nan)),
        "years_since_emptied": yse, "ph": num(v["ph"]), "ec_mScm": num(v["ec"]),
    })
    return assemble(d)


def englund_kampala():
    k = pd.read_excel(RAW / "englund2020_kampala_hanoi/kampala_data.xlsx").iloc[1:]  # row 1 = units
    d = pd.DataFrame({
        "dataset": "englund2020_kampala", "campaign_year": "2016-2017", "city": "Kampala", "country": "Uganda",
        "sample_id": k["code"], "outlier_flag": (k["out"].astype(str).str.upper() == "YES").astype(float),
        "ts_gL": num(k["TS g"]), "ts_gL_source": "measured g/L", "vs_gL": num(k["VS g"]), "cod_gL": num(k["COD g"]),
        "nh4_mgL": num(k["NH4N mg"]),
        "containment": k["CoTyp"].map(cont), "watertight": yesno(k["WaTig"]), "volume_m3": num(k["CoVol"]),
        "users": num(k["NoUse"]),
        "household": k["OrCat"].astype(str).str.contains("Household").astype(float),
        "greywater": k["WaCha"].map(greywater), "water_connection": yesno(k["WaCon"]),
        "solid_waste": np.where(k["SoWas"].astype(str).str.strip().str.lower() == "no", 0.0, 1.0),
        "additives": k["BiAdd"].map(additives_any),
        "age_yr": np.nan,  # CoAge is a placeholder (25 yr for 179/180 samples)
        "never_emptied": 0.0,  # samples were collected from emptying trucks
        "years_since_emptied": num(k["SyWee"]) / 52,  # reported emptying interval (weeks)
        "ph": num(k["pH"]),  # no EC measured (column 'C' is colour)
    })
    return assemble(d)


def ward2022():
    p = pd.read_csv(RAW / "ward2022_kampala_naivasha/data_field_samples.csv", encoding="latin-1")
    lining = p["Containment_lining"].astype(str).str.lower()
    le = p["When was the system last emptied?"]
    never = le.astype(str).str.contains("never", case=False) | p["Was it fully emptied at that time? "].astype(str).str.contains("never", case=False)
    yse = le.map(text_to_years)
    d = pd.DataFrame({
        "dataset": "ward2022_kampala_naivasha", "campaign_year": "2021", "city": p["City"],
        "country": p["City"].map({"Kampala": "Uganda", "Naivasha": "Kenya"}), "sample_id": p.iloc[:, 0],
        "ts_gL": num(p["TS_gL"]), "ts_gL_source": "measured g/L", "vs_gL": num(p["VS_gL"]),
        "cod_gL": num(p["COD_gL"]) if "COD_gL" in p else np.nan,
        "nh4_mgL": num(p["NH4N_mgL"]), "cst_s": num(p["CST_s"]),
        "containment": p["Containment_type"].map(cont),
        "watertight": np.where(lining.str.contains("fully"), 1.0, np.where(lining.str.contains("partial|unlined"), 0.0, np.nan)),
        "users": num(p["Estimated users per day"]),
        "household": p["Type_of_establishment"].astype(str).str.startswith("Household").astype(float),
        "flush_toilet": p["Toilet_type_simp"].map(flush),
        "greywater": p["Type(s) of wastewater entering the containment"].map(greywater),
        "water_connection": yesno(p["Water tap inside building"]) if "Water tap inside building" in p else np.nan,
        "solid_waste": yesno(p["Is there solid waste in the containment?"]),
        "additives": p["Additives"].map(additives_any),
        "age_yr": p["How old is the containment?"].map(age_category),
        "never_emptied": np.where(never, 1.0, np.where(yse.notna(), 0.0, np.nan)),
        "years_since_emptied": yse, "ph": num(p["pH"]), "ec_mScm": num(p["EC_mScm"]),
    })
    return assemble(d)


def fecalcanuga():
    p = pd.read_csv(RAW / "fecalcanuga/phys_chem_parameter.csv")
    p = p[p["date"].astype(str).str.startswith("2024")]  # 2024 = Kampala; 2023 = Canada
    a = p.groupby("sample_id")[["pH", "EC", "ts", "vs", "COD", "ammonia"]].mean().reset_index()  # mean of 3 depths
    s = pd.read_csv(RAW / "fecalcanuga/household_survey.csv").drop(columns=["containment_volume"], errors="ignore")
    c = pd.read_csv(RAW / "fecalcanuga/containment.csv")[["sample_id", "containment_volume"]]
    a = a.merge(s, on="sample_id", how="left").merge(c, on="sample_id", how="left")
    lining = a["lining"].astype(str).str.lower()
    d = pd.DataFrame({
        "dataset": "fecalcanuga_kampala", "campaign_year": "2024", "city": "Kampala", "country": "Uganda",
        "sample_id": a["sample_id"],
        "ts_gL": a["ts"], "ts_gL_source": "measured g/L (mean of 3 depths)", "vs_gL": a["vs"],
        "cod_gL": a["COD"] / 1000, "nh4_mgL": a["ammonia"],
        "containment": a["containment"].map(cont),
        "watertight": np.where(lining.str.contains("fully"), 1.0, np.where(lining.str.contains("partial|unlined"), 0.0, np.nan)),
        "volume_m3": num(a["containment_volume"]), "users": num(a["users"]),
        "household": (a["establishment_type"] == "Household").astype(float),
        "flush_toilet": a["toilet_type"].map(flush), "greywater": a["wastewater_type"].map(greywater),
        "water_connection": yesno(a["water_connection"]), "solid_waste": yesno(a["solid_waste"]),
        "additives": yesno(a["additives"]), "age_yr": a["containment_age"].map(age_category),
        "never_emptied": np.where(num(a["last_emptied"]).notna(), 0.0, np.nan),
        "years_since_emptied": num(a["last_emptied"]), "ph": a["pH"], "ec_mScm": a["EC"] / 1000,
    })
    return assemble(d)


def sircilla():
    f = RAW / "sircilla_india/sircilla_data.xlsx"
    if not f.exists():
        return None
    s = pd.read_excel(f)
    yse = num(s["yrs_last_des"])
    never = s["res_last_des"].astype(str).str.strip().str.lower().eq("not done")
    d = pd.DataFrame({
        "dataset": "sircilla_india", "campaign_year": "2018-2019", "city": "Sircilla", "country": "India",
        "sample_id": s["id"],
        "ts_gL": num(s["ts"]) / 1000, "ts_gL_source": "measured mg/L / 1000",  # source reports mg/L
        "vs_gL": num(s["vs"]) / 1000, "cod_gL": num(s["cod"]) / 1000,
        "containment": s["cont_type"].map(cont), "watertight": yesno(s["water_tight"]),
        "volume_m3": num(s["vol_cont"]), "users": num(s["num_users"]),
        "household": (s["cont_source"] == "Household").astype(float),
        "flush_toilet": s["toilet_type"].map(flush), "solid_waste": yesno(s["solid_waste"]),
        "age_yr": num(s["age_cont"]),
        "never_emptied": np.where(never, 1.0, np.where(yse.notna(), 0.0, np.nan)),
        "years_since_emptied": yse,
    })
    return assemble(d)


def englund_hanoi():
    x = pd.ExcelFile(RAW / "englund2020_kampala_hanoi/hanoi_data.xlsx")
    parts = []
    # H = households, P = public toilets, O/R/B = offices, restaurants, buildings.
    # Sheet W (Kim Lien treatment plant) holds treatment-plant samples, not containments: excluded.
    for sh in [s for s in x.sheet_names if not s.strip().endswith(" W") and s.strip() != "Characterization W"]:
        d0 = x.parse(sh, header=2)  # row 3 holds the short labels (Code, N inhabitant, ST age, ...)
        d0.columns = [str(c).strip() for c in d0.columns]
        h = x.parse(sh, header=1)
        h.columns = [str(c).strip() for c in h.columns]
        h = h.iloc[1:].reset_index(drop=True)
        d0 = d0.reset_index(drop=True)
        code = h.iloc[:, 0]
        keep = code.notna()
        g = lambda df, c: num(df[c]) if c in df else pd.Series(np.nan, index=df.index)
        d = pd.DataFrame({
            "dataset": "englund2020_hanoi", "campaign_year": "2016-2017", "city": "Hanoi", "country": "Vietnam",
            "sample_id": sh.strip() + ":" + code.astype(str),
            "ts_gL": g(h, "TS"), "ts_gL_source": "measured g/L", "vs_gL": g(h, "VS"),
            "cod_gL": g(h, "COD") / 1000, "nh4_mgL": g(h, "N-NH4"),
            "containment": "septic tank",
            "household": 1.0 if sh.strip().endswith(" H") else 0.0,
            "watertight": yesno(d0["Sealed?"]) if "Sealed?" in d0 else np.nan,
            "volume_m3": g(d0, "V ST (m3)"), "users": g(d0, "N inhabitant"),
            "greywater": d0["WW origin"].astype(str).map(greywater) if "WW origin" in d0 else np.nan,
            "age_yr": g(d0, "ST age"), "never_emptied": np.nan,
            "years_since_emptied": g(d0, "Empty"),  # 'Age of the sludge [years]'
            "ph": g(h, "pH"), "ec_mScm": g(h, "EC") / 1000,  # source EC in uS/cm
        })
        parts.append(d[keep.values])
    return assemble(pd.concat(parts, ignore_index=True))



# ---------------------------------------------------------------- Global Faecal Sludge Database (sludgedata.online)
# Labs whose source study is NOT already loaded above (others duplicate Ward, Andriessen, Strande/Englund, Prasad).
SD_NEW_LABS = {20: "Bassan_2013 Ouagadougou", 22: "Esanju Dar es Salaam", 24: "Marwa Dar es Salaam",
               30: "AIT Thailand", 31: "CDD India", 32: "Velkushanova_2021 India", 34: "ENPHO Nepal",
               35: "ITB Indonesia", 36: "UTEC Uruguay", 37: "WASH R&D South Africa", 38: "BITS Goa India"}
SD_PARAMS = {50: "ts_gL", 1: "ts_pct", 49: "vs_gL", 3: "vs_pct", 9: "cod_mgL", 13: "nh4", 14: "nh3",
             20: "ph", 21: "ec_mScm"}


def band_mid(x):
    """'10 to 20' -> 15; 'never emptied' -> nan."""
    m = re.match(r"\s*([\d.]+)\s*to\s*([\d.]+)", str(x))
    return (float(m.group(1)) + float(m.group(2))) / 2 if m else np.nan


def sd_containment(names):
    n = " ".join(names).lower()
    if "septic" in n and ("pit" in n and "soak" not in n):
        return "other"  # mixed pit + septic
    if "septic" in n:
        return "septic tank"
    if any(k in n for k in ("pit", "vip", "vault", "uddt", "dry toilet", "fossa")) and "soak" not in n:
        return "pit latrine"
    if n.strip():
        return "other"
    return np.nan


def sd_flush(point):
    p = str(point).lower()
    if any(k in p for k in ("dry", "uddt", "urine diverting dry", "composting", "vip")):
        return 0.0
    if "flush" in p:
        return 1.0
    return np.nan


def sludgedata_new():
    if not (SD / "samples.csv").exists():
        return None
    smp = pd.read_csv(SD / "samples.csv")
    res = pd.read_csv(SD / "results.csv")
    labs = res[["sample_id", "lab_id"]].drop_duplicates()
    res = res[res["parameter_id"].isin(SD_PARAMS)]
    w = res.pivot_table(index="sample_id", columns="parameter_id", values="result_value", aggfunc="first").rename(columns=SD_PARAMS)
    smp = smp.merge(labs, on="sample_id").merge(w, left_on="sample_id", right_index=True, how="left")
    smp = smp[smp["lab_id"].isin(SD_NEW_LABS)]
    lab_country = pd.read_csv(SD / "labs.csv").set_index("lab_id")["country"]
    cont = (pd.read_csv(SD / "sample_containments.csv").merge(pd.read_csv(SD / "containment_types.csv"), on="containment_id")
            .groupby("sample_id")["name"].apply(list))
    point = pd.read_csv(SD / "sampling_locations.csv").set_index("location_id")["name"]
    g = lambda c: num(smp[c]) if c in smp else pd.Series(np.nan, index=smp.index)
    ts = g("ts_gL").fillna(g("ts_pct") * 10)
    vs_pct = g("vs_pct").where(g("vs_pct").between(20, 100))          # QC: implausible VS%TS discarded
    vs = g("vs_gL").where(g("vs_gL") <= ts).fillna(ts * vs_pct / 100)
    vs = vs.where((vs / ts).between(0.2, 1.0))
    le = smp["last_emptied"].astype(str)
    lining = smp["lining"].astype(str).str.lower()
    d = pd.DataFrame({
        "dataset": "sludgedata_" + smp["lab_id"].map(SD_NEW_LABS).str.split().str[0].str.lower(),
        "campaign_year": smp["sample_date"].astype(str).str[:4].replace("nan", np.nan),
        "city": smp["city_sample"].str.split(",").str[0].str.strip(),
        "country": smp["lab_id"].map(lab_country), "sample_id": "SD" + smp["sample_id"].astype(str),
        "ts_gL": ts, "ts_gL_source": "Global FS Database (g/L or %wt x 10)", "vs_gL": vs,
        "cod_gL": g("cod_mgL") / 1000, "nh4_mgL": g("nh4").fillna(g("nh3")),
        "containment": smp["sample_id"].map(cont).apply(lambda v: sd_containment(v) if isinstance(v, list) else np.nan),
        "watertight": np.where(lining.str.contains("fully"), 1.0,
                               np.where(lining.str.contains("partial|unlined"), 0.0, np.nan)),
        "volume_m3": g("containment_volume"), "users": g("number_users"),
        "flush_toilet": smp["sampling_location_id"].map(point).map(sd_flush),
        "age_yr": smp["containment_age"].map(band_mid),
        "never_emptied": np.where(le.str.contains("never"), 1.0, np.where(le.map(band_mid).notna(), 0.0, np.nan)),
        "years_since_emptied": le.map(band_mid),
        "ph": g("ph"), "ec_mScm": g("ec_mScm"),
    })
    return assemble(d.reset_index(drop=True))


def city_covariates():
    """City-level climate and context variables from the Global Faecal Sludge Database (one value per city)."""
    if not (SD / "samples.csv").exists():
        return None
    smp = pd.read_csv(SD / "samples.csv")
    smp["city"] = smp["city_sample"].str.split(",").str[0].str.strip()
    gw = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
    c = smp.groupby("city").agg(city_tavg_c=("tavg_c", "median"), city_precip_mm=("pavg_mm_per_yr", "median"),
                                city_rel_hum_pct=("rel_hum_avg_pct", "median"), city_elevation_m=("elevation", "median"),
                                city_pop_density_km2=("pop_density_km2", "median"),
                                city_groundwater=("groundwater_level", lambda x: x.map(gw).median()),
                                city_climate_zone=("climate_zone", lambda x: x.mode().iat[0] if x.notna().any() else np.nan))
    return c.reset_index()


# ---------------------------------------------------------------- main
def main():
    parts = []
    for fn in (ward2021, volaser, englund_kampala, ward2022, fecalcanuga, sircilla, englund_hanoi, sludgedata_new):
        try:
            d = fn()
            if d is None:
                print(f"[skip] {fn.__name__}: not downloaded")
                continue
            parts.append(d)
            print(f"[ok]   {fn.__name__}: {len(d)} rows")
        except Exception as e:
            print(f"[!!]   {fn.__name__}: {type(e).__name__}: {e}")
    df = pd.concat(parts, ignore_index=True)
    df["outlier_flag"] = df["outlier_flag"].fillna(0)
    # Not included: lusaka_microbial (133/135 samples are the Ward 2021 Lusaka samples),
    # sam2023_multicity and sam2022_eps (1 sample per city / lab reactors).
    df = df[df[["ts_gL", "vs_gL", "cod_gL", "nh4_mgL", "cst_s"]].notna().any(axis=1)]
    cc = city_covariates()
    if cc is not None:
        df = df.merge(cc, on="city", how="left")
        miss = sorted(df.loc[df["city_tavg_c"].isna() & (df["region"] == "SSA"), "city"].unique())
        print(f"[ok]   city covariates joined; SSA cities without covariates: {miss}")
    else:
        for c in CITY_COV:
            df[c] = np.nan
    df.to_csv(PROC / "fs_model_table.csv", index=False)

    cov = df.groupby("dataset")[PREDICTORS + CITY_COV].apply(lambda g: g.notna().mean().mul(100).round(0))
    cov.insert(0, "n", df.groupby("dataset").size())
    cov.to_csv(OUT / "predictor_coverage.csv")
    tc = df.groupby(["region", "country", "city"])[TARGETS].count()
    tc.to_csv(OUT / "target_coverage.csv")
    pd.DataFrame({"column": list(DICTIONARY), "definition": list(DICTIONARY.values())}).to_csv(OUT / "data_dictionary.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"\nModel table: {len(df)} samples -> {PROC / 'fs_model_table.csv'}")
    print("\n=== Samples with each target, by city ===")
    print(tc.to_string())
    print("\n=== % of samples with each predictor, by dataset ===")
    print(cov.to_string())


if __name__ == "__main__":
    main()
