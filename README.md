# fs-transferability

Code and aggregated results for:

> Lamore, Y.  Ayana, J.W.,Zifu Li *Cross-city transferability of machine-learning models1 for fecal sludge characterization: Leave-one-city-out
validation and local calibration in Sub-Saharan Africa* Manuscript under review, Journal of Environmental Management.

The scripts download seven open fecal sludge datasets, harmonise them into one modelling table, and evaluate how well models trained in some cities predict total solids (TS), volatile solids (VS), chemical oxygen demand (COD) and ammonium (NH4+-N) in a city they have never seen (leave-one-city-out, LOCO). They also run the within-city, external, mixed-model, local-calibration and engineering-sensitivity analyses reported in the paper and its Supplementary Material.

## Repository contents

| Path | Content |
|---|---|
| `01_download_data.py` | Downloads the open datasets from their DOIs and checks each file against `data/MANIFEST.csv` (SHA-256) |
| `02_inventory.py` | Standardises units, counts samples and flags cross-dataset duplicates |
| `03_harmonize.py` | Builds `data/processed/fs_model_table.csv` (harmonised modelling table) and `outputs/data_dictionary.csv` |
| `04_model.py` | Main analyses: LOCO, within-city cross-validation, external tests, conformal intervals, SHAP, local calibration |
| `05_revision.py` | Additional analyses R1 to R13 (sample flow, study composition, variance components, source sensitivity, mixed models, hyperparameter sensitivity, bootstrap intervals, stratified and matched local calibration, SHAP consistency, engineering sensitivity, transfer scenarios, city-level metrics) |
| `figures_tables/` | Scripts for every figure and LaTeX table in the paper and Supplementary Material |
| `data/MANIFEST.csv` | Source DOI, URL, size and SHA-256 of every downloaded file |
| `data/data_dictionary.csv` | Definition, unit and coding of every column of the modelling table |
| `results/model/` | Aggregated outputs of `04_model.py` used in the paper |
| `results/revision/` | Aggregated outputs of `05_revision.py` (R1 to R13) |
| `run_all.bat`, `run_model.bat`, `run_revision.bat` | One-click Windows runners |
| `record_versions.bat` | Writes `python --version` and `pip freeze` to `outputs/package_versions.txt` |

Sample-level measurements are **not** redistributed here, because three source packages carry no explicit licence. Instead, the scripts rebuild the modelling table from the original sources. For the same reason the per-sample prediction file (`loco_predictions.csv`) is produced locally by `04_model.py` but is not archived.

## How to reproduce

Requirements: Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python 01_download_data.py      # downloads Eawag Open Data and GitHub sources
python 02_inventory.py
python 03_harmonize.py          # -> data/processed/fs_model_table.csv
python 04_model.py              # -> outputs/model/   (5 to 15 min; --quick skips GP and balanced XGBoost)
python 05_revision.py           # -> outputs/revision/ (30 to 60 min)
cd figures_tables
python make_figures.py && python figures_pro.py && python figures_design.py && python figures_revision.py && python make_tables.py
```

On Windows, double-click `run_all.bat`, then `run_model.bat`, then `run_revision.bat`.

**One manual step.** The Global Database for Faecal Sludge Analysis (GFSD) is not downloadable by script. Export the database from <https://sludgedata.online> and place its CSV tables (`samples.csv`, `results.csv`, `labs.csv`, `sample_containments.csv`, `containment_types.csv`, `sampling_locations.csv`, ...) in `data/raw/sludgedata/`. The paper used the release of 21 November 2026. Without these files `03_harmonize.py` runs but omits the GFSD samples (Dar es Salaam, Ouagadougou, Pietermaritzburg).

The figure and table scripts read fresh outputs from `outputs/` when they exist and otherwise fall back to the archived files in `results/`. Figures that need sample-level data (map, observed vs predicted, city distributions) require the full pipeline to be run first.

Random seeds are fixed (42 to 46 for repeated cross-validation; 42 for bootstraps and calibration draws). See Supplementary Material S8 for the software environment used for the reported results.

## Data sources

| Source | DOI / location | Licence |
|---|---|---|
| Ward et al. (2021), Lusaka | doi:10.25678/00037X | MIT |
| Andriessen et al. (2023), Volaser seven-country set | doi:10.25678/0007KN | none stated (Eawag terms) |
| Englund et al. (2020), Kampala and Hanoi | doi:10.25678/0000TT | GPL v3 |
| Ward et al. (2023), Kampala and Naivasha | doi:10.25678/000702 | none stated (Eawag terms) |
| Prasad et al. (2021), Sircilla (India) | doi:10.25678/0002VH | none stated (Eawag terms) |
| Shaw et al. (2026), fecalcanuga, Kampala | doi:10.5281/zenodo.20795916; github.com/openwashdata/fecalcanuga | CC BY 4.0 |
| Global Database for Faecal Sludge Analysis | https://sludgedata.online | CC BY 4.0 |

Please cite the original data publications when using the rebuilt table.

## Licence

Code: MIT (see `LICENSE`). Aggregated results in `results/`: CC BY 4.0. Source data remain under their original terms.

## Citation

See `CITATION.cff`. A DOI will be added on publication.

## Contact

Jifar Wakuma Ayana, University of Science and Technology Beijing (d202561032@xs.ustb.edu.cn)
