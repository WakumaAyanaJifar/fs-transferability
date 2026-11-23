"""
01_download_data.py
Download every public fecal sludge dataset used in the study into data/raw/<dataset>/,
check each file's size, and write data/raw/MANIFEST.csv (source, DOI, size, SHA-256).

Uses only the Python standard library. Run from this folder:
    python 01_download_data.py
Re-running skips files that are already present with the right size.
"""
import csv
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
E = "https://opendata.eawag.ch/dataset"
GH = "https://raw.githubusercontent.com/openwashdata"

# (dataset folder, DOI / source, file name to save, URL, expected bytes or None)
FILES = [
    # --- Ward et al. 2021, Water Research: Lusaka, 465 samples (core) ---
    ("ward2021_lusaka", "10.25678/00037X", "mastersheet_final_ward2020.csv", f"{E}/1b00ef13-5661-4ba4-8d03-86ff832a61c2/resource/10597609-d2cb-4b22-b796-877ef6d107a0/download/mastersheet_final_ward2020.csv", 295669),
    ("ward2021_lusaka", "10.25678/00037X", "data_label_explanation_mastersheet.md", f"{E}/1b00ef13-5661-4ba4-8d03-86ff832a61c2/resource/3affd138-18b7-453e-9096-2727d3d39536/download/data-label-explanation_mastersheet.md", 5581),
    ("ward2021_lusaka", "10.25678/00037X", "readme.md", f"{E}/1b00ef13-5661-4ba4-8d03-86ff832a61c2/resource/cc1de8d1-591a-4d55-aeb8-6602017ea7bd/download/readme.md", 2898),
    ("ward2021_lusaka", "10.25678/00037X", "license.txt", f"{E}/1b00ef13-5661-4ba4-8d03-86ff832a61c2/resource/36181381-7128-4b34-9f52-e5a6ea674f14/download/license.txt", 1072),
    ("ward2021_lusaka", "10.25678/00037X", "all_models_cross_validated_performance.ipynb", f"{E}/1b00ef13-5661-4ba4-8d03-86ff832a61c2/resource/7caf728a-7126-42c3-987f-df7392cde651/download/all-models-cross-validated-performance.ipynb", 30242),
    # --- Lusaka city-wide microbial study, 135 containments (core; check overlap with Ward 2021) ---
    ("lusaka_microbial", "10.25678/0008ZW", "metadata.csv", f"{E}/785e8852-a4b9-43d0-9fe1-668b6eb236ff/resource/c86e9147-0f07-4433-ba77-470298708241/download/metadata.csv", 55858),
    ("lusaka_microbial", "10.25678/0008ZW", "data_label_explanation_ss.txt", f"{E}/785e8852-a4b9-43d0-9fe1-668b6eb236ff/resource/ce7d035d-d7ca-41bd-b88e-38798ef603f2/download/data_label_explanation_ss.txt", 3386),
    ("lusaka_microbial", "10.25678/0008ZW", "readme.txt", f"{E}/785e8852-a4b9-43d0-9fe1-668b6eb236ff/resource/00b280bf-a542-4d5b-9f76-a77b50315f32/download/readme.txt", 3495),
    # --- Volaser, Andriessen et al. 2023: 7 countries (core) ---
    ("volaser_7countries", "10.25678/0007KN", "complete_dataset_volaser.csv", f"{E}/1d1a38f1-3d2c-46d7-8607-fd0cf359f63d/resource/d485b39a-5d18-4a21-bab3-4586925d0f0c/download/complete-dataset-volaser_for-submission.csv", 127355),
    ("volaser_7countries", "10.25678/0007KN", "metadata_volaser.xlsx", f"{E}/1d1a38f1-3d2c-46d7-8607-fd0cf359f63d/resource/6ee169fb-77bf-4f92-963c-2c589ce37714/download/metadata-complete-dataset-volaser.xlsx", 15875),
    ("volaser_7countries", "10.25678/0007KN", "readme.txt", f"{E}/1d1a38f1-3d2c-46d7-8607-fd0cf359f63d/resource/c2e6da90-3018-487f-aca3-01baa2399817/download/readme.txt", 768),
    # --- Englund et al. 2020, JEM: Kampala 180 + Hanoi 60 (core + external test) ---
    ("englund2020_kampala_hanoi", "10.25678/0000TT", "kampala_data.xlsx", f"{E}/0d5e14dd-bdb1-4ce2-9da8-e1bec9275b79/resource/b15af45c-6a49-4327-8512-f332cae49749/download/kampala_data.xlsx", 67007),
    ("englund2020_kampala_hanoi", "10.25678/0000TT", "hanoi_data.xlsx", f"{E}/0d5e14dd-bdb1-4ce2-9da8-e1bec9275b79/resource/b0b16f81-1e1e-44b4-9752-d913ed47b43a/download/hanoi_data.xlsx", 60580),
    ("englund2020_kampala_hanoi", "10.25678/0000TT", "metadata_kampala.pdf", f"{E}/0d5e14dd-bdb1-4ce2-9da8-e1bec9275b79/resource/c612ecdc-06a4-4e15-b95b-bfb9c2a7fb1a/download/metadata_kampala.pdf", 111706),
    ("englund2020_kampala_hanoi", "10.25678/0000TT", "metadata_hanoi.pdf", f"{E}/0d5e14dd-bdb1-4ce2-9da8-e1bec9275b79/resource/3e5a1705-582f-4b47-b7ac-ee0b2fbf14a2/download/metadata_hanoi.pdf", 126958),
    ("englund2020_kampala_hanoi", "10.25678/0000TT", "fsludge_scripts.zip", f"{E}/0d5e14dd-bdb1-4ce2-9da8-e1bec9275b79/resource/a941d3f1-90a0-47e9-9f5f-439294de1350/download/fsludge.zip", 372063),
    ("englund2020_kampala_hanoi", "10.25678/0000TT", "readme.txt", f"{E}/0d5e14dd-bdb1-4ce2-9da8-e1bec9275b79/resource/4335b7fd-1bce-4075-9959-1dbd503615ab/download/readme.txt", 1339),
    # --- Ward et al. 2022/2023, JEM: Kampala + Naivasha field samples (core) ---
    ("ward2022_kampala_naivasha", "10.25678/000702", "data_field_samples.csv", f"{E}/c4290e2a-b001-4972-b736-1b65e79a3177/resource/55e0890e-af77-4307-8bc7-adcdcb039dae/download/all_data_field_samples.csv", 9746),
    ("ward2022_kampala_naivasha", "10.25678/000702", "data_label_explanation_field_samples.csv", f"{E}/c4290e2a-b001-4972-b736-1b65e79a3177/resource/3b27767a-f681-4a26-b825-6eba2abcea24/download/data_label_explaination_field-samples.csv", 4007),
    ("ward2022_kampala_naivasha", "10.25678/000702", "questionnaire_kampala_rawdata.csv", f"{E}/c4290e2a-b001-4972-b736-1b65e79a3177/resource/5c074eca-1d27-419b-9678-c63584a4d339/download/questionnaire_kampala_rawdata.csv", 15709),
    ("ward2022_kampala_naivasha", "10.25678/000702", "questionnaire_naivasha_rawdata.csv", f"{E}/c4290e2a-b001-4972-b736-1b65e79a3177/resource/e0d0fd58-ecd1-4076-90d7-98bcfcb111f1/download/questionnaire_naivasha_rawdata.csv", 16456),
    ("ward2022_kampala_naivasha", "10.25678/000702", "readme.txt", f"{E}/c4290e2a-b001-4972-b736-1b65e79a3177/resource/f7f60505-57c4-4b94-bd32-2e573c16e971/download/readme.txt", 2107),
    # --- Sircilla, India (external test outside Africa) ---
    ("sircilla_india", "10.25678/0002VH", "sircilla_data.xlsx", f"{E}/e48f54ab-8783-4baa-8eec-1cbaa4457cdd/resource/d2002a20-0a07-4a0f-9db2-64a5003b90d6/download/sircilla_data.xlsx", 46861),
    ("sircilla_india", "10.25678/0002VH", "metadata_sircilla.pdf", f"{E}/e48f54ab-8783-4baa-8eec-1cbaa4457cdd/resource/7acd2dcd-ac95-42e2-8a58-f30ec3c77376/download/metadata_sircilla.pdf", 105076),
    ("sircilla_india", "10.25678/0002VH", "readme.txt", f"{E}/e48f54ab-8783-4baa-8eec-1cbaa4457cdd/resource/0f46f29f-e52c-473b-85d2-3139d3697476/download/readme.txt", 667),
    # --- Sam et al. 2023: 10-city feces vs fecal sludge (supporting) ---
    ("sam2023_feces_to_fs", "10.25678/0007MP", "fecal_sludge_samples.csv", f"{E}/3886f09b-5def-4b75-ae01-60f6675161a0/resource/05c9d9f3-0629-423c-b97b-843dc8e5e4b2/download/fecal_sludge_samples.csv", 4267),
    ("sam2023_feces_to_fs", "10.25678/0007MP", "feces_samples.csv", f"{E}/3886f09b-5def-4b75-ae01-60f6675161a0/resource/6134993e-f6f3-40ff-853f-8615eedf985f/download/feces_samples.csv", 7058),
    ("sam2023_feces_to_fs", "10.25678/0007MP", "data_label_explanation_ss.txt", f"{E}/3886f09b-5def-4b75-ae01-60f6675161a0/resource/ce931360-202a-4760-9144-83a7e412365f/download/data_label_explanation_ss.txt", 5876),
    # --- Sam et al. 2022: EPS lab reactors (supporting) ---
    ("sam2022_eps", "10.25678/0006FP", "fs_batchreactor_data.csv", f"{E}/d46d626e-a679-4cba-b08a-b53cada93c8d/resource/87e1bdc0-5524-490d-aa02-ab21b58cce56/download/fs_batchreactor_data.csv", 28637),
    ("sam2022_eps", "10.25678/0006FP", "data_label_explanation.txt", f"{E}/d46d626e-a679-4cba-b08a-b53cada93c8d/resource/85bca0cd-c26a-4ab1-9ab0-7957714ae70e/download/data_label_explanation_.txt", 7561),
    # --- fecalcanuga (openwashdata): Kampala + Canada 2023-24 (core for Kampala sites) ---
    ("fecalcanuga", "github.com/openwashdata/fecalcanuga", "phys_chem_parameter.csv", f"{GH}/fecalcanuga/main/inst/extdata/phys_chem_parameter.csv", 32201),
    ("fecalcanuga", "github.com/openwashdata/fecalcanuga", "containment.csv", f"{GH}/fecalcanuga/main/inst/extdata/containment.csv", 3366),
    ("fecalcanuga", "github.com/openwashdata/fecalcanuga", "household_survey.csv", f"{GH}/fecalcanuga/main/inst/extdata/household_survey.csv", 14216),
    ("fecalcanuga", "github.com/openwashdata/fecalcanuga", "dictionary.csv", f"{GH}/fecalcanuga/main/data-raw/dictionary.csv", 9755),
    ("fecalcanuga", "github.com/openwashdata/fecalcanuga", "LICENSE.md", f"{GH}/fecalcanuga/main/LICENSE.md", 18651),
    # --- fsmglobal (openwashdata): emptying demand, 175 countries (context for planning) ---
    ("fsmglobal", "github.com/openwashdata/fsmglobal", "fsmglobal.csv", f"{GH}/fsmglobal/main/inst/extdata/fsmglobal.csv", 75892),
    ("fsmglobal", "github.com/openwashdata/fsmglobal", "dictionary.csv", f"{GH}/fsmglobal/main/data-raw/dictionary.csv", 1121),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "fs-ml-research/1.0"})
    for i in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            return
        except Exception as e:  # network hiccup: retry
            if i == tries:
                raise
            print(f"   retry {i} after error: {e}")
            time.sleep(3 * i)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for folder, doi, name, url, size in FILES:
        dest = RAW / folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (size is None or dest.stat().st_size == size):
            status = "already present"
        else:
            try:
                fetch(url, dest)
                got = dest.stat().st_size
                status = "OK" if size is None or got == size else f"SIZE MISMATCH ({got} vs {size})"
            except Exception as e:
                status = f"FAILED: {e}"
        ok = status in ("OK", "already present")
        print(f"[{'OK ' if ok else '!! '}] {folder}/{name}  {status}")
        if not ok:
            failed.append(f"{folder}/{name}")
        rows.append({
            "dataset": folder, "source": doi, "file": name, "url": url,
            "bytes": dest.stat().st_size if dest.exists() else "",
            "sha256": sha256(dest) if ok else "", "status": status,
        })
    with open(RAW / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows) - len(failed)}/{len(rows)} files OK. Manifest: {RAW / 'MANIFEST.csv'}")
    if failed:
        print("Failed files (re-run the script to retry):")
        for f in failed:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
