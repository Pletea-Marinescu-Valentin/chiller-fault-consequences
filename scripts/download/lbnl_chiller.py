"""LBNL FDD data sets, "Chiller plant" (LBNL and PNNL, 2022), doi:10.25984/1881324, CC BY 4.0.

Source: https://faultdetection.lbl.gov/dataset/simulated-chiller-plant/ (one
archive with 24 one-year CSV files, and the data inventory). The archive is not
extracted: the CSV files are read from it once and cached as parquet under
data/interim/lbnl_chiller/.
"""

from _common import Remote, fetch

DATASET = "lbnl_chiller"
BASE = "https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_Chiller_Plant/"
FILES = [
    Remote(BASE + "LBNL_FDD_Data_Sets_Chiller_Plant.zip", "LBNL_FDD_Data_Sets_Chiller_Plant.zip",
           1570938957),
    Remote(BASE + "LBNL_FDD_Data_Sets_Chiller_Plant.pdf", "LBNL_FDD_Data_Sets_Chiller_Plant.pdf",
           754410),
]

if __name__ == "__main__":
    for remote in FILES:
        fetch(DATASET, remote)
    from cfc.datasets.lbnl_chiller import build_cache

    build_cache()
