"""LBNL FDD data sets, "RTU" (LBNL, NREL, ORNL, 2022), doi:10.25984/1881324, CC BY 4.0.

Source: https://faultdetection.lbl.gov/dataset/rtu/ (one archive with the
experimental, field and simulated RTU sets, and the data inventory). Only the
simulated set (folder "Simulated RTU") is used; its CSV files are read from
the archive and cached as parquet under data/interim/lbnl_rtu/.
"""

from _common import Remote, fetch

DATASET = "lbnl_rtu"
BASE = "https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_RTU/"
FILES = [
    Remote(BASE + "LBNL_FDD_Data_Sets_RTU.zip", "LBNL_FDD_Data_Sets_RTU.zip", 305603469),
    Remote(BASE + "LBNL_FDD_Data_Sets_RTU.pdf", "LBNL_FDD_Data_Sets_RTU.pdf", 1643253),
]

if __name__ == "__main__":
    for remote in FILES:
        fetch(DATASET, remote)
    from cfc.datasets.lbnl_rtu import CASES, load_case

    for case in CASES:
        load_case(case)
