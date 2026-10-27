# chiller-fault-consequences

Code and reproduction pipeline for the paper *Which Fault to Repair First?
Estimating the Consequences of Chiller-Plant Faults from Building Automation
Data*.

A building automation system with fault detection and diagnostics reports more
faults than a maintenance team can repair. The paper estimates the operational
consequence of each diagnosed fault (its energy penalty against a fault-free
counterfactual and its deficit of chilled-water supply) from building
automation data, with and without power meters, and measures how much better
repair decisions become when faults are ranked by this estimate instead of by
documented intensity, fault-type priority or anomaly magnitude. The data are
the public LBNL chiller-plant set (one simulated year per fault, same weather
as the fault-free year); the simulated LBNL rooftop unit serves as replication.

**Every figure and every number in the paper is produced by a script in this
repository.** This README walks you through running the whole pipeline and
checking that you obtain exactly the same results.

---

## What you need

| | |
|---|---|
| Python | 3.11 or newer. The paper was produced with **Python 3.13.12** on Windows 11 (x86-64). |
| Disk | about 4 GB free (two archives of 1.6 GB and 0.3 GB, 1.9 GB of parquet cache) |
| Network | to download the public datasets once |
| Time | download, about 15 s of tests and about 2.5 minutes of experiments |
| `make` | optional; every step below also has a plain command |

No GPU, no commercial software and no account are needed.

---

## Step 1: get the code

```bash
git clone https://github.com/Pletea-Marinescu-Valentin/chiller-fault-consequences.git
cd chiller-fault-consequences
```

## Step 2: create the Python environment

Install the exact package versions used for the paper from
`requirements-lock.txt`, then the project itself.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

All later commands assume this environment is active.

## Step 3: run the unit tests

```bash
pytest            # or: make test
```

Expected: `24 passed` in about 15 seconds. The tests run on synthetic data and
small hand-made frames, among others:

- the decision value is 1 for the oracle, about 0 for a constant score and
  negative for an inverted one; portfolios hold distinct faults and weeks;
- cross-fitted models never predict a fold they were fitted on;
- the virtual meter recovers a synthetic plant's power out of fold, ignores
  points that drive no component, and drops or replaces a diagnosed point;
- unit conversion and the exchange of the two mislabelled column pairs.

## Step 4: download the data

```bash
python scripts/download/lbnl_chiller.py     # or: make data (both datasets)
python scripts/download/lbnl_rtu.py
```

The scripts fetch the public LBNL FDD data sets "Chiller plant" and "RTU"
(doi:10.25984/1881324, CC BY 4.0) from fdddata.lbl.gov. Each script:

1. downloads the archive and its data inventory (PDF) into `data/raw/<dataset>/`
   (resumable);
2. checks their sizes and the SHA-256 digests pinned in
   `scripts/download/SHA256SUMS`, and stops if any of them differs;
3. converts the CSV files it needs to parquet under `data/interim/<dataset>/`
   (1-min and hourly), reading them straight from the archive.

The dataset pages ask users to register through a short form; please do so if
you use the data. Nothing in `data/` is committed to this repository.

## Step 5: run the experiments

With `make`:

```bash
make experiments
```

Without `make` (the order matters: `tab_estimation` writes the weekly estimates
that `fig_ranking` reads):

```bash
for e in tab_audit fig_consequences fig_weekly tab_estimation fig_ranking tab_rtu; do
  python experiments/$e.py || break
done
```

Each script reads its settings and fixed random seed from `configs/<name>.yaml`
and writes:

- figures to `paper/figs/<name>.pdf`;
- result tables to `results/tables/<name>.csv`;
- intermediate results to `results/<name>.json`;
- every number quoted in the paper to `paper/generated/<name>.tex`, as LaTeX
  macros, and the paper's tables to `paper/generated/<name>_table.tex`.

| Script | Paper item | Run time* |
|---|---|---|
| `experiments/tab_audit.py` | Sec. III-B, data audit: files, swapped labels, duplicates, plant regimes | 8 s |
| `experiments/fig_consequences.py` | Table I and Fig. 1, annual consequences of every fault | 4 s |
| `experiments/fig_weekly.py` | Fig. 2, weekly plant energy and penalties | 2 s |
| `experiments/tab_estimation.py` | Table II, weekly estimates with and without power meters | 1 min 30 s |
| `experiments/fig_ranking.py` | Table III and Fig. 3, decision value of the priority rules | 7 s |
| `experiments/tab_rtu.py` | Sec. IV-D, replication on the simulated rooftop unit | 10 s |

\* Measured on a 16-thread x86-64 laptop.

## Step 6: check that you obtained the same results

```bash
python scripts/verify_numbers.py     # or: make verify
```

The script compares every number your run wrote to `paper/generated/` with the
reference values stored in `results/tables/paper_numbers.csv`. The paper quotes
a subset of these numbers; the rest are additional values written by the same
scripts. The expected output is:

```
390 of 390 numbers match the paper.
```

Any difference is listed with the macro name, the reference value and your
value, and the script exits with status 1. The result tables and figures are
also tracked in git, so

```bash
git status --short results/tables paper/figs
```

should print nothing. The figure PDFs can differ byte-wise if your system lacks
the Times New Roman font; the numbers are not affected.

`make reproduce` runs Steps 3, 5 and 6 in one go.

---

## Determinism

- **Seeds.** All randomness (gradient-boosted trees, portfolio sampling, tie
  breaking) is seeded from the YAML files in `configs/`.
- **Package versions.** Gradient-boosted trees (LightGBM) and numerical
  libraries can change results slightly between versions. Use
  `requirements-lock.txt` for an exact match.

## Notes on the data

The loaders correct and document a few properties of the archives:

- **Swapped labels (chiller plant).** `OA_TEMP` and `OA_TEMP_WB` are
  exchanged, and so are `CWL_SEC_SW_TEMP` and `CWL_SEC_RW_TEMP`. The evidence
  (wet bulb above dry bulb, the tower setpoint and chilled-water reset
  equations, the logged secondary load) is recomputed by `tab_audit.py`.
- **Undocumented cases.** The chiller-plant archive holds two chiller-fouling
  years (`chiller_fouling_065`, `chiller_fouling_095`) that the inventory does
  not list.
- **Near duplicates.** Bypass-valve leakage and stuck valve at 50 % and 75 %
  have the same annual consequences.
- **Rooftop unit humidity.** `RTU_OA_HUM` differs between runs although the
  outdoor temperature agrees, so it is not used as weather.

## How the code is organized

| Path | Content |
|---|---|
| `src/cfc/datasets/` | loaders for the LBNL chiller plant and the simulated rooftop unit (SI units, label fixes, parquet cache) |
| `src/cfc/context.py` | exogenous context features, weekly folds and weekly sums |
| `src/cfc/consequences.py` | paired ground-truth energy penalty, service deficit and the priced consequence |
| `src/cfc/estimators.py` | cross-fitted counterfactual models, the observed-minus-counterfactual estimate, the anomaly score |
| `src/cfc/virtual_meter.py` | component virtual meter on basic points, diagnosis-aware inputs, envelope check |
| `src/cfc/proxies.py` | documented intensity and fault-type priority |
| `src/cfc/ranking.py` | Kendall tau, random portfolios and the decision value |
| `configs/` | one YAML file per experiment, with all parameters and seeds |
| `experiments/` | one script per figure or table of the paper |
| `scripts/download/` | dataset download with SHA-256 verification and parquet caching |
| `scripts/verify_numbers.py` | comparison of your numbers with the reference |
| `results/tables/` | result tables, weekly estimates and the reference numbers |
| `paper/figs/` | the figures of the paper |
| `tests/` | unit tests |

The manuscript source is not part of this repository; `make paper` only works
with it.

## Data source

Lawrence Berkeley National Laboratory, LBNL FDD Data Sets (chiller plant and
rooftop unit), doi:10.25984/1881324, CC BY 4.0. Described in: Granderson, J.,
Lin, G., Chen, Y., Casillas, A., et al. (2023). A labeled dataset for building
HVAC systems operating in faulted and fault-free states. *Scientific Data*, 10,
342. doi:10.1038/s41597-023-02197-w. Please cite the dataset if you use it.

## License

The code is released under the MIT License; see `LICENSE`. The datasets are
distributed by their authors under their own terms.
