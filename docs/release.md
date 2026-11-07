# Release checklist (steps that need the owner)

## 1. Dataset DOI (Zenodo)
1. Run `scripts/reproduce.py --only report` so `RESULTS.md` and figures are current.
2. Create an archive of the release data:
   `results/*/tables/*.parquet` (except `generations.parquet`, or include it if size allows), `data/scenarios/`,
   `data/networks/*.inp` and `*.meta.json`, `configs/`, `docs/data_dictionary.md`, `docs/preregistration.md`.
3. On zenodo.org (signed in with your account) create a new upload, attach the archive, licence CC-BY-4.0 for data
   (L-Town-derived files stay EUPL-1.2), add the GitHub release `v2.0` as a related identifier, publish.
4. Put the DOI into `README.md` and `paper/manuscript.md` (Data and code availability).

## 2. Second-machine reproduction (build guide P8.5)
On another computer:
```
git clone https://github.com/likhith1253/watercdn && cd watercdn
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-lock.txt && .venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/reproduce.py --only e1
```
Compare the regenerated `results/e1/phase6/t2_main_results.csv` with the committed one (identical results are
expected on the same OS; floating-point differences across OS/CPU are possible and should be reported).

## 3. Preprint
Complete author list, affiliations and the AI-assistance statement in `paper/manuscript.md`, convert to the venue
template (e.g. Elsevier LaTeX for *Swarm and Evolutionary Computation* / *Applied Soft Computing*), and post on arXiv
(cs.NE) before submission.
