# TLE-WDN: Should this past run be reused?

**Risk-controlled transfer decisions for recurring leak–pressure management in water distribution networks**
(research track *Trajectory Learning Evolution*, TLE).

> Given a library of past evolutionary runs and a new leak scenario, decide before spending the budget whether to
> warm-start from one specific past run or to start fresh, so that expected savings are maximised while the rate of
> harmful transfers stays below a level α chosen by the operator.

Pump speeds and PRV settings after a leak are found with Differential Evolution on an EPANET model. Every solved
case leaves a run behind; warm-starting a new search from an old run's elites sometimes saves many simulations and
sometimes traps the search (negative transfer). This repository builds the whole study: simulator, optimiser,
transfer action, paired-run benchmark, predictors, a Learn-then-Test gate with a finite-sample harm bound, the
SEETO-, BCKT- and PMTO-style baselines, and generalisation experiments (goal change, layout change, held-out
network).

* Results: **[RESULTS.md](RESULTS.md)** (generated from result files) · figures in `figures/`
* Plan and status: [PROJECT_PLAN.md](PROJECT_PLAN.md) · [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) ·
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) · [ARCHITECTURE.md](ARCHITECTURE.md)
* Frozen settings: [docs/preregistration.md](docs/preregistration.md) · decisions: [docs/decisions.md](docs/decisions.md)
* Baselines and adaptations: [docs/baselines.md](docs/baselines.md) · reading notes: `docs/reading/`
* Data dictionary: [docs/data_dictionary.md](docs/data_dictionary.md) · timing: [docs/timing.md](docs/timing.md)

## Installation

Python ≥ 3.11 (developed on 3.12, Windows 11; pure-Python dependencies, EPANET 2.2 ships inside WNTR for
Windows/Linux/macOS).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest
```

`make env` and `make test` do the same where `make` exists. `requirements-lock.txt` pins every package (WNTR 1.5.0,
numba, LightGBM, scikit-learn, …); simulator versions change numerical results, so the lock file is part of the
experiment.

## Data

* **Net3** is the EPANET example network shipped with WNTR; `src/tle/hydraulics/network_prep.py` turns it into
  `data/networks/net3_prv.inp` (two PRV zones, controls removed, pressure-driven demand, SI units). The edited file
  and its metadata (edits, zones, hashes) are committed.
* **L-Town** (BattLeDIM, EUPL-1.2) is downloaded by `scripts/e4_prepare.py` from the BattLeDIM repository into
  `data/networks/downloads/` (not committed) and converted to `data/networks/ltown_prv.inp`.
* Scenario tables, split manifests and pair manifests (`data/scenarios/`) are generated from a recorded master seed
  and committed with their SHA-256 hashes.

## Running the study

```bash
.venv/Scripts/python scripts/reproduce.py              # everything (resumable; several hours on a laptop)
.venv/Scripts/python scripts/reproduce.py --only e1    # main benchmark and headline tables
.venv/Scripts/python scripts/reproduce.py --only report
```

Stage by stage (each script documents itself with `--help`):

| Step | Script | Output |
|---|---|---|
| objective weights (normalisation rule) | `calibrate_weights.py` | `configs/generated/*_weights.yaml` |
| scenarios, splits, pairs | `make_scenarios.py` | `data/scenarios/e1_*` |
| DE tuning and budget rule | `tune_de.py` | `configs/generated/de_tuning.yaml` |
| pilot map, k/ρ grid | `pilot.py` | `results/e1/pilot/`, `figures/f2_*` |
| library, reference and paired runs | `run_campaign.py` | `results/e1/jobs/` (not committed) |
| tables, labels, features, audit, dictionary | `build_tables.py` | `results/e1/tables/*.parquet`, `docs/data_dictionary.md` |
| tier ablation M0–M6 | `phase4_models.py` | `results/e1/phase4/`, `figures/f3_*` |
| gate, re-splits, probe mode, test evaluation | `phase5_gate.py` | `results/e1/phase5/`, `figures/f4–f6` |
| baselines B0–B7, oracle, statistics | `phase6_baselines.py` | `results/e1/phase6/`, `figures/f7_*` |
| goal / layout shift | `make_shift_scenarios.py`, `phase7_shift.py` | `results/e2/`, `results/e3/` |
| held-out network | `e4_prepare.py`, `phase7_shift.py`, `phase7_e4.py` | `results/e4/` |
| larger L-Town holdout (E4z) | `e4z_campaign.py`, `e4z_build_tables.py`, `phase7_shift.py`, `phase7_e4.py` | `configs/e4z.yaml`, `results/e4z/` |
| sensitivity, amortisation, report | `sensitivity.py`, `phase8_report.py` | `results/e1/sensitivity/`, `RESULTS.md` |

Set `TLE_JOBS` to limit worker processes (default: CPUs − 1).

E4z reuses the committed E4y training and calibration aggregates and adds 15
independent L-Town test scenarios (35 total). Reproduce its simulations and
analysis with `scripts/e4z_campaign.py --jobs 8`, then
`scripts/e4z_build_tables.py`, `scripts/phase7_shift.py --config configs/e4z.yaml`,
and `scripts/phase7_e4.py --config configs/e4z.yaml`. The simulation runner
resumes completed jobs if interrupted.

## Method in one paragraph

DE/rand/1/bin (own implementation, exact evaluation budget B) optimises 24 decision variables (two pump speed
multipliers and two PRV settings over six 4-hour windows) of a penalised objective (leak volume, energy cost under a
time-of-use tariff, service-pressure shortfall, storage deficit) on a pressure-driven EPANET model with an orifice
leak. The transfer action injects the top-3 distinct elites of one past run (with 3% noise) into 40% of the initial
population. For every target scenario and candidate source, fresh and warm runs with common random numbers give a
signed, matched-budget utility. Gradient-boosted predictors trained on scenario, trajectory and hydraulic-signature
features estimate a utility lower bound and a harm probability; the gate transfers from the best-ranked source only
if its harm probability is below a threshold certified by Learn-then-Test on calibration targets.

## Extensions: shift-aware, pooled and monitored decisions

The core experiments show where decide-once transfer fails (rare harm on Net3/DE, broken guarantee under layout
shift, no certification on a new network with few calibration cases). The extensions ([docs/v2_plan.md](docs/v2_plan.md)) add:

* **trajectory monitoring** (`src/tle/v2/monitor.py`): continue or abort a warm start after g generations, comparing
  the warm run's progress with the expected fresh trajectory; exact offline evaluation from logged runs;
* **shift-aware gate** (`src/tle/v2/shift.py`): density-ratio-weighted Learn-then-Test with an effective sample size
  that collapses (→ abstain) when calibration and new targets do not overlap;
* **pooled cross-network learning and calibration** (Net3 + L-Town);
* an optimiser-robustness check with a genetic algorithm (`scripts/ga_check.py`) and 40 extra L-Town calibration
  targets (`configs/e4x.yaml`).

```bash
.venv/Scripts/python scripts/v2_analysis.py monitor --config configs/e4x.yaml
.venv/Scripts/python scripts/v2_analysis.py shift
.venv/Scripts/python scripts/v2_analysis.py pooled
```

## Repository layout

```
configs/            experiment configs (E1-E4) + generated frozen values
data/networks/      edited .inp files and metadata
data/scenarios/     scenario tables, split and pair manifests (+ hashes)
docs/               pre-registration, decisions, baselines (T1), reading notes, data dictionary, timing
src/tle/            package (hydraulics, de, scenarios, features, models, baselines, analysis)
scripts/            one script per phase step; reproduce.py runs them all
results/            analysis outputs (CSV/JSON); raw job files and Parquet tables are regenerated
figures/            generated figures only
tests/              unit tests and sanity tests T1-T5
```

## Known limitations

See the limitations section of [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md): single optimiser, benchmark
networks, guarantee relative to the finite-repeat harm label and to exchangeability, re-implemented (not the
authors') baselines, cheap simulations (the study claims evaluation efficiency, not wall-clock savings of
expensive simulators).

## Real-network calibration extension

The proposed next validation step using the public BattLeDIM historical L-Town measurements is documented in
[`docs/battledim_calibration.md`](docs/battledim_calibration.md). It keeps the source data out of Git, separates
chronological calibration and validation periods, checks the nominal EPANET model before tuning, and distinguishes
model-based evaluation from real operational deployment.

Run `python scripts/calibrate_battledim.py` to download and checksum the required public files, fit the initial
nominal-model calibration, and regenerate the held-out report and calibrated EPANET input under
`results/battledim_calibration/`.

## Licence

Code: MIT (see `LICENSE`). L-Town derived files: EUPL-1.2 (BattLeDIM). Net3: public domain (EPANET examples).
