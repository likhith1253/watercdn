# BattLeDIM calibration extension

## Purpose and scope

The current L-Town experiment uses an EPANET model and generated operating scenarios. This extension is the next validation step: use the public BattLeDIM historical measurements to calibrate/check the L-Town hydraulic model before evaluating warm starts. BattLeDIM is based on a real Cyprus network, but its published network model is nominal and the challenge measurements/model are not a live utility control interface. Report this as validation against a real-network-derived benchmark, not a field deployment.

## Public inputs

The [Zenodo release](https://doi.org/10.5281/zenodo.4017659) includes historical 2018 five-minute SCADA measurements, leak reports, the nominal EPANET input, and a real-network model. The script downloads only the nominal model, pressure/flow/level data, and configuration, verifies their published checksums, and keeps them outside Git. It does not download demand records or the very large real-network model because this first-pass fit does not use them.

## Reproduced first-pass calibration

`scripts/calibrate_battledim.py` downloads the official nominal model, 2018 SCADA pressure/flow/tank-level CSVs, and sensor configuration from BattLeDIM Zenodo on first run, verifies the published MD5 checksums, and keeps them under the ignored `data/networks/downloads/battledim/` directory. The first reported leak begins on 8 January, so the script uses Jan 1–5 for calibration and Jan 6–7 as a chronological holdout, sampled hourly from the five-minute records. This is a pre-first-reported-leak interval, not proof that no unreported leakage was present. It fits two global multipliers—base demand and pipe roughness—inside ±10% of nominal using Differential Evolution. Sensor types are weighted equally, with each sensor error normalized by its observed standard deviation. The source model's native 5-minute demand-pattern interval is retained. BattLeDIM meter flow (m³/h) and WNTR simulated flow (m³/s) are both explicitly converted to L/s before comparison.

Run (internet access is needed on first run):

```powershell
python scripts/calibrate_battledim.py
```

The checked-in report is [`results/battledim_calibration/calibration_report.json`](../results/battledim_calibration/calibration_report.json). The corrected run estimates demand multiplier = 1.00608 and global pipe roughness multiplier = 1.04853. Equal-sensor-type normalized RMSE falls from 0.501 to 0.471 on Jan 1–5 and from 0.953 to 0.912 on held-out Jan 6–7 (4.4% lower held-out aggregate error). The holdout breakdown is more informative: mean pressure RMSE improves from 0.224 to 0.196 m (normalized 1.146 to 0.991); mean flow RMSE changes from 5.06 to 5.13 L/s (normalized 1.102 to 1.120); tank-level RMSE changes from 0.263 to 0.269 m (normalized 0.612 to 0.624). Thus the fit modestly improves pressure but does not improve flow or tank level. A normalized RMSE near 1 means the residual is on the order of that sensor's variation within this short holdout; the physical RMSE values are provided so readers can judge magnitude in engineering units. The script also writes the calibrated nominal model and a separate optimizer-ready L-Town copy under `results/battledim_calibration/`; metadata records source and hashes. Raw observations remain outside Git. These are benchmark fit results, not validation of a utility model or unseen control settings.

The earlier report values were invalid: the calibration had overridden the model's 5-minute demand-pattern interval and compared WNTR flow in m³/s directly with observations in L/s. The current script corrects both issues and includes regression coverage for the flow conversions. Do not compare the earlier scores with the corrected run.

## Calibration and evaluation protocol

1. Fitting uses the unmodified BattLeDIM `L-TOWN.inp`. It does not alter the project's existing `ltown_prv.inp` or benchmark outputs; a separate calibrated optimizer-ready copy is written beneath the results directory.
2. Sensor mapping is read from the official dataset configuration. Flow observations in m³/h and simulated WNTR flow in m³/s are each converted to L/s; pressure and tank level are in metres. Five-minute observations are sampled at hourly timestamps with a three-minute matching tolerance. The native 300-second demand-pattern interval is preserved in the model.
3. The split is chronological: Jan 1–5 for fitting, Jan 6–7 held out. It precedes the first reported leak, but unreported leakage cannot be ruled out. This short split covers only one weekend and limits seasonal and operating-regime coverage.
4. The first-pass fit changes only the global demand multiplier and global pipe roughness multiplier, each bounded to ±10% of nominal. Pressure, flow, and tank-level sensor groups contribute equally; sensor residuals are scaled by each sensor’s observed standard deviation.
5. The report includes nominal and calibrated train/holdout errors by sensor and by measurement type. Improvement is modest and confined to pressure on this split; flow and tank-level residuals do not improve. The heldout covers just one weekend, so it does not establish year-round predictive performance.
6. A stronger next study needs leak-aware model inputs for periods after Jan 8 and additional seasonal holdouts. Only after that calibration is validated should historical demands/tank levels be used to build optimization scenarios and compare fresh DE with warm starts.
7. A utility-approved model-in-the-loop or shadow trial remains necessary to validate unseen control interventions. Historical SCADA cannot reveal the outcome of control settings that were never applied.

## Important limitation

Historical SCADA observations do not directly reveal the result of pump/PRV settings that were never operated. The proposed optimizer can therefore be tested retrospectively only through the calibrated hydraulic model. A utility-approved model-in-the-loop study or prospective operator shadow trial is needed to validate unseen control interventions.

## Reference

BattLeDIM dataset and official documentation: [dataset (Zenodo)](https://doi.org/10.5281/zenodo.4017659), [data description](https://battledim.ucy.ac.cy/?page_id=33), [challenge site](https://battledim.ucy.ac.cy/).
