# BattLeDIM calibration extension

## Purpose and scope

The current L-Town experiment uses an EPANET model and generated operating scenarios. This extension is the next validation step: use the public BattLeDIM historical measurements to calibrate/check the L-Town hydraulic model before evaluating warm starts. BattLeDIM is based on a real Cyprus network, but its published network model is nominal and the challenge measurements/model are not a live utility control interface. Report this as validation against a real-network-derived benchmark, not a field deployment.

## Public inputs

The [Zenodo release](https://doi.org/10.5281/zenodo.4017659) includes historical 2018 five-minute SCADA measurements, leak reports, the nominal EPANET input, and a real-network model. The script downloads only the nominal model, pressure/flow/level data, and configuration, verifies their published checksums, and keeps them outside Git. It does not download demand records or the very large real-network model because this first-pass fit does not use them.

## Completed first-pass calibration

`scripts/calibrate_battledim.py` downloads the official nominal model, 2018 SCADA pressure/flow/tank-level CSVs, and sensor configuration from BattLeDIM Zenodo on first run, verifies the published MD5 checksums, and keeps them under the ignored `data/networks/downloads/battledim/` directory. The 2018 leak configuration shows the first listed event beginning on 8 January, so the script uses Jan 1–5 for calibration and Jan 6–7 as a chronological holdout, sampled hourly from the five-minute records. It fits two global multipliers—base demand and pipe roughness—inside ±10% of nominal using Differential Evolution. Sensor types are weighted equally, with each sensor error normalized by its observed standard deviation.

Run (internet access is needed on first run):

```powershell
python scripts/calibrate_battledim.py
```

The checked-in report is [`results/battledim_calibration/calibration_report.json`](../results/battledim_calibration/calibration_report.json). On the released data, demand multiplier = 0.91116 and global pipe roughness multiplier = 1.03051. The equal-sensor-type normalized RMSE falls from 1.999 to 1.949 on Jan 1–5 and from 2.984 to 2.571 on held-out Jan 6–7 (13.8% lower held-out error). The script writes the calibrated nominal model and a calibrated copy of the project's optimizer-ready L-Town network under `results/battledim_calibration/`; its metadata records the source and hashes. Raw observations remain outside Git. These errors remain large, so this is a modest first-pass calibration, not a claim that the model is operationally validated.

## Calibration and evaluation protocol

1. Fitting uses the unmodified BattLeDIM `L-TOWN.inp`. It does not alter the project's existing `ltown_prv.inp` or benchmark outputs; a separate calibrated optimizer-ready copy is written beneath the results directory.
2. Sensor mapping is read from the official dataset configuration. Flow observations in m³/h are converted to L/s; pressure and tank level are in metres. Five-minute observations are sampled at hourly timestamps with a three-minute matching tolerance.
3. The split is chronological: Jan 1–5 for fitting, Jan 6–7 held out. This is a short, known leak-free interval, which limits seasonal and operating-regime coverage.
4. The first-pass fit changes only the global demand multiplier and global pipe roughness multiplier, each bounded to ±10% of nominal. Pressure, flow, and tank-level sensor groups contribute equally; sensor residuals are scaled by each sensor’s observed standard deviation.
5. The report includes nominal and calibrated train/holdout errors by sensor. The observed held-out improvement is encouraging but the residual normalized error remains high; do not treat this model as operationally validated.
6. A stronger next study needs leak-aware model inputs for periods after Jan 8 and additional seasonal holdouts. Only after that calibration is validated should historical demands/tank levels be used to build optimization scenarios and compare fresh DE with warm starts.
7. A utility-approved model-in-the-loop or shadow trial remains necessary to validate unseen control interventions. Historical SCADA cannot reveal the outcome of control settings that were never applied.

## Important limitation

Historical SCADA observations do not directly reveal the result of pump/PRV settings that were never operated. The proposed optimizer can therefore be tested retrospectively only through the calibrated hydraulic model. A utility-approved model-in-the-loop study or prospective operator shadow trial is needed to validate unseen control interventions.

## Reference

BattLeDIM dataset and official documentation: [dataset (Zenodo)](https://doi.org/10.5281/zenodo.4017659), [data description](https://battledim.ucy.ac.cy/?page_id=33), [challenge site](https://battledim.ucy.ac.cy/).
