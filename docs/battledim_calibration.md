# BattLeDIM calibration extension

## Purpose and scope

The current L-Town experiment uses an EPANET model and generated operating scenarios. This extension is the next validation step: use the public BattLeDIM historical measurements to calibrate/check the L-Town hydraulic model before evaluating warm starts. BattLeDIM is based on a real Cyprus network, but its published network model is nominal and the challenge measurements/model are not a live utility control interface. Report this as validation against a real-network-derived benchmark, not a field deployment.

## Public inputs

Download the BattLeDIM dataset from [Zenodo](https://doi.org/10.5281/zenodo.4017659) and keep the source archive outside Git. The release includes historical 2018 five-minute SCADA measurements, leak reports, the nominal EPANET input, and a real-network model. Record the dataset version, checksum, and licence with each run. Do not commit the archive or raw SCADA files.

## Calibration and evaluation protocol

1. Use the unmodified BattLeDIM network model for calibration. Do not calibrate the project’s edited `ltown_prv.inp`: it has controls removed and a different 24-hour setup, so it is not the model used to produce the published historical measurements.
2. Map SCADA sensors to model node/link IDs from the dataset documentation. Retain only timestamps and sensors with valid observations and matching model locations. Convert units explicitly and document missing-data handling.
3. Split by time, before tuning: use an earlier contiguous period for calibration and a later period for held-out validation. Do not randomly split five-minute rows; adjacent readings are temporally dependent.
4. First fit only observable nuisance parameters (for example, demand multipliers/pattern factors and a small, documented set of pipe roughness or pump parameters). Minimize a normalized error over observed pressure, flow, and tank-level series, with physically plausible bounds and regularization toward nominal model values. Keep sensor types separately scored so one unit or high-volume sensor cannot dominate.
5. Report train and held-out error by sensor and time period, plus the nominal-model baseline. Reject a calibration that improves only the fitting period or requires implausible parameter values.
6. After calibration is frozen, create leak optimization scenarios from documented historical conditions and leak reports. Use the historical data only to define realistic demands, tank levels, and event distributions; use EPANET to evaluate alternative controls. Keep a final chronological event block untouched until the final comparison.
7. Compare fresh DE, warm-start DE, and the decision policy at matched evaluation budgets and seeds. Report network calibration error alongside optimizer utility, feasibility, leakage, energy, and pressure outcomes. Do not claim real-world operational benefit from simulator outcomes alone.

## Important limitation

Historical SCADA observations do not directly reveal the result of pump/PRV settings that were never operated. The proposed optimizer can therefore be tested retrospectively only through the calibrated hydraulic model. A utility-approved model-in-the-loop study or prospective operator shadow trial is needed to validate unseen control interventions.

## Reference

BattLeDIM dataset and official documentation: [dataset (Zenodo)](https://doi.org/10.5281/zenodo.4017659), [data description](https://battledim.ucy.ac.cy/?page_id=33), [challenge site](https://battledim.ucy.ac.cy/).
