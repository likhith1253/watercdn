"""Fit and validate the nominal BattLeDIM L-Town EPANET model against 2018 SCADA.

This reproducible first-pass calibration uses only the interval before the
first reported leak in 2018. Undocumented leaks cannot be ruled out. It fits
global demand and pipe-roughness multipliers on
Jan 1-5 and reports held-out Jan 6-7 scores. It does not calibrate the edited
ltown_prv optimization network or claim validation of unobserved interventions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import wntr
import yaml
from scipy.optimize import differential_evolution

from tle.hydraulics.network_prep import build_ltown_prv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/networks/downloads/battledim"
DEFAULT_OUT = ROOT / "results/battledim_calibration"
START = pd.Timestamp("2018-01-01 00:00:00")
END = pd.Timestamp("2018-01-08 00:00:00")
FIT_END = pd.Timestamp("2018-01-06 00:00:00")
ZENODO_FILES = {
    "L-TOWN.inp": "bfbb16b8b463e0576200ba6ad360f68e",
    "2018_SCADA_Pressures.csv": "d389d8541350c19ff0bfc6b80f246d35",
    "2018_SCADA_Flows.csv": "d0602c06946b46287e956f007e4264ee",
    "2018_SCADA_Levels.csv": "92a512d6749ddf66fd6d99f2bdde6cc4",
    "dataset_configuration.yaml": "48486401f5b4d0447023f5ce5d242c52",
}


def ensure_dataset(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    for name, expected_md5 in ZENODO_FILES.items():
        path = data / name
        if not path.exists():
            url = f"https://zenodo.org/api/records/4017659/files/{name}/content"
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        if actual != expected_md5:
            raise ValueError(f"Checksum mismatch for {path.name}: {actual} != {expected_md5}")


def normalize_inp(path: Path) -> None:
    """Drop trailing spaces emitted by WNTR's EPANET writer (valid INP whitespace)."""
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")


def read_scada(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["Timestamp"])
    return df.set_index("Timestamp").sort_index()


def scada_flow_to_lps(flow_m3h: pd.DataFrame) -> pd.DataFrame:
    """Convert BattLeDIM flow-meter units (m3/h) to L/s."""
    return flow_m3h / 3.6


def wntr_flow_to_lps(flow_m3s: pd.DataFrame) -> pd.DataFrame:
    """Convert WNTR's SI simulation result units (m3/s) to SCADA units (L/s)."""
    return flow_m3s * 1000.0


def load_observations(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pressure = read_scada(data / "2018_SCADA_Pressures.csv")
    flow = scada_flow_to_lps(read_scada(data / "2018_SCADA_Flows.csv"))
    levels = read_scada(data / "2018_SCADA_Levels.csv")
    hourly = pd.date_range(START, END, freq="h", inclusive="left")
    def hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reindex(hourly, method="nearest", tolerance=pd.Timedelta(minutes=3))
    return hourly_frame(pressure), hourly_frame(flow), hourly_frame(levels)


def simulate(inp: Path, demand_scale: float, roughness_scale: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inp = Path(inp).resolve()
    wn = wntr.network.WaterNetworkModel(str(inp))
    wn.options.time.duration = int((END - START).total_seconds())
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    # Keep the model's native 5-minute pattern interval. Overriding this to
    # hourly makes EPANET advance through only every 12th demand multiplier.
    wn.options.time.pattern_start = 0
    wn.options.hydraulic.demand_multiplier = float(demand_scale)
    for _, pipe in wn.pipes():
        pipe.roughness *= float(roughness_scale)
    # EPANET writes scratch .inp/.bin/.rpt files. Isolate them so a calibration
    # run leaves the project clean and concurrent runs do not collide.
    with tempfile.TemporaryDirectory(prefix="battledim_epanet_") as scratch:
        result = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(scratch) / "epanet"))
    idx = pd.date_range(START, END, freq="h", inclusive="left")
    # EPANET results use seconds from simulation start; use hourly points only.
    pressure = result.node["pressure"].reindex(range(0, len(idx) * 3600, 3600)).set_axis(idx)
    # WNTR result tables use SI m3/s internally; compare to SCADA in L/s.
    flow = wntr_flow_to_lps(result.link["flowrate"].reindex(range(0, len(idx) * 3600, 3600)).set_axis(idx))
    # EPANET reports tank water surface as head; subtract the tank's bottom elevation.
    level = result.node["head"].reindex(range(0, len(idx) * 3600, 3600)).set_axis(idx)
    for tank in wn.tank_name_list:
        level[tank] -= float(wn.get_node(tank).elevation)
    return pressure, flow, level


def sensor_rmse(pred: pd.DataFrame, obs: pd.DataFrame, names: list[str], mask: np.ndarray) -> dict:
    scores = {}
    for name in names:
        if name not in pred or name not in obs:
            continue
        a, b = pred.loc[mask, name].to_numpy(float), obs.loc[mask, name].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        scores[name] = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
    return scores


def composite(scores: dict[str, float], obs: pd.DataFrame, names: list[str], mask: np.ndarray) -> float:
    scales = []
    for name in scores:
        x = obs.loc[mask, name].to_numpy(float)
        scales.append(max(float(np.nanstd(x)), 1e-6))
    return float(np.mean([v / s for v, s in zip(scores.values(), scales)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--maxiter", type=int, default=4, help="DE calibration iterations (default: 4)")
    args = ap.parse_args()
    inp = args.data_dir / "L-TOWN.inp"
    ensure_dataset(args.data_dir)
    pattern_timestep_seconds = int(wntr.network.WaterNetworkModel(str(inp)).options.time.pattern_timestep)
    obs_p, obs_f, obs_l = load_observations(args.data_dir)
    sensor_cfg = yaml.safe_load((args.data_dir / "dataset_configuration.yaml").read_text(encoding="utf-8"))
    fit_mask = np.asarray(obs_p.index < FIT_END)
    valid_mask = np.asarray(obs_p.index >= FIT_END)
    # Include sensors that are explicitly listed in the BattLeDIM configuration.
    pnames = [s for s in sensor_cfg["pressure_sensors"] if s in obs_p]
    fnames = [s for s in sensor_cfg["flow_sensors"] if s in obs_f]
    lnames = [s for s in sensor_cfg["level_sensors"] if s in obs_l]

    def evaluate(x: np.ndarray, mask: np.ndarray) -> tuple[float, dict, dict]:
        p, f, l = simulate(inp, float(x[0]), float(x[1]))
        pieces = {
            "pressure": (sensor_rmse(p, obs_p, pnames, mask), obs_p, pnames),
            "flow": (sensor_rmse(f, obs_f, fnames, mask), obs_f, fnames),
            "tank_level": (sensor_rmse(l, obs_l, lnames, mask), obs_l, lnames),
        }
        per_sensor = {k: v for d, _, _ in pieces.values() for k, v in d.items()}
        # Equal weight per sensor type, then equal weight among sensors in type.
        by_type = {
            label: {
                "normalized_rmse": composite(scores, frame, names, mask),
                "mean_rmse": float(np.mean(list(scores.values()))) if scores else None,
                "sensor_count": len(scores),
            }
            for label, (scores, frame, names) in pieces.items() if scores
        }
        return (float(np.mean([v["normalized_rmse"] for v in by_type.values()])), per_sensor, by_type)

    cache: dict[tuple[float, float], float] = {}
    def objective(x: np.ndarray) -> float:
        key = tuple(np.round(x, 6))
        if key not in cache:
            cache[key] = evaluate(x, fit_mask)[0]
        return cache[key]

    # Bounds express deliberately narrow, physically plausible deviations from nominal.
    fit = differential_evolution(objective, [(0.90, 1.10), (0.90, 1.10)], seed=20261125,
                                 maxiter=args.maxiter, popsize=4, polish=True, workers=1)
    nominal_train, nominal_train_by, nominal_train_types = evaluate(np.array([1.0, 1.0]), fit_mask)
    nominal_valid, nominal_valid_by, nominal_valid_types = evaluate(np.array([1.0, 1.0]), valid_mask)
    fit_train, fit_train_by, fit_train_types = evaluate(fit.x, fit_mask)
    fit_valid, fit_valid_by, fit_valid_types = evaluate(fit.x, valid_mask)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibrated_inp = args.out_dir / "ltown_battledim_calibrated.inp"
    calibrated_wn = wntr.network.WaterNetworkModel(str(inp))
    for _, pipe in calibrated_wn.pipes():
        pipe.roughness *= float(fit.x[1])
    for _, junction in calibrated_wn.junctions():
        for demand in junction.demand_timeseries_list:
            demand.base_value *= float(fit.x[0])
    wntr.network.write_inpfile(calibrated_wn, str(calibrated_inp), units="LPS", version=2.2)
    normalize_inp(calibrated_inp)
    calibrated_sha = hashlib.sha256(calibrated_inp.read_bytes()).hexdigest()
    calibrated_prv_dir = args.out_dir / "ltown_prv_calibrated"
    calibrated_prv_inp = build_ltown_prv(calibrated_prv_dir, source_model=calibrated_inp)
    calibrated_prv_meta = calibrated_prv_dir / "ltown_prv.meta.json"
    normalize_inp(calibrated_prv_inp)
    prv_meta = json.loads(calibrated_prv_meta.read_text(encoding="utf-8"))
    prv_meta["inp_sha256"] = hashlib.sha256(calibrated_prv_inp.read_bytes()).hexdigest()
    calibrated_prv_meta.write_text(json.dumps(prv_meta, indent=2), encoding="utf-8")
    report = {
        "dataset": "BattLeDIM Zenodo record 4017659 (2018 historical SCADA)",
        "dataset_files_md5": ZENODO_FILES,
        "source_model": "L-TOWN.inp (nominal model)",
        "source_model_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
        "calibrated_model": calibrated_inp.name,
        "calibrated_model_sha256": calibrated_sha,
        "calibrated_optimizer_network": calibrated_prv_inp.relative_to(args.out_dir).as_posix(),
        "calibrated_optimizer_metadata": calibrated_prv_meta.relative_to(args.out_dir).as_posix(),
        "calibrated_optimizer_network_sha256": hashlib.sha256(calibrated_prv_inp.read_bytes()).hexdigest(),
        "wntr_version": wntr.__version__, "epanet_version": "2.2",
        "model_pattern_timestep_seconds": pattern_timestep_seconds,
        "time_window": {"start": str(START), "end_exclusive": str(END), "fit_end": str(FIT_END)},
        "hourly_samples": int(len(obs_p)),
        "scada_units": {"pressure": "m", "flow_input": "m3/h (converted to L/s)",
                        "wntr_flow_output": "m3/s (converted to L/s)", "tank_level": "m"},
        "fit_parameters": {"demand_multiplier": float(fit.x[0]), "global_pipe_roughness_multiplier": float(fit.x[1])},
        "normalized_rmse_equal_sensor_type": {
            "nominal": {"fit": nominal_train, "heldout": nominal_valid},
            "calibrated": {"fit": fit_train, "heldout": fit_valid},
        },
        "rmse_by_sensor_type": {
            "nominal_fit": nominal_train_types, "nominal_heldout": nominal_valid_types,
            "calibrated_fit": fit_train_types, "calibrated_heldout": fit_valid_types,
        },
        "rmse_by_sensor": {
            "nominal_fit": nominal_train_by, "calibrated_fit": fit_train_by,
            "nominal_heldout": nominal_valid_by, "calibrated_heldout": fit_valid_by,
        },
        "optimizer": {"name": "scipy differential_evolution", "seed": 20261125,
                       "maxiter": args.maxiter, "iterations": int(fit.nit),
                       "converged": bool(fit.success), "termination_message": str(fit.message),
                       "evaluations": int(fit.nfev)},
        "interpretation": "Initial two-parameter fit on the pre-first-reported-leak Jan 1-5 interval; undocumented leaks cannot be ruled out. Jan 6-7 is held out. Inspect per-sensor-type metrics because the aggregate can mask group-specific residuals. Not evidence for unseen control actions.",
    }
    (args.out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["normalized_rmse_equal_sensor_type"], indent=2))
    print(f"Parameters: demand={fit.x[0]:.5f}, roughness={fit.x[1]:.5f}; report={args.out_dir / 'calibration_report.json'}")


if __name__ == "__main__":
    main()
