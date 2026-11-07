# Pre-registration — TLE-WDN

Frozen settings of the study. Values change only through a dated amendment at the end of this file.
Written on 2026-11-07, before any E1 label, model or gate result was computed. The values in
`configs/` and `configs/generated/` are the machine-readable source; this file explains them.

## 1. Networks and roles

| Stage | Network | File | Role |
|---|---|---|---|
| E0 | Net3-PRV | `data/networks/net3_prv.inp` | unit tests and debugging |
| E1–E3 | Net3-PRV | `data/networks/net3_prv.inp` (sha256 in `net3_prv.meta.json`) | main benchmark, goal change, layout change |
| E4 | L-Town-PRV (BattLeDIM, EUPL-1.2) | `data/networks/ltown_prv.inp` (sha256 in `ltown_prv.meta.json`) | held-out network |

Net3-PRV edits (`src/tle/hydraulics/network_prep.py`): all controls removed; two PRV zones created on 2-pipe cuts
(PRV_A on pipe 238 with pipe 289 closed, 20 junctions; PRV_B on pipe 135 with pipe 161 closed, 9 junctions);
pump-335 bypass pipe 330 closed; 24 h horizon with 1 h steps; pressure-driven demand (p_min 0 m, p_req 20 m,
exponent 0.5); SI units. Bulk-supply offtakes 15, 35, 123, 203 (industrial patterns at dead ends) are excluded from
the service-pressure term (they keep pressure-driven demand).

Simulator: EPANET 2.2 toolkit bundled with WNTR 1.5.0, called through `src/tle/hydraulics/epanet.py` and the
compiled loop `kernel.py`; verified against `wntr.sim.EpanetSimulator` (max pressure difference < 1e-3 m, test
`test_matches_wntr_epanet_simulator`). One simulator for the whole study.

## 2. Decision variables and objective

* W = 6 windows of 4 h; x = pump speed multipliers in [0.4, 1.0] and PRV outlet settings (Net3 [15, 45] m,
  L-Town [25, 60] m) per window; D = 24 on both networks.
* J = w_L·leak volume [m³] + w_E·energy cost + w_P·pressure shortfall [m·h] + w_T·system storage deficit [m³].
  Energy cost uses a time-of-use tariff (0.10 off-peak 0–7 h, 23–24 h; 0.20 otherwise).
* Leak: EPANET emitter with coefficient C_d·A·√(2g), C_d = 0.75 (orifice law, pressure-dependent).
* Weight rule (`src/tle/hydraulics/weights.py`): 200 random decisions on the centre scenario; w_L, w_E = 1/median;
  penalty weights = 2R / q10 of violations above the feasibility tolerance (10× tolerance when fewer than 10 random
  violations exist). Frozen values: `configs/generated/net3_weights.yaml`, `configs/generated/ltown_weights.yaml`.
* Feasible: shortfall ≤ 0.001 m·h and storage deficit ≤ 1 m³.

## 3. Optimiser and budget

DE/rand/1/bin, own implementation, reflection at bounds, uniform initial population. Grid F ∈ {0.4, 0.6, 0.8},
CR ∈ {0.5, 0.9}, N ∈ {20, 40} on 6 training scenarios × 5 seeds at B = 1000; frozen **F = 0.4, CR = 0.5, N = 20**.
Budget rule: median remaining regret r(B) = (Q_B − Q_5B)/(Q_init − Q_5B) in [3%, 20%], closest to the log-middle;
frozen **B = 400** (median r = 9.5%). Evidence: `configs/generated/de_tuning.yaml`.

## 4. Transfer action

Top-k distinct elites of the source run's archive (feasible first, min normalised RMS distance 0.05), injected into
the first round(ρN) slots of the target's initial population with Gaussian noise σ = 3% of each range, clipped.
Grid k ∈ {1, 3, 5}, ρ ∈ {0.1, 0.2, 0.4} on 14 constructed pilot pairs (training targets), criterion = highest mean
utility; frozen **k = 3, ρ = 0.4** (m = 8 injected of N = 20). Evidence: `results/e1/pilot/kr_grid.csv`.

## 5. Definitions

* Utility U = mean over R matched repeats of (Q_fresh − Q_warm) / σ_fresh; σ_fresh = seed std of fresh final J,
  floored at 1% of |mean|. Harm = 1[U < −ε], **ε = 0.5**.
* Anytime utility, evaluations to target E (Q* = median fresh best-so-far at 0.8 B; censored at B), transfer gain
  TG = (E_fresh − E_warm)/E_fresh; regret to oracle; share of oracle utility captured.

## 6. Scenarios and splits

Factors: leak node (zone-stratified: main 50%, zone A 30%, zone B 20%), leak area log-uniform [2e-4, 3e-3] m²,
demand multiplier [0.8, 1.2], peak shift −3…3 h, pattern amplitude [0.8, 1.2], initial tank level [0.3, 0.8].
Scrambled Sobol sampling, master seed 20261107. Splits (full configuration): library 40, train 60,
calibration 80, test 40, stratified by zone. Manifests and hashes: `data/scenarios/e1_*`.

## 7. Pair design

S = 6 sources per target (2 near, 2 mid, 2 far tertiles of scenario distance in the library); R = 8 matched repeats
(test targets R = 10); common random numbers (shared initial draw and DE stream).

## 8. Feature tiers

a scenario facts, b source trajectory, c hydraulic signatures (incl. one reference evaluation of the target),
d probe (generation-0 values of injected elites). Only information available before the decision; every column is
listed with `available_before_decision` in `docs/data_dictionary.md`.

## 9. Models and validation

Gradient-boosted trees (LightGBM; grid num_leaves {4, 8}, min_child_samples {10, 30}, n_estimators {150, 400},
learning rate 0.03, L2 5) for mean utility, 10%/90% quantiles and harm probability (class-balanced, Platt
recalibration on grouped out-of-fold predictions); ridge / logistic guards. Grouped 5-fold CV over training
targets; calibration and test targets untouched. Models M0–M6 as in the build guide; M1 = SEETO-style state cosine.

## 10. Gate

Rank candidate sources by the 10% utility quantile; accept the top source if p̂_harm ≤ λ. λ by Learn-then-Test
fixed-sequence binomial testing on calibration targets (one pair per target), **α ∈ {0.10, 0.20}** (0.20 main),
**δ = 0.05**. Main reported quantity: harm rate among accepted transfers (conditional version). Probe mode:
q ∈ {1, 2, 4, 8} probed elites; headline B7 uses q = 8 (pre-specified).

## 11. Baselines and fairness

B0 fresh, B1 nearest scenario, B2 SEETO-style similarity (top-1 look-up; B2γ multi-source run), B3 BCKT-style,
B4 PMTO-style, B5 historical bins, B6 proposed zero-shot gate, B7 probe gate, oracle. Same DE, B, N, simulator and
seeds for all; adaptations in `docs/baselines.md`.

## 12. Hypotheses

* H1 (RQ1): out-of-fold Spearman ρ(û, U) of M5 > M2 and > M1 on training targets (bootstrap CI over targets of the
  difference excludes 0).
* H2 (RQ2): realised harm rate among accepted ≤ α in ≥ 90% of 200 random calibration/test re-splits.
* H3 (RQ3): B6 mean realised utility on test targets > B1, B2, B4, B5 (paired Wilcoxon, Holm-corrected, 5%).
* H4 (RQ4): harm-rate control degrades with layout shift level (E3) and is restored by recalibration (E2/E4).

## 13. Statistics

Unit = target scenario (repeats averaged first). Paired Wilcoxon signed-rank with Holm correction; Friedman over
methods; rank-biserial correlation and Cliff's δ; bootstrap CIs over targets (10 000 resamples for headline numbers,
2 000 for ablations); Clopper–Pearson intervals for harm rates.

## 14. Stopping rules, branch decisions, amendments

Gate failures are reported as findings (build guide, Appendix F). C3 is kept only if tiers b/c improve on tier a
with a CI excluding 0.

### Amendment log

| ID | Date | Change | Reason |
|---|---|---|---|
| A-001 | 2026-11-07 | The Learn-then-Test sequence starts at the first threshold accepting ≥ ⌈ln δ / ln(1−α)⌉ calibration points | With the guide's skeleton the strictest threshold accepts one point, its p-value (1−α) > δ stops the fixed sequence and the gate never transfers. Thresholds that cannot be certified even with zero harms are skipped; the order still depends only on p̂, never on labels, so the guarantee is unchanged. |
| A-002 | 2026-11-07 | Python 3.12 instead of 3.11 | Installed interpreter; all packages pinned in `requirements-lock.txt`. |
| A-003 | 2026-11-07 | Storage balance on total system storage instead of per tank | A large leak next to tank 2 drains it whatever the pumps do; operators state the daily balance for total storage. |
| A-004 | 2026-11-07 | Scenario ranges narrowed (leak area ≤ 3e-3 m², demand and amplitude [0.8, 1.2]) | With the guide's wider example ranges about 15% of scenarios had no feasible operation (zone-B bursts, extreme peaks). |
