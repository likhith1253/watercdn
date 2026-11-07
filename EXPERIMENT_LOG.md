# Experiment log

Chronological record of runs. Numbers are copied from the named result files (regenerate with the scripts).

## 2026-11-07

### Simulator checks (Phase 1)
* Net3-PRV pressures equal WNTR's EpanetSimulator to < 1e-5 m for the same decision (`tests/test_hydraulics.py`).
* EPANET's emitter outflow equals the orifice law C_d·A·√(2gp) used for the leak volume (test).
* Run-to-run dependence found and fixed: without `EN_INITFLOW` a simulation differed at ~1e-8 depending on the
  previously simulated decision.
* Python ctypes overhead (~1,400 calls per run) dominated; replaced by a numba loop (EPANET solve ≈ 3.5 ms per 24 h
  run, idle, at the clock the laptop ran then).
* Feasibility probe: node 15 (bulk offtake) below 20 m even at full pump speed → bulk nodes excluded from the
  service-pressure term; per-tank balance infeasible for leaks next to tank 2 → system storage balance; wide scenario
  ranges gave 17% infeasible scenarios → ranges narrowed (A-003, A-004).

### Weights (`configs/generated/net3_weights.yaml`)
w_leak = 6.56e-4 /m³, w_energy = 1.47e-3 per currency unit, w_shortfall = 7.22e-2 per m·h, w_tank = 0.227 per m³
(200 random decisions on the centre scenario; 0 of 200 random decisions feasible).

### DE tuning and budget (`configs/generated/de_tuning.yaml`)
Best setting F 0.4, CR 0.5, N 20 (median Q/Q_best 1.005). Median remaining regret r(B): 300 → 14.6%, 400 → 9.5%,
600 → 4.3%, 800 → 2.9%, 1000 → 2.7%. Chosen B = 400. One tuning scenario (e1_0086, zone-B leak, high demand) has no
feasible operation at any setting.

### Pilot (`results/e1/pilot/`)
k/ρ grid on 14 constructed pairs × 8 repeats: best mean utility k = 3, ρ = 0.4 (mean U 1.57, 1 pair with U < −ε).
Transfer map at (3, 0.4): 12/14 pairs clearly helpful (95% CI > 0); none clearly harmful; the worst pairs are
large demand shifts (U = −0.50, CI [−1.62, 0.48]) and fully mismatched sources (U ≈ 0). Re-running the pilot after
the simulator rewrite reproduced every number bit for bit.

### E1 campaign (`results/e1/reports/phase3_quality.json`)
Library 40 source runs, 220 reference evaluations (+ attainability runs of 5B), 1,520 (target, repeat) jobs =
10,640 DE runs, 0 failed simulations, 29,041 CPU-s of DE run time (33 min wall-clock on 15 workers).
1,080 pairs: helpful 80.4%, neutral 16.8%, harmful 2.9% (near 1.7%, mid 2.2%, far 4.7%); mean U 1.48, spread 1.15,
median SE 0.36. 12.2% of targets have no feasible operation within 5B; fresh feasibility on attainable targets
93.4%. Split audit passed. **Phase 3 gate failed for the harmful class (D-011).**

### Phase 4 (`results/e1/phase4/`)
OOF Spearman (95% CI over targets): M1 similarity 0.05 [−0.07, 0.17]; M2 scenario 0.51 [0.34, 0.64]; M3 0.58;
M4 0.57; M5 0.58 [0.41, 0.69]; M6 probe 0.67 [0.56, 0.75]; label-permuted M5 0.10. M5 − M2 +0.070 [0.003, 0.148];
M6 − M5 +0.088 [0.026, 0.165]; order test null. Tier importance in M5: a 0.39, c 0.18, b 0.005. Harm AUROC
uninformative (5 harmful training pairs). Decision D-012.

### Phase 5 (`results/e1/phase5/`; test targets first read 2026-11-07, time in `phase5_summary.json`)
Re-splits (200): harm ≤ α in 100% of splits for α ≥ 0.10, 95.5% at α = 0.05 (where the gate never transfers in 52%
of splits). Test: λ = 0.022 for α = 0.10 and 0.20; 39/40 transfers, 0 harmful (CP upper 9.0%), mean U 1.74,
87.5% of oracle. Probe sizes 1–8: identical decisions and utility.

### Phase 6 (`results/e1/phase6/t2_main_results.csv`)
Mean U on 40 test targets: oracle 1.99, M5 ungated 1.77, B2γ SEETO-style 1.75, B6 gate 1.74, B7 probe 1.74,
B2 similarity 1.68, B5 historical 1.61, B1 nearest 1.54, B4 PMTO-style 1.49, B3 BCKT-style 1.02, fresh 0.
Holm-corrected Wilcoxon vs B6: B1 p = 0.039, B4 p = 0.039, B3 p < 0.001, fresh p < 0.001; B2, B2γ, B5 n.s.
Friedman p = 1.5e-41. B3 wall-clock 142 s per run (GP refits) vs ~3–6 s for the others.

### E2 goal change (`results/e2/`)
480 pairs, harmful 3.3%. Frozen E1 predictor: Spearman 0.61 (G1, 25 m) and 0.63 (G2, energy ×2). Frozen λ: G1 1/19
harmful accepted (5.3%), 73.5% of oracle; G2 0/20, 76.2%. Recalibrated on 20 targets: α = 0.10 cannot certify
(needs ≥ 29 accepted) → never transfers.

### E3 layout change (`results/e3/`)
540 pairs; harmful share 8.3% / 13.3% / 17.8% for 1 / 2 / 3 closed pipes; frozen predictor Spearman 0.57 / 0.28 /
0.12; frozen gate (α = 0.20) realised harm 6.7% / 10% / 23.3%, oracle share 74% / 44% / 2%. 57% of E3 targets
unattainable. **The harm bound breaks at 3 closures.**

### Sensitivity (`results/e1/sensitivity/t4_sensitivity.csv`; 20 training targets × 6 sources × 5 repeats)
Mean U / harmful share: base 1.51 / 1.7%; k = 1 1.35 / 1.7%; k = 5 1.54 / 1.7%; ρ = 0.1 0.99 / 5.8%;
ρ = 0.2 1.29 / 4.2%; B = 200 2.38 / 2.5%; B = 600 1.23 / 4.2%; rank correlation with base 0.83–0.96.
ε = 0.25: harm 4.8%, gate abstains 85%, mean U 0.41; ε = 1.0: gate never transfers (D-014).
Cross-check: 700/700 re-run curves identical to the main campaign.

### Amortisation (`results/e1/reports/amortisation.json`)
233 evaluations saved per task; labelling cost 1.34 M (train) / 3.14 M (train + calibration) evaluations →
break-even after ~5,800 / ~13,500 future tasks.

### E4 L-Town
Budget rule chose B = 1000 (median regret 6.3%). Lean design S = 3, R = 4, attainability 2B (D-013).
