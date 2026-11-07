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

### E1 campaign
Library 40 source runs, 220 reference evaluations (+ attainability runs of 5B), 1,520 (target, repeat) jobs =
10,640 DE runs. Partial look at 114 complete pairs (not used for any decision): mean U 1.62, harm share 1.8% —
a first sign of the build guide's "transfer almost always helps" branch.
