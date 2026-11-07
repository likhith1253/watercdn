# Decision log

Dated record of decisions and branch outcomes. Numbers quoted here come from the files named next to them.

| ID | Date | Decision | Evidence / reason |
|---|---|---|---|
| D-001 | 2026-11-07 | Main network Net3 with two PRV zones built on 2-pipe cuts (PRV_A: pipe 238, close 289; PRV_B: pipe 135, close 161); close bypass pipe 330 | Net3 has no valves; a cut search (`network_prep.py`) found source-free zones that a single PRV can control |
| D-002 | 2026-11-07 | EPANET 2.2 toolkit through a compiled loop, not WNTRSimulator | 24 h run in ~1.5–4 ms vs ~0.1 s; identical pressures to WNTR's EpanetSimulator (< 1e-3 m). `EN_initH(EN_INITFLOW)` so a run never depends on the previous one |
| D-003 | 2026-11-07 | DE frozen at F 0.4, CR 0.5, N 20 | Tuning grid, `configs/generated/de_tuning.yaml` (best median Q/Q_best on 6 training scenarios × 5 seeds) |
| D-004 | 2026-11-07 | B = 400 | Median remaining regret 9.5% at 400 (band 3–20%); `configs/generated/de_tuning.yaml` |
| D-005 | 2026-11-07 | System storage balance; 4 bulk offtakes excluded from service pressure | Node 15 is below 20 m even at full pump speed in the original Net3; a leak next to tank 2 makes per-tank balance infeasible (amendment A-003) |
| D-006 | 2026-11-07 | Scenario ranges narrowed | Feasibility probe: 17% of scenarios infeasible with the wide ranges, 12% with narrow demand; leaks ≥ 2e-3 m² deep in zone B are physically unattainable (amendment A-004). Residual unattainable share is reported per experiment (`phase3_quality.json`) |
| D-007 | 2026-11-07 | k = 3, ρ = 0.4 | Highest mean utility over 14 pilot pairs (`results/e1/pilot/kr_grid.csv`) |
| D-008 | 2026-11-07 | Pilot Phase-2 gate only partly met | 12/14 pilot pairs clearly helpful (95% CI > 0); none clearly harmful at R = 8 (worst: large demand shift, U = −0.50, CI [−1.6, 0.5]). Harm is checked on the E1 training pairs instead (Phase 3 gate) |
| D-009 | 2026-11-07 | Learn-then-Test start rule (amendment A-001) | the guide's skeleton can never transfer (first p-value = 1 − α) |
| D-010 | 2026-11-07 | E4 on L-Town keeps D = 24 (1 pump + 3 PRVs × 6 windows), PRV range [25, 60] m | Settings below ~30 m starve demand nodes (constant-decision probe in the session log); B chosen with the E1 regret rule |
