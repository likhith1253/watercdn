# Project plan

Source of truth: *TLE-WDN — Risk-controlled transfer decisions, Project build guide v1.0* (7 Nov 2026), phases 0–9.
The project is one research track (not separate papers); the three base papers (SEETO, BCKT, PMTO) define the
rival answers and become baselines B2/B3/B4. Reading notes: `docs/reading/`.

## Research questions → evidence

| RQ | Question | Evidence produced by |
|---|---|---|
| RQ1 | Can the signed utility of a named warm start be predicted; do trajectory/hydraulic tiers beat scenario parameters and similarity? | `scripts/phase4_models.py` → T3, F3 |
| RQ2 | Does the calibrated gate keep harm ≤ α on new scenarios; how far under layout shift? | `phase5_gate.py` (F4, F5), `phase7_shift.py` (E2, E3, E4) |
| RQ3 | At equal budget, does predict-then-decide beat SEETO-, BCKT-, PMTO-style transfer, counting library cost? | `phase6_baselines.py` (T2, F7), `phase8_report.py` (F9) |
| RQ4 | How does benefit change with leak location, size, demand, layout? | pilot map F2, pair table, E3 shift levels |

## Contributions → status (see IMPLEMENTATION_STATUS.md for verified outcomes)

C1 paired matched-budget utility dataset · C2 risk-controlled gate with harm bound · C3 hydraulic signatures as
predictors · C4 zero-shot vs probe · C5 open benchmark + break-even.

## Phase checklist

| Phase | Deliverable | Status |
|---|---|---|
| P0 | repo layout, pinned env (`requirements-lock.txt`, `pyproject.toml`), reading notes, pre-registration | done |
| P1 | simulator, objective, weights rule, DE, tuning, budget rule, tests | done |
| P2 | transfer action, library record, paired runner, sanity tests T1–T6, pilot map, k/ρ grid | done (harmful-pair part of the gate: see D-008) |
| P3 | scenarios, splits, pairs, resumable engine, labels, features, data dictionary, quality report | campaign running → see status |
| P4 | predictors, tier ablation, order test, leakage audit, importance | code done; results pending campaign |
| P5 | gate, re-splits, trade-off, operator check, probe mode, final test evaluation | code done; results pending |
| P6 | baselines B0–B7 + oracle, adaptation table | code done; results pending |
| P7 | E2 goal, E3 layout, E4 L-Town (zero-shot, recalibrated, retrained), sensitivity | code done; runs pending |
| P8 | statistics, figures, amortisation, release package, cross-check | code done; pending |
| P9 | manuscript | out of scope of the autonomous build (results and figures are prepared for it) |

## Compute plan (16-thread laptop, ~1.5–4 ms per Net3 evaluation idle, 2–4× slower under full load)

E1 full configuration: 180 targets × R (8; test 10) × 7 runs × B 400 ≈ 4.3 M evaluations (+ 220 reference runs of
2 000). E2: 80 targets, E3: 90 targets (R 8, S 6). E4 on L-Town (~33 ms/evaluation): 80 targets, R 4, S 6.
