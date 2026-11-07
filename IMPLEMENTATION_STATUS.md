# Implementation status

Last updated 2026-11-07. "Verified" means executed end to end with the outputs named; numbers come from those files.

## Components

| Component | State | Verification |
|---|---|---|
| EPANET 2.2 binding + compiled loop, scenario configuration, objective | complete | 15 tests; equals WNTR EpanetSimulator (< 1e-5 m); emitter flow equals orifice law; history-independent |
| Net3-PRV and L-Town-PRV network builders | complete | metadata with hashes in `data/networks/*.meta.json` |
| Weight rule, DE/rand/1/bin, exact budget, logging | complete | budget, determinism, failure handling tests; tuning and regret rule run |
| Transfer action, library, paired runner (common random numbers) | complete | sanity tests T1–T5; T6 = bit-identical pilot re-run after code change and sensitivity re-runs |
| Scenario sampler, splits, pair manifest, resumable engine | complete | E1 (1 080 pairs), E2 (480), E3 (540), split audit passed |
| Labels, tables, features (tiers a–d), data dictionary | complete | `results/*/tables`, `docs/data_dictionary.md` |
| Predictors, ablation M0–M6, order test, leakage audit | complete | `results/e1/phase4/` |
| Learn-then-Test gate, re-splits, operator check, probe mode | complete | `results/e1/phase5/`; 4 gate tests |
| Baselines B0–B7 + oracle (SEETO-, BCKT-, PMTO-style) | complete | `results/e1/phase6/` (400 run-based baseline runs) |
| E2 goal change, E3 layout change | complete | `results/e2/phase7`, `results/e3/phase7` |
| E4 held-out L-Town | complete (lean design, D-013) | `results/e4/` (240 pairs; frozen, recalibrated, retrained gates) |
| Sensitivity (ε, k, ρ, B), amortisation, report | complete | `results/e1/sensitivity`, `results/e1/reports`, `RESULTS.md`; 700/700 cross-check runs identical |
| Manuscript (Phase 9) | not started (out of scope of the build) | figures and tables are generated for it |

## Phase gates

| Gate | Outcome |
|---|---|
| P0 environment + notes + pre-registration | passed |
| P1 reproducibility, regret 3–20%, feasibility ≥ 95% | reproducibility and regret (9.5%) passed; fresh feasibility 93.4% on attainable targets (12% of targets have no feasible operation) — slightly below 95%, documented (D-006, D-011) |
| P2 T1–T6, one clearly helpful and one clearly harmful pair | tests pass; 12/14 pilot pairs clearly helpful, **no clearly harmful pilot pair** (D-008) |
| P3 helpful and harmful ≥ 15% each | **failed for harm: 2.9% harmful** (helpful 80.4%) — "transfer almost always helps" branch (D-011) |
| P4 leakage audit; tier comparison with CIs | passed; C3 kept in narrowed form (D-012) |
| P5 harm ≤ α in ≥ 90% of re-splits; gate sometimes transfers | passed (100% at α ≥ 0.10; 95.5% at α = 0.05), but trivially because harm is rare |
| P6 every baseline end to end, adaptation log | passed (`docs/baselines.md`) |
| P7 E2–E4 ran with CIs | passed: E2, E3, E4 and sensitivity ran |
| P8 headline table reproduces; numbers traceable | `RESULTS.md` generated from result files; cross-check identical; a run on a second machine was not possible here |

## Hypotheses (pre-registered) — outcome

| H | Outcome |
|---|---|
| H1 M5 > M2 and > M1 (Spearman) | supported: +0.070 [0.003, 0.148] and +0.53 [0.36, 0.68] |
| H2 harm ≤ α in ≥ 90% of re-splits | supported on E1 (100% for α ≥ 0.10), but with only 2.9% harmful pairs |
| H3 B6 > B1, B2, B4, B5 | partly: > B1, B4, B3 (Holm p ≤ 0.039); not distinguishable from B2, B2γ, B5 |
| H4 control degrades with layout shift, restored by recalibration | degradation supported (E3: realised harm 23% > α at 3 closures; E4 predictor ρ 0.08); restoration **not** supported — with ≤ 30 new calibration targets the recalibrated gate cannot certify and abstains |

## Contributions — what the data support

* C1 paired matched-budget utility dataset: delivered (E1–E4, 2 340 pairs, released tables).
* C2 risk-controlled gate: valid where harm is rare (E1, E2); breaks under layout shift (E3); certification
  impossible with 30 calibration targets when harm is ~17% (E4).
* C3 hydraulic signatures: narrowed — tier c adds modest predictive value; trajectory summaries do not.
* C4 zero-shot vs probe: the probe improves prediction (M6 − M5 +0.09) but changes no E1 decision.
* C5 benchmark + break-even: benchmark released; the labelled library does not amortise in this cheap-simulation
  setting (break-even ~5 800 tasks).

## Known limitations

* On Net3 with this action, warm-starting is beneficial in ~80% of pairs and harmful in ~3%, so the harm bound is
  easy to satisfy and the abstain option rarely matters; most of the value comes from choosing the source.
* Harm-probability models are weak (≤ 5 harmful training pairs); the utility ranking carries the decisions.
* The guarantee is relative to the finite-repeat harm label (R = 8–10) and to exchangeability; under layout shift
  (E3, 3 closed pipes) it fails (realised harm 23% > α = 0.20).
* 12% of E1 targets and 57% of E3 targets have no feasible operation; their utilities are driven by penalty terms.
* Single optimiser (DE); the optional GA robustness check, OKAEM and M-GRU were not run (order test negative).
* Baselines are re-implementations; the official BCKT MATLAB code was not run.
* Simulations are cheap (ms), so savings are reported in evaluations, not wall-clock of expensive simulators.
* Raw job files of an early commit range were committed by mistake (wrong ignore pattern) and later untracked;
  they remain in git history because history is not rewritten.
