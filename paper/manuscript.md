# When does reusing a past optimisation run hurt? Trajectory-monitored, risk-controlled warm-start decisions for leak–pressure management in water distribution networks

*Draft manuscript (preprint version). Author list, affiliations and AI-assistance statement to be completed by the
authors. Every number in Sections 6–7 is taken from the result files named in brackets; regenerate with
`scripts/reproduce.py`.*

## Abstract

Water utilities choose pump speeds and pressure-reducing-valve (PRV) settings after a leak with evolutionary
optimisers that call a hydraulic simulator thousands of times. Leaks recur with variations, so every solved case
leaves a run behind, and a new search can be warm-started from the best solutions of an old run. Whether this
helps or hurts is unknown beforehand. We formulate the choice as a decision problem — warm-start from a named past
run or start fresh — with an operator-chosen cap α on the rate of harmful transfers, and we build a decision layer
in front of an unchanged optimiser: a predictor of the signed, matched-budget utility of a named warm start, a
Learn-then-Test gate with a finite-sample harm bound, and a trajectory monitor that can abort a warm start after a
few generations. On 2 460 paired source–target cases from two benchmark networks (Net3 with two PRV zones; the
782-junction L-Town), measured with common random numbers, we find that (i) the utility of a named warm start is
predictable before the run (Spearman 0.58, 0.67 with a probe), while the widely used "similar means useful" rule is
not (0.05); (ii) on Net3 with Differential Evolution warm starts almost always help, and the gate holds its harm bound
in every re-split, but the same pairs optimised with a genetic algorithm are harmful three times as often, and
pair utilities do not transfer across optimisers; (iii) the harm guarantee breaks under layout change (realised harm
23% at α = 0.20 after three pipe closures) and certification becomes impossible on a held-out network with
30 calibration cases. We report these failures as findings and test three remedies: trajectory monitoring,
shift-aware calibration and pooled cross-network learning. Code, scenarios and the paired-run benchmark are released.

## 1 Introduction

*(to be finalised after the v2 results; outline)*
1. Recurring leaks; evolutionary pump/PRV optimisation; each case leaves a run.
2. Sequential transfer optimisation reuses runs (SEETO, BCKT, PMTO); negative transfer is acknowledged but there is
   no decision with a stated risk.
3. Contributions C1–C5 (Section 1.4 of the build guide), stated against "not found in the literature searched".

## 2 Related work

Sequential transfer optimisation: solution transfer (Xue et al., TEVC 2024), similarity-based source selection
(SEETO, Fang et al. 2026; MGAD 2025), Bayesian competitive transfer (BCKT, Lu et al. 2025), parametric multi-task
optimisation (PMTO, Wei et al. 2025), learned evolutionary models (OKAEM 2025). Trajectory-based algorithm
selection (Jankovic et al. 2022; Renau and Hart 2024). Risk control: Learn-then-Test (Angelopoulos et al. 2021),
conformal risk control (2024), weighted conformal prediction under covariate shift (Tibshirani et al. 2019),
selective prediction with risk control (Bai and Jin 2026). WDN optimisation: pump scheduling with deep RL (Pei et al.
2025), feasibility-guided evolutionary pump optimisation (2025), BattLeDIM / L-Town (2020).

## 3 Problem formulation

A past run j stores its scenario θ_j, trajectory τ_j, elites E_j and hydraulic signature h_j. For a new scenario θ*
with budget B the actions are a_0 (fresh) and a_j (inject the top-k elites of run j). Quality Q_B(a) is the best
penalised objective after exactly B evaluations. Utility U_j = (mean Q_B(a_0) − mean Q_B(a_j)) / σ_fresh over R
matched repeats; harm = 1[U_j < −ε]. A policy maximises E[U] subject to Pr(harm | transfer) ≤ α.

## 4 Method

4.1 Simulator and objective: EPANET 2.2, pressure-driven demand, orifice leak (C_d A √(2gp)); J = leak volume +
energy cost (time-of-use tariff) + service-pressure shortfall + storage deficit, weights by a frozen normalisation rule.
4.2 Optimiser and action: DE/rand/1/bin (F 0.4, CR 0.5, N 20) with exact budget; top-3 distinct elites with 3% noise
injected into 40% of the initial population; common random numbers between fresh and warm runs.
4.3 Features: scenario facts (a), source trajectory (b), hydraulic signatures with one reference evaluation (c),
probe of the injected elites (d); all available before the decision.
4.4 Predictor: gradient-boosted quantile regression (10%, mean, 90%) and a calibrated harm classifier.
4.5 Gate: the top source by the 10% utility quantile is accepted if p̂_harm ≤ λ; λ by fixed-sequence Learn-then-Test
starting at the first threshold that can be certified (n ≥ ⌈ln δ / ln(1−α)⌉).
4.6 Trajectory monitor (TLE): at generation g the warm run's own progress is compared with the expected fresh
progress (anchored on the random generation-0 members shared with the fresh run); abort → fresh restart with the
remaining budget; λ_g by Learn-then-Test; outcomes evaluated exactly from logged runs.
4.7 Shift-aware gate: density-ratio weights from a domain classifier between calibration and new targets; weighted
harm rate tested at an effective sample size min(Kish, Σw), so the gate abstains when overlap is poor.

## 5 Experimental design

E1 Net3-PRV (180 targets, S = 6, R = 8/10, B = 400), E2 goal change, E3 layout change (1–3 closed pipes), E4 L-Town
(B = 1000 by the same regret rule; 100 + 40 targets, S = 3, R = 4); baselines B0–B7 and oracle; statistics over
targets (Wilcoxon–Holm, Friedman, bootstrap, Clopper–Pearson); pre-registration with dated amendments.

## 6 Results

### 6.1 The paired-run benchmark (C1) [results/*/reports/phase3_quality.json]

On Net3 with DE (E1, 1 080 pairs, 180 targets) 80.4% of named warm starts help (U > 0.5), 16.8% are neutral and
2.9% harm; harm grows with scenario distance (near 1.7%, far 4.7%). The label noise is small relative to the spread
of utilities (median standard error / spread = 0.32). On L-Town (E4, 360 pairs, 120 targets) the picture changes:
61.7% helpful, 22.2% neutral, 16.1% harmful.

### 6.2 Utility is predictable; similarity is not (RQ1, C3) [results/e1/phase4/]

Out-of-fold Spearman correlation between predicted and measured utility on training targets: scenario facts (M2)
0.51 [0.34, 0.64]; plus trajectory and hydraulic-signature tiers (M5) 0.58 [0.41, 0.69], a gain of +0.070
[0.003, 0.148]; plus the generation-0 probe (M6) 0.67 [0.56, 0.75]. The SEETO-style hydraulic-state similarity alone
reaches 0.05 [−0.07, 0.17], scenario-space distance −0.07. Permuting the hydraulic-signature tier inside M5 costs
0.18 Spearman, the trajectory tier 0.005, and shuffling generation order changes nothing: the value lies in physical
signatures, not in search histories. Permuting labels across targets drops performance to 0.10 (leakage audit).

### 6.3 Decisions on the same network (RQ2, RQ3) [results/e1/phase5, phase6]

Learn-then-Test keeps the realised harm rate among accepted transfers at or below α in 100% of 200 random
calibration/test re-splits for α ≥ 0.10 (95.5% at α = 0.05). On the untouched test targets the gate (α = 0.20)
transfers on 39 of 40 targets with no harmful transfer (Clopper–Pearson upper bound 9.0%), mean utility 1.74, 87.5%
of the oracle. It significantly exceeds always-nearest transfer (1.54), the PMTO-style task model (1.49) and
BCKT-style competition (1.02) (Holm-corrected Wilcoxon p ≤ 0.039), but is statistically indistinguishable from
SEETO-style multi-source injection (1.75) and similarity top-1 (1.68). With harm this rare the gate's abstention
option almost never matters.

### 6.4 Robustness: optimiser, goal, layout, network [results/e1/ga_check, e2, e3, e4x]

*Optimiser.* On 120 identical pairs, a genetic algorithm instead of DE lowers the mean utility from 1.51 to 0.44 and
raises the harmful share from 1.7% to 13.3%; pair utilities under the two optimisers are uncorrelated (Spearman
0.08). Whether a past run helps is a property of the (source, target, optimiser) triple.
*Goal change (E2).* The frozen predictor keeps its accuracy (0.61–0.63) and the frozen gate stays within α.
*Layout change (E3).* Harm rises with the number of closed pipes (8.3%, 13.3%, 17.8%), predictor accuracy falls
(0.57, 0.28, 0.12) and the frozen gate's realised harm reaches 23.3% > α = 0.20 at three closures: the
exchangeability assumption fails and so does the guarantee.
*New network (E4).* The Net3 predictor does not transfer (0.08). With 30 or even 70 L-Town calibration targets, no
gate certifies α ≤ 0.20 (the harm rate among the best-ranked candidates is too close to α), so the certified gate
abstains everywhere, while an L-Town-trained ranking without a gate captures 72% of the oracle.

### 6.5 Remedies (v2) [results/v2/V2_RESULTS.md]

*Shift-aware gate.* Re-weighting the calibration harm labels by domain-classifier density ratios, with an
effective sample size min(Kish, Σw), removes the E3 violation (realised harm 23% → 0%, by abstaining), leaves the
unshifted E1 decisions unchanged at α = 0.20, and refuses to transfer a Net3 gate to L-Town. Its cost is visible on a
detectable but harmless shift (E2, stricter pressure requirement), where it abstains and forgoes a utility of 1.38.
*Pooled calibration.* Calibrating on Net3 and L-Town together "certifies" L-Town only by borrowing Net3's low harm
rate; the shift-aware variant rejects it.
*Trajectory monitoring.* Watching the warm run for 1–8 generations and aborting to a fresh restart did not help: the
risk-controlled monitor cannot certify on L-Town and aborts everything, and even a monitor tuned for utility without a
guarantee converges to "never abort" (L-Town 1.61 vs 1.61; Net3 1.77 vs 1.77). Early progress does not separate
harmful warm starts well enough to repay the restart cost.

### 6.6 Cost [results/e1/reports/amortisation.json]

The gate saves 233 evaluations per task (B = 400), whereas labelling the training pairs cost 1.34 M evaluations:
break-even after about 5 800 future tasks. With millisecond simulations the labelled library does not pay for itself.

## 7 Discussion and limitations

Three messages survive. First, the benefit of a named warm start is predictable from physical information available
before the run, while similarity — the selection rule most transfer methods rely on — carries almost no information
about it. Second, a finite-sample harm bound is easy to hold where harm is rare and impossible to certify where it is
common unless the harm predictor is strong; on L-Town the bottleneck is harm prediction (AUROC ≈ 0.58), not the
decision rule. Third, exchangeability is the weak point: layout changes break the guarantee, and a shift-aware gate
restores safety only by abstaining. Limitations: two benchmark networks; cheap simulations; labels from 4–10
repeats; baselines are re-implementations (the official BCKT code was not run); the GA result shows that findings
must be stated per optimiser; the shift-aware test is an approximation, not an exact finite-sample guarantee.

## Data and code availability

Repository with pinned environment, scenario manifests, paired-run tables and scripts; dataset DOI to be added.
