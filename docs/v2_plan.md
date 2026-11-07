# TLE-WDN extensions — design (shift-aware gate, pooled learning, trajectory monitoring)

The core experiments measured three concrete failures of decide-once transfer with a fixed calibration set:

1. where harm is common (L-Town, 17% harmful pairs) the gate cannot certify α ≤ 0.20 with 30 calibration targets
   and abstains everywhere, although a ranking alone captures 72% of the oracle;
2. under layout shift (E3, three closed pipes) the frozen gate keeps transferring and realised harm reaches 23% > α;
3. the decision is taken once, before any target evaluation, although the warm run's own trajectory reveals early
   whether the injected elites are leading the search astray.

The extensions add a method for each, evaluated exactly on the logged runs (no new simulations needed for 1–3).

## V2-A Trajectory-monitored transfer (the TLE core)

Two-stage policy: stage 1 picks the source (zero-shot ranking, optionally gated); stage 2 monitors the warm run at
checkpoints g ∈ {1, 2, 4, 8} generations and either continues or aborts to a fresh restart.

* Exact offline evaluation: under common random numbers a fresh restart after c = N(1 + g) spent evaluations equals
  the fresh run truncated at B − c; its final quality is read from the fresh best-so-far curve.
* Trajectory features at g: warm-run progress, diversity decay, feasible share, improvement rate; and the *gap to
  the expected fresh trajectory*: log(best_warm(g)) − log(best_rest0 · r̂_fresh(g)), where best_rest0 is the best of the
  N − m random generation-0 members shared with the fresh run and r̂_fresh(g) is a regressor of fresh progress
  ratios trained on training targets' fresh runs (scenario features only).
* Label: continuing is harmful if (Q_abort − Q_warm)/σ_fresh < −ε (repeat level).
* Risk control: Learn-then-Test on calibration targets (one repeat per target) bounds the harm rate among runs that
  are allowed to continue; abort costs are counted.

## V2-B Shift-aware gate (fails safe under shift)

A domain classifier separates calibration-target features from the new targets' features (unlabelled, transductive).
Its density-ratio weights w = p(new|x)/p(cal|x) re-weight the calibration harm labels; the weighted harm rate is
tested with the effective sample size n_eff = (Σw)²/Σw² (Kish). When the new targets are far from calibration,
n_eff collapses and the gate abstains instead of silently transferring. The binomial test on n_eff is an
approximation (not an exact finite-sample guarantee) and is reported as such.

## V2-C Pooled cross-network learning and calibration

Network-normalised features allow pooling Net3 and L-Town training pairs (predictor) and calibration targets
(gate, with the V2-B weights correcting for the network shift). Question: does pooling make certification possible
on L-Town with few local calibration targets, and does the pooled predictor beat the L-Town-only one?

## Benchmarks

E1 (Net3, harm rare) is the control; L-Town (E4 + 40 extra calibration targets, harm ~17%) and E3 (layout shift) are
the settings where the decision layer must earn its keep.

## Success criteria (stated before running)

* V2-A: on L-Town, realised utility of monitored transfer > zero-shot ranking alone, with repeat-level harm among
  continued runs ≤ α in ≥ 90% of re-splits.
* V2-B: on E3 level 3, realised harm among accepted ≤ α (by abstaining), at a stated utility cost.
* V2-C: L-Town gate certifies at α = 0.20 with pooled calibration, or the failure is quantified.
