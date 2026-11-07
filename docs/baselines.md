# Baselines and adaptations (Table T1)

All methods share DE/rand/1/bin (F 0.4, CR 0.5, N 20), budget B, simulator and the (target, repeat) seeds of the
paired runs, so every method's random initial members and DE stream equal the fresh run's (common random numbers).
Look-up methods choose among the S = 6 measured candidate sources of a target or abstain; their outcome is read from
the pair table. Run-based methods (B2γ, B3, B4) are new DE runs on the test targets with the same seeds.
None of the reimplementations is the authors' code; each is the closest faithful variant we could build.

| ID | Method | Original design | What we implemented | Changes and why |
|---|---|---|---|---|
| B0 | Fresh DE | – | random initial population | – |
| B1 | Always transfer (nearest) | – | warm start from the candidate with the smallest scenario distance | stress test, not an upper bound |
| B2 | SEETO-style similarity, top-1 | Fang et al. 2026: autoencoder latent of the meteorological state, cosine similarity, top-γ = 5 sources, softmax weights (Γ = 0.065), 20% of the population injected, adaptive surrogate | state vector = hourly service-node pressures / p_req under a mid-range reference decision (1 evaluation); rank candidates by cosine similarity; same warm-start action from the top source | no autoencoder (the hydraulic state is low-dimensional and already physical); surrogate level left out; one evaluation of information cost reported separately |
| B2γ | SEETO-style multi-source | as above | top-5 candidates by cosine, softmax weights Γ = 0.065, round(ρN) slots allocated by largest remainder, each source's best distinct solutions copied as-is | candidates restricted to the target's S sources (same as every method); single objective, so "non-dominated + crowding" becomes best-J distinct selection; no surrogate |
| B3 | BCKT-style competition | Lu et al. 2025: concurrent multi-task surrogate-assisted search; Bayesian transferability τ (Spearman prior N(R, 0.05²), evidence T = realised/predicted improvement), Thompson sampling, argmax competition every δ = 10 evaluations, transfer of the source's best point | inside DE: every 10 target evaluations (after generation 0) the current trial vector (defender, Δ_in from an isotropic GP's LCB μ − 2σ on the target archive) competes with every candidate source (Δ_ex = τ̃·impn·max(y_target), impn from a GP fitted once on the frozen source archive); a winning source's next best not-yet-transferred archive point replaces the trial and goes through DE selection; the winner's posterior is updated with R and T | sources are frozen archives (sequential setting); DE trial instead of a SAS acquisition; next-best untransferred point instead of re-sending the same point; code-version normalisation and likelihood-variance schedule 0.05²·exp(−(FE−FE₀)/δ); isotropic RBF GPs (scikit-learn) instead of MATLAB `fitrgp`; official MATLAB code not run (see IMPLEMENTATION_STATUS) |
| B4 | PMTO-style task model | Wei et al. 2025: unified GP over (x, θ) with task evolution; online task model = one GP per decision variable θ → x*, trained on the top 70% of tasks | task model only ("PMTO-lite"): θ = scenario factors + physical leak-node descriptors; one ARD-RBF GP per decision variable on the library runs' best solutions (top 70% by gap to the scenario's long-run optimum); the posterior mean and round(ρN)−1 perturbed copies (σ = 3%) are injected | no offline task evolution or unified GP (the library is given); injection instead of the pure online answer (both visible: slot 0 holds the prediction itself) |
| B5 | Historical statistics | – | mean training-pair utility in bins of |Δ log leak area|, |Δ demand| (training terciles) and zone mismatch; transfer to the best predicted candidate if its bin mean > 0 | – |
| B6 | Proposed zero-shot | – | M5 (tiers a–c) predictor + Learn-then-Test gate, α = 0.10 / 0.20 | – |
| B7 | Proposed with probe | – | M6 (tiers a–d) after probing q = 8 injected elites of the zero-shot top source; rejection costs q evaluations (fresh best-so-far at B − q) | – |
| Oracle | Hindsight | – | best measured action per target (fresh counts as 0) | ceiling only |

Hyperparameters of B2γ, B3 and B4 are the papers' published values (γ, Γ, δ, σ_I², 70%), so no tuning budget was
spent on any baseline; the proposed models were tuned by grouped CV on training targets only.
