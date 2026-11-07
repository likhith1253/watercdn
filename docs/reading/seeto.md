# SEETO: reading notes

**Paper:** H. Fang, B. Li, P. Yang, "Efficient Parameter Calibration of Numerical Weather Prediction Models via Evolutionary Sequential Transfer Optimization", arXiv:2601.08663 **v4** (30 Aug 2026). Read from the full PDF of v4, including the supplement.

## Problem
Tasks arrive one after another. Each task calibrates WRF physics parameters (6 continuous parameters, Table I) for one 12-hour forecast window. The objectives are bi-objective: RMSE of wind speed and RMSE of temperature against CLDAS observations. Every evaluation is one WRF run, about 4.5 min on 32 cores. An archive A = {(s_i, D_i, G_i)} holds, for each earlier task i, its observed meteorological state s_i, every evaluated (x, F) pair D_i, and a fitted GP surrogate G_i. The aim is to give a new target task a good start under a very small budget.

## Method
**Task/state representation.** A convolutional autoencoder is trained offline on ERA5 data from 2013 to 2016, using MSE reconstruction loss. The encoder has 3 stages (ConvNormAct + ResidualBlock) with 64/128/256 channels, GroupNorm, GELU and AdamW (lr 8e-4, weight decay 1e-5). The latent dimension is 256. The input is λ = 19 channels: 4 surface variables plus 5 upper-air variables at 3 pressure levels. Each time step t of a task's observed field s^t (λ×H×W) is encoded to z^t = f(φ, s^t). Similarity does **not** use the decision or objective space, and it needs no target-task samples.

**Similarity, Eq. 3: mean cosine similarity over time steps.**
sim_i = (1/T) Σ_{t=1..T} (z_i^t · z_o^t) / (‖z_i^t‖ ‖z_o^t‖)
Source and target are compared step by step at the same index t, so the windows must have equal length T.

**Source selection.** A hard top-γ filter keeps the γ most similar archived tasks, with **γ = 5** (Table II). Inside that subset, Eq. 5 gives softmax weights:
w_i = exp(sim_i/Γ) / Σ_{j=1..γ} exp(sim_j/Γ), with temperature **Γ = 0.065**.
The ablation "NTWS-SEETO" uses top-1 instead (γ = 1) and does worse in 36 of 40 comparisons.

**Solution-level warm start (Algorithm 1, lines 10-18).** Population size n_p = 100. The injected count is n_opt = ⌊ρ·n_p⌋ with **ρ = 0.2**, which gives 20 injected solutions. Source i contributes n_i = w_i · n_opt solutions. The paper does not say how it rounds n_i. The n_i solutions come from that source's evaluated set D_i: run non-dominated sorting, take front F1, and truncate by crowding distance if |F1| > n_i. If |F1| < n_i, add whole subsequent fronts, then truncate the last partial front by crowding distance (Eq. 7). The remaining n_p − |P_inj| = 80 individuals are **uniform random**. Injected solutions are copied as-is. The paper describes no mapping, perturbation or re-evaluation, and the decision space is the same across tasks.

**Model-level transfer: adaptively weighted surrogate (Eq. 6).**
G_o = (1 − β) Σ_{i∈S} w_i G_i + β G_loc, with β = 1 − exp(−c·FE)
G_loc is a GP fitted only on target evaluations. Before any target evaluation exists, only the source ensemble is used. The constant c is **0.038 or 0.017**. Table II and the text say it switches on a threshold **τ = 0.7**: the larger c applies when a high-similarity source exists (written "w_i ≥ τ" in the paper), and the smaller c otherwise. Each iteration generates offspring by evolution under G_o, picks one candidate with an acquisition function, and runs WRF on it. One expensive evaluation is made per iteration. The SAS-CKT-MO baseline section implies the acquisition is HV improvement. The specific variation operators (SBX/PM vs. DE) are **not stated**; the implementation runs on PlatEMO. After the run, the target's (s_o, D_o, G_o) are added to the archive.

## Protocol
There are K = 20 source tasks (12-hour windows, 1 to 11 July 2017) and 10 target tasks (August 2017). By construction, targets 1-7 have highly similar sources and targets 8-10 do not. The main budget is **FE = 20** WRF evaluations, matching an operational window of about 1-1.5 h. An extended budget of FE = 60 is also reported. Each setting gets 10 independent runs. The metric is HV with reference point r = (1, 3), plus a Wilcoxon rank-sum test. Baselines are MOTPE, MOASMO, KMO and SAS-CKT-MO; KMO and MOASMO start from 20 QMC samples. Ablations are EIT-SEETO (injection only), SET-SEETO (surrogate only) and NTWS-SEETO (top-1). Sensitivity is reported for λ, c, Γ and τ.

## Limitation we build on
On low-similarity targets (8-10), SEETO's advantage shrinks. With FE = 60, KMO and MOASMO overtake it late in the run, and SEETO-KMO/-MOASMO end up worse than their non-transfer versions. The authors call this negative transfer. In the conclusion they say static similarity alone is "still insufficient to fully accommodate the dynamically changing search requirements", and they ask for more adaptive ways to avoid negative transfer. Transfer is always on: ρ = 0.2 injection happens whatever the similarity is. The top-γ filter is relative, not absolute, so the paper has no "do not transfer" decision. τ only changes the surrogate fade-out rate c. That gap motivates a learned decision about whether, and how much, to warm-start.

## SEETO-style baseline for our DE pump/PRV setting
1. **Task descriptor z:** our state features (demand pattern, tank levels, tariff, etc.) take the place of the meteorological autoencoder. Either use the raw normalised vector or a small AE latent. If the features are time-indexed, compute the per-step cosine and average it over time, as Eq. 3 does; otherwise use one cosine.
2. Keep the top γ = 5 archived tasks and compute softmax weights w_i with temperature Γ = 0.065. Retune Γ if the scale of our cosine values differs.
3. **DE initial population:** take ⌊0.2·NP⌋ individuals from the archived final evaluated sets, with source i contributing ≈ w_i·n_opt. Choose them with non-dominated sort plus crowding distance, or with plain best-fitness ranking if our objective is single-objective penalised cost. Fill the rest uniformly at random. Use no repair beyond bound clipping.
4. Optional: add model-level transfer with a weighted source-GP ensemble blended into a target GP using β = 1 − exp(−c·FE). The paper's pure "injection only" variant is EIT-SEETO. That is the cleanest analogue for a warm-start-decision study.
5. Always transfer. Never gate on low similarity, so the baseline stays faithful.

## Code availability
None. The arXiv page and the PDF (v4) contain no repository link. The implementation is described as built on PlatEMO (MATLAB).
