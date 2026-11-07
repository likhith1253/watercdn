# PMTO: reading notes

**Paper:** T. Wei, J. Liu, A. Gupta, P. S. Tan, Y.-S. Ong, "(θl, θu)-Parametric Multi-Task Optimization: Joint Search in Solution and Infinite Task Spaces", arXiv:2503.08394 **v5** (9 Jul 2026). Read from the full PDF of v5, including the supplement.

## Problem
The setting is a continuous, bounded family of tasks: argmin_{x∈X} f(x, θ) for every θ ∈ Θ = (θl, θu) ⊂ R^D. The task parameter θ is known side information. The goal is a task model M: Θ → X with M(θ) ≈ x*_θ, so that any new θ gets a solution at run time with **zero** new evaluations. The paper targets expensive, derivative-free problems.

## Method
**Offline mode: a unified GP over X × Θ.** One GP is trained on D_pmt = {(x, θ_m, f(x, θ_m))} with inputs [x, θ], an RBF kernel, and hyperparameters fitted by Adam (lr 0.01, 500 epochs). Each task in the pool picks its next query with UCB (minimisation form):
x_m^(t) = argmax_x −μ_pmt(x, θ_m) + β σ_pmt(x, θ_m), with β = 1.0.
Theorem (Sec. IV): if κ_pmt((x,θ),(x',θ)) = κ_ind(x,x'), the maximal information gain is no larger than for independent per-task GPs. That gives a tighter GP-UCB regret bound, which is the knowledge-transfer argument.

**Task model M (the task→solution map): a GP, not a neural network.** M is built from V independent GPs, one per decision dimension v. Each is trained on D*_v = {(θ_m, x*_{m,v})}, where x*_m is the best solution found so far for pool task θ_m. The posterior is M_v(θ) = N(μ̃_v(θ), σ̃_v(θ)), with the standard GP mean and variance formulas (Eqs. 15-16), and M(θ) = [M_1(θ), …, M_V(θ)]. For the final model, only the top p = 70% of near-optimal pairs are used to train it (footnote 3). The paper does not say outright whether online prediction uses the posterior mean or a sample, but the metric f(M(θ), θ) implies a point prediction, most likely μ̃. It also does not say how predictions outside the X bounds are handled.

**Task evolution (offline exploration of Θ).** Each iteration adds one new task θ_new to the pool Ψ. It is chosen to maximise a DPP-style diversity score:
g(θ) = Σ_v det Q_v([θ_1..θ_M, θ]), with Q_v[i,j] = κ̃_v(θ_i, θ_j), using the task-model kernels.
The maximiser is a simple GA (P = 100, G = 50, binary tournament, SBX with η_c = 15 and p_c = 0.9, PM with η_m = 20 and p_m = 0.9). All pool tasks then get one UCB query each, and both GPs are refit (Algorithm 3).

**Online mode.** Given a new θ, return M(θ) directly with no search. In the evaluation, K random θ values are drawn in Θ, f(M(θ), θ) is computed for each, and quantiles at 5/25/50/75/95% are reported.

## Assumptions on Θ and extrapolation
- Θ is continuous, **bounded** and low-dimensional (D = 2-5 in the experiments; synthetic θ ∈ [0,1]^D). All tasks in the bounds are assumed **equally likely** (Sec. II-C). θ is assumed known and informative, and X is assumed shared across tasks.
- **Extrapolation:** the paper does not discuss or test θ outside (θl, θu). Every online test θ is sampled inside the bounds, and the method's name builds the bounds in. The only nearby remark is that the unified GP "can potentially generalize" beyond the *fixed training task set*, meaning interpolation within Θ. Diversity sampling can place tasks far from regions of interest, which hurts low quantiles on easy problems; the authors note this as a trade-off.
- Stated limitations: GP scalability to high dimensions, and multi-objective PMTO left open.

## Protocol
The total budget is Ntot = 2000 with Ninit = 200, starting from M = 20 tasks sampled by LHS (for the fixed-task baselines). There are 20 trials and a Wilcoxon signed-rank test at 90% confidence. Baselines: independent GP-BO, PMTO-FT (fixed tasks), SELF (expensive MTO), and PMTO-RT (random task sampling). The problems are 8 synthetic ones, f(x,θ) = g(λ(x − σ(Lθ))) with Sphere, Ackley, Rastrigin and Griewank, plus robot arm, crane-load I/II, and a minimax truss design.

## Limitation we build on
PMTO maps θ to a solution regardless of whether a query θ is in-distribution. It has no confidence-based decision to fall back to search from scratch, and no out-of-range handling. The posterior σ̃_v(θ) is available but is not used to make that call. In our setting, operating states can drift outside the offline range, which leaves exactly the "should I trust the warm start?" question open.

## PMTO-style baseline for our DE pump/PRV setting
1. θ is our continuous state descriptor (for example demand multipliers, initial tank levels, tariff level), normalised to [0,1]^D using the bounds of the offline archive.
2. Offline: for each archived task m, store θ_m and the best DE solution x*_m. Fit V independent GPs θ → x_v with an RBF/ARD kernel, keeping the top 70% of tasks by quality. A cheaper "PMTO-lite" skips task evolution and the unified GP and uses only the task model on our existing archive. Say so explicitly when reporting.
3. Online: x̂ = clip(μ̃(θ_new)). Either (a) evaluate x̂ alone (pure PMTO online mode), or (b) seed DE by injecting x̂, optionally with a few samples from N(μ̃, σ̃²) for diversity, then run normal DE. Report both. (b) goes beyond the paper.
4. Test both in-range and out-of-range θ_new, since the paper does not cover the second case.

## Code availability
Yes. The supplementary material (Sec. S-I) links https://github.com/ambigeV/pmto-tevc, and the repo URL responded HTTP 200 on 2026-11-07. The arXiv abs page itself lists no code link.
