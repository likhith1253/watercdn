# BCKT — reading notes

**Paper.** Lu, Xue, Zhang, Zhang, Chen, Cao, Liu, Tan. *Multi-Task Surrogate-Assisted Search with Bayesian Competitive Knowledge Transfer for Expensive Optimization*. arXiv:2510.23407. Code: github.com/XmingHsueh/MSAS-BCKT.

**Setting in the paper:** *multi-task* — n tasks are optimised at the same time, each by its own surrogate-assisted search (SAS). The sources are the other tasks, and their archives keep growing during the run. Our setting is different: sequential, with a frozen library of sources.

Notation: task j is the target and task i is a source. D^j = [X^j, y^j] is the archive of truly evaluated points (minimisation). \hat f_i is a surrogate fitted on D^i. All tasks share one normalised space [0,1]^d, and in the code every task has the same d.

## 1. Internal and external improvement
- **Internal improvement (Eq. 1).** The SAS returns its best candidate x_p^j on the target surrogate under the infill criterion I(.), which is minimised:
  `Δ_in^j = min_{x∈X^j} I(x) − I(x_p^j)`.
  For BO-LCB, I = μ − 2σ from the target GP. For the RBF-based solvers, I is the plain prediction. Δ_in is in raw objective units (`improvement_internal.m`).
- **External improvement (Eq. 2).** The source's best-ever solution x_best^i = argmin y^i is scored with the *source* surrogate over the target's evaluated points:
  `Δ_p^{ij} = [min_{x∈X^j} \hat f_i(x) − min(y^i)] · max(y^j)/max(y^i)`, and `Δ_ex^{ij} = τ^{ij} Δ_p^{ij}`.
  The code (`improvement_external.m`) differs slightly:
  - It computes `impn = (min \hat f_i(X^j) − min y^i) / max([y^i; \hat f_i(X^j)])`.
  - If impn < 0, the source's Δ_ex is set to −inf, which excludes that source.
  - Otherwise `Δ_ex = τ̃ · impn · max(y^j)`.

  So the normaliser in the code is not the paper's. Both versions implicitly assume y > 0.

## 2. Bayesian transferability τ^{ij}
τ^{ij} is a latent scalar. It scales the improvement predicted from the source into the improvement actually realised on the target. Every term below is Gaussian, so the posterior stays Gaussian (Eqs. 4–15).

- **Prior belief.** Starts as p^{(0)} = 1 (non-informative). After each update, the previous posterior becomes the prior. The paper mentions that a domain-informed prior could be used, but none is used in the experiments.
- **A-priori criterion.** `R_k^{ij}` is the Spearman rank correlation between \hat f_i(X^j) and y^j (Eq. 6). It enters as `p^c(τ) = N(R_k, σ_R²)` with a constant `σ_R² = σ_I² = 0.05²`.
- **Likelihood (empirical evidence).** Observed only after a transfer has actually happened. The observation is realised improvement divided by predicted improvement (Eq. 8):
  `T_k^{ij} = [min(y^j) − f_j(x_best^i)] / Δ_p^{ij}`, with `p(T_k|τ) = N(T_k, σ_{T_k}²)`.
  The paper sets `σ_{T_l}² = σ_I² e^{−l}`, so the variance shrinks with the transfer count l.
- **Update at the k-th transfer.**
  `p^{(k)}(τ) ∝ N(T_k, σ_T²) · N(R_k, σ_R²) · p^{(k−1)}(τ)`.
  In closed form (Eqs. 13–14):
  `τ̂_k = ω Σ_l z_l T_l + (1−ω)(1/k) Σ_l R_l`, and `σ_k² = σ_I²/(k+ε)`,
  where `ε = Σ_{l≤k} e^l`, `ω = ε/(k+ε)`, and `z_l = e^l/ε`.
  As k grows, the observed evidence T dominates the rank-correlation prior; the paper argues this estimate is asymptotically unbiased.
- **Use at each competition.** Draw a Thompson-style sample `τ̃^{ij} ~ N(τ̂_k, σ_k²)` (Eq. 15) and plug it into Δ_ex.
- **What the code does** (`mteo_solver.m`, `knowledge_competition.m`):
  - It uses a sequential precision-weighted update: `μ ← (μ σ_new² + x σ²)/(σ² + σ_new²)`, `σ² ← σ² σ_new²/(σ² + σ_new²)`.
  - The update runs in two steps: first with x = R (σ_new² = 0.05²) at every trigger, then with x = T after a transfer.
  - The first trigger initialises μ = R and σ² = 0.05².
  - If the target's own candidate wins, the R step is rolled back. In net, R is therefore counted only on transfer events, which matches Eq. 5.
  - The likelihood variance is `0.05²·exp(−(FE_used − FE_init)/gen_gap)`. **This is a paper/code discrepancy:** the code shrinks the variance with elapsed evaluations, while the paper shrinks it with the transfer count l.

## 3. Competition rule and budget
- **One evaluation per task per iteration.** Each iteration, every task runs one SAS acquisition to get (x_p^j, Δ_in^j), then exactly one real evaluation is made per task.
- **Trigger.** BCKT runs only when `mod(|D^j|, δ) = 0`, with δ = 10. The code checks `mod(FE_used, n·10)`, which amounts to every 10 evaluations per task. On iterations without a trigger, x_p^j is evaluated directly.
- **Competition.** When triggered, set Δ^{jj} = Δ_in^j and Δ^{ij} = Δ_ex^{ij} for each source i, then evaluate the argmax (Eq. 18).
  - A transfer happens only if `max_i Δ_ex^{ij} > Δ_in^j`. On a tie, the target's own candidate is kept.
  - **Only the winning source** has its posterior updated.
- **What is transferred.** Only the source's current best-ever solution. A footnote in the paper says other "promising" solutions could be used instead. The point is copied unchanged into the target, since all tasks share the normalised space and there is no mapping.
- **Cost.** A transfer uses one target evaluation from the same budget. To compute T, the code also calls f_j on that same point once before the counted evaluation, without counting it. Because it is the same point, this does not change any decision.
- **Duplicates.** If a candidate lies within 5e−3 (L∞ distance, normalised space) of an existing archive point, it is randomly perturbed with growing scales until it is new.

## 4. Sources, surrogates, benchmarks
- **How sources are represented.**
  - Each source is its full archive D^i = [X^i, y^i], and it keeps growing because the tasks are optimised together.
  - Each source has a GP (`fitrgp`, linear basis, standardised), refit at every trigger. The same GP is used for R (predictions on X^j) and for Δ_p.
  - The candidate set is just {argmin y^i}, one point per source.
- **Backbone SAS.** BCKT plugs into any SAS solver. Six were tested:
  - BO-LCB: GP with LCB (w = 2), optimised on the surrogate by an EA with population 50 for 50 generations.
  - TLRBF, GL-SADE, DDEA-MESS, LSADE, AutoSAEA.
- **Benchmarks.**
  - 9 two-task MTOPs grouped by similarity (HS/MS/LS). Tasks are shifted Ackley, Griewank, Sphere, Rastrigin, Elliptic, Rosenbrock, Weierstrass, Schwefel and similar, with d = 10–20.
  - 6 five-task MaTOPs with grouped similarities.
  - A case study with 3 well-placement tasks: NPV objective, 70 simulations per task, BO-LCB backbone.
- **Budgets and settings.**
  - 50 LHS initial points per task (the same design for all tasks).
  - 200 × n FEs in total: 400 for the MTOPs, 1000 for the MaTOPs.
  - δ = 10, σ_I² = 0.05².
  - 30 runs, compared with the Wilcoxon rank-sum test at α = 0.05.
- **Peer methods.** MaTDE, MFEA-II and MMaTEA-DGT (population 10 × 20 generations); LCB-EMT, RAMTEA and SELF. Per-problem details are in the supplement, which I did not read.
- **Repository.**
  - **Language:** MATLAB. The README contains only the title, and I found no licence file.
  - **Entry points:** `mteo_main.m` (MTOPs) and `mateo_main.m` (MaTOPs). Each loops over problems × backbones × 30 runs and calls `utils/mteo_solver.m`.
  - **Output:** results go to `.\results-mteo\<opt>\` and `.\results-mateo\<opt>\`. These folders must already exist, and the paths use Windows separators.
  - **Core BCKT files:** `utils/ckt-components/{knowledge_competition, improvement_external, similarity_quantification}.m` and `utils/commons/improvement_internal.m`.
  - **Dependencies (inferred from the calls):** Statistics & ML Toolbox (`fitrgp`, `fitglm`, `normrnd`) and the Deep Learning/Neural Network Toolbox (`newrbe` for RBF).
  - `utils/optimizers/saea_source.m`, a single-task GP-LCB source solver, is in the repo, but I did not see the main scripts call it.

## 5. Proposed minimal "BCKT-style" adaptation for our setting
Our setting: the target is optimised by DE/rand/1/bin with population NP and a fixed budget of B true evaluations. The sources are a library of completed runs, each with its archive D^i.

| # | Original | Adaptation | Why |
|---|---|---|---|
| 1 | Sources evolve concurrently | Treat sources as **frozen** archives D^i. Fit \hat f_i once per source on D^i: a GP, or an RBF if the archive is large. Subsample to about 500 points or fewer, or use a sparse GP. | Sequential setting |
| 2 | Own candidate x_p comes from the SAS; Δ_in comes from the infill criterion | The own candidate is a DE trial vector of the current generation. Compute Δ_in = min_{x∈X^t} \hat f_t(x) − \hat f_t(u) with a **cheap target surrogate** (GP or RBF on the target archive, plain prediction). If a GP is used, use LCB μ − 2σ to stay closer to BO-LCB. At a trigger, the defender is the trial vector u* with the largest Δ_in. | DE has no acquisition function, and Eq. 1 needs some target model |
| 3 | 1 evaluation per task per iteration; trigger when mod(\|D\|, 10) = 0 | Trigger every δ = 10 target evaluations (once per generation if NP = 10). At a trigger, if max_i Δ_ex^{i} > Δ_in(u*), **replace u* with x_best^i**. Evaluate it and run DE's one-to-one selection against u*'s parent. Evaluate all other trial vectors as usual. Every evaluation counts against B. | Keeps "one slot, own vs. source" with the same budget |
| 4 | x_best^i is the current best of a growing source | With a frozen source, x_best^i would be re-sent every time. After it has been transferred, the next transfer from source i takes the **best archive point not yet transferred**. The alternative is the code's near-duplicate perturbation (ε = 5e−3), which turns a repeat into a random local probe. Report which option is used. | The original never re-sends the same point by design; a frozen archive would |
| 5 | R = Spearman(\hat f_i(X^t), y^t), σ_R² = 0.05² | Unchanged, recomputed at each trigger on the current target archive. It needs about NP target evaluations first, so there is no transfer in generation 0. Optional: initialise μ from library metadata (e.g. network or demand similarity) as the prior belief, flagged as an extension. | Faithful |
| 6 | T = realised / predicted improvement; σ_T² decays | Keep the formula. Pick **one** decay schedule: the paper's σ_I² e^{−l} (per transfer) or the code's 0.05²·exp(−(FE − FE_0)/δ) (per elapsed evaluation). I recommend the code's, since it produced the published results. State the choice. | Paper/code discrepancy |
| 7 | Normalisation by max(y) | Keep it if pump energy cost and the penalised objectives are strictly positive. Otherwise shift y to be positive within each task, or replace max(y) with the range. Either change is a deviation and must be stated. | Eq. 2 is ill-defined for y ≤ 0 |
| 8 | Argmax over the n−1 other tasks | With a library of N sources, compute Δ_ex for all N. That costs O(N·\|X^t\|) surrogate predictions per trigger. Pre-filtering to the top-K by R is possible, but it adds a selection step that BCKT does not have. | Scalability |
| 9 | Shared [0,1]^d, same d | Requires the same decision vector (pump schedules and PRV settings) across tasks. Map each task's bounds to [0,1]^d. Different networks or dimensions are outside BCKT's scope and would need an explicit mapping. | Makes the assumption explicit |

**Kept intact:**
- Thompson sampling of τ̃ from the Gaussian posterior.
- The argmax competition, with ties going to the target's own candidate.
- Updating only the winning source, and only on transfer events.
- Excluding a source when Δ_p < 0.
