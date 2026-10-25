# Pre-registration (written 2026-10-20 06:57 CEST, before any LLM / random-search design run)

**Object being designed**: the per-node gossip aggregation rule
`aggregate(x_self, X_nb, t, T, state) -> (m,C)` used for T=10 synchronous consensus iterations on
knowledge matrices (soft predictions on a shared public set, m=1000, C=10).

**Designer feedback (the only signal any designer sees)**: suite `design` =
FMNIST (Dirichlet 0.5, 20 nodes, 4 honest noisy nodes w/ 60% label noise) snapshots at FL rounds {1,4,10}
x attacks {none, gaussian, signflip, labelflip, alie} x ER(p=0.3) graph, 20% Byzantine, placement seed 0.
Score = 0.5 * mean accuracy + 0.5 * worst-case accuracy of honest consensus targets on public labels.

**Selection rule (fixed)**: within each design run, the final program is the candidate with highest score on
suite `design_val` (same distribution, Byzantine placement seeds {1,2}). No held-out result is used for selection.

**Held-out evaluation (never shown to any designer)**:
1. `heldout_attack`: ipm, mimic, zerosum, dissensus (He et al. 2022), targeted (one-hot hijack), adaptive
   (black-box query of the frozen rule, Shejwalkar & Houmansadr style).
2. `heldout_data`: CIFAR-10 and FMNIST Dirichlet(0.1) knowledge, all 11 attacks.
3. `heldout_topo`: ring lattice, Watts-Strogatz, random geometric, sparse ER.
4. `heldout_frac`: Byzantine fraction {0.1,0.2,0.3,0.4} x {random, localized} placement.
5. End-to-end DFL on CIFAR-10 (distillation with consensus targets), honest test accuracy.

**Comparisons**: 8 handcrafted decentralized rules with hyper-parameters tuned on `design`
(results/tuned_baselines.json); random program search over a hand-written grammar with (a) the same
evaluation budget as one LLM run (24 programs) and (b) 8x that budget (200 programs), selected by the same rule;
LLM zero-shot (round-1 candidates only); 3 independent LLM runs (consistency); 1 run with a smaller LLM.

**Reporting**: all candidates, prompts and responses are logged in design_runs/. Negative results are reported.
