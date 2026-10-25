# Robust Aggregation Is Optimization: code, data and results

Code, every raw result, the pre-registration, every LLM-agent prompt and response, and the extended
version of the paper (`paper/extended_version.pdf`, with a 20-page appendix).

## Layout
```
PREREGISTRATION.md      held-out protocol, written before any automated design run
code/
  fl.py                 batched 20-node training on one GPU (grouped CNNs for images, MLPs for 20 Newsgroups)
  dfl.py                end-to-end decentralized FL with knowledge distillation (or DFedAvg)
  consensus.py          gossip simulator, 12 attacks, 8 hand-designed rules (+ ARCO configs)
  arco.py               ARCO: anchored robust consensus optimisation (losses, scales, MM/grad solvers)
  arco_configs.py       the three frozen ARCO configurations (G, B, H)
  evaluate.py           design / validation / held-out scenario suites
  loop.py, step.sh      LLM generate-evaluate-refine loop (driver side)
  randsearch.py         random program search over a hand-written grammar
  tune_baselines.py     baseline hyper-parameter tuning on the design suite
  tune_arco.py, variant_screen.py, frontier.py    ARCO exploration and design-side frontier
  heldout.py, heldout_drift.py, heldout_web.py, rerun_adaptive.py, lite_all.py   held-out evaluation
  drift_exp.py, kappa_sweep.py, verify_bound.py, abl_drift.py    analyses
  figs.py, tables*.py, supplement.py                             figures, tables, extended appendix
  e2e_queue.sh          sequential end-to-end job runner (skips finished jobs)
design_runs/
  A, B, C (large model), H (small model): prompt_r*.md, response_r*.md, every program, history.json
  R0/                   200 random-search programs
  ablation/             ablations of the best LLM-found rule
  arco/                 ARCO configurations as standalone programs
results/                all JSON results and logs (snapshots/, heldout/, e2e/, ...)
```

## Reproduce
Python 3.11, PyTorch 2.2, NumPy, scikit-learn, NetworkX, matplotlib. Datasets download automatically into `data/`.
```
cd code
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # macOS + fork
# 1. knowledge snapshots (clean runs)
python3 dfl.py --dataset fmnist  --method Mean --attack clean --seed 100 --rounds 10 --dump 1,4,10 --out ../results/snapshots/fmnist_s100.json
python3 dfl.py --dataset cifar10 --method Mean --attack clean --seed 200 --rounds 10 --dump 1,4,10 --out ../results/snapshots/cifar_s200.json
python3 dfl.py --dataset fmnist --alpha 0.1 --method Mean --attack clean --seed 300 --rounds 10 --dump 1,4,10 --out ../results/snapshots/fmnist_a01_s300.json
python3 dfl.py --dataset news20  --method Mean --attack clean --seed 400 --rounds 10 --dump 1,4,10 --out ../results/snapshots/news20_s400.json
# 2. tuning, design-side frontier, held-out evaluation
python3 tune_baselines.py ; python3 frontier.py
python3 heldout.py --baselines ; python3 heldout_drift.py ; python3 heldout_web.py ; python3 drift_exp.py ; python3 kappa_sweep.py ; python3 verify_bound.py
# 3. end-to-end training
./e2e_queue.sh ../results/e2e_jobs.txt ; ./e2e_queue.sh ../results/e2e_jobs2.txt ; ./e2e_queue.sh ../results/e2e_jobs3.txt ; ./e2e_queue.sh ../results/e2e_jobs_web.txt
# 4. figures and tables
python3 figs.py trajectory frontier frac driftT kappa ; python3 tables.py ; python3 tables_llm.py ; python3 tables_www.py ; python3 supplement.py
```
The LLM loop is driven by `loop.py prompt|ingest|select`; the designer only ever saw `design_runs/*/prompt_r*.md`.

## Deviations from the pre-registration
* The adaptive attack had a bug (state-size mismatch for stateful rules); it was fixed and **all** rules' adaptive rows were recomputed.
* The drift attack and the ARCO family were introduced after the pre-registration, in response to the optimisation analysis.
  ARCO configurations were selected on design-side (FMNIST) data only; earlier exploratory screens touched held-out data.
* The Web-text suite (20 Newsgroups) and its end-to-end runs were added afterwards; no rule was re-tuned or re-selected.
