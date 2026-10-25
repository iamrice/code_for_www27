"""Anchor-strength sweep (analysis, not selection): ARCO-B / ARCO-G configs with varying kappa."""
import json, multiprocessing as mp, numpy as np
import evaluate as E
from arco import make_arco
from arco_configs import CONFIGS
K = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
def job(a):
    cfg, k, ds, r, atk, s = a
    E.T_CONS = 40
    c = dict(CONFIGS[cfg]); c["kappa"] = k
    o = E.run_scenario(make_arco(**c), (ds, r, atk, "er", 0.2, "random", s))
    return dict(cfg=cfg, kappa=k, attack=atk, rel=o["acc"] / o["oracle_acc"], disagree=o["disagree"])
if __name__ == "__main__":
    jobs = [(c, k, ds, r, a, s) for c in ("ARCO-B", "ARCO-G") for k in K for ds in ("cifar10", "fmnist_a01") for r in (1, 4, 10)
            for a in ("drift", "none", "labelflip", "targeted") for s in (1, 2)]
    with mp.get_context("fork").Pool(4) as p:
        out = p.map(job, jobs)
    json.dump(out, open(f"{E.ROOT}/results/kappa_sweep.json", "w"))
    for c in ("ARCO-B", "ARCO-G"):
        for a in ("drift", "none", "labelflip", "targeted"):
            print(c, f"{a:9s}", " ".join(f"{np.mean([o['rel'] for o in out if o['cfg']==c and o['kappa']==k and o['attack']==a]):.3f}" for k in K))
        print(c, "disagree(none)", " ".join(f"{np.mean([o['disagree'] for o in out if o['cfg']==c and o['kappa']==k and o['attack']=='none']):.3f}" for k in K))
