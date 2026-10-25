"""Frontier on DESIGN-side data only (FMNIST snapshots): gross-attack robustness (design_val score)
vs inlier-drift robustness (drift, T=40) for every rule; used to pick ARCO configurations."""
import json, itertools, multiprocessing as mp
import numpy as np
import consensus as C, evaluate as E
from arco import make_arco
from programs import load_program

def rules():
    R = {f"ARCO {l} {c} q{q} d{d} k{k}": ("arco", dict(loss=l, center=c, q=q, delta=d, kappa=k))
         for l, c, q, d, k in itertools.product(("huber", "log"), ("self", "median"), (0.25, 0.5), (0.5, 1.0, 2.0), (0.0, 0.1))}
    for b in C.BASELINES:
        R[b] = ("base", b)
    for n, f in [("LLM-A", "A/A1.4_ReputationClippedGossip.py"), ("LLM-B", "B/B2.3_B2_3_TwoLevelRobustMean.py"),
                 ("LLM-C", "C/C1.3_HellingerGM.py"), ("LLM-H", "H/H5.1_H5_1_PersistentBandClippedMix.py")]:
        R[n] = ("prog", f"{E.ROOT}/design_runs/{f}")
    return R

def build(spec):
    kind, a = spec
    return make_arco(**a) if kind == "arco" else C.BASELINES[a] if kind == "base" else load_program(a)

def job(args):
    name, spec, T, atk, r, s = args
    E.T_CONS = T
    out = E.run_scenario(build(spec), ("fmnist", r, atk, "er", 0.2, "random", s))
    return name, T, atk, out["acc"] / out["oracle_acc"]

if __name__ == "__main__":
    R = rules()
    jobs = [(n, sp, T, atk, r, s) for n, sp in R.items() for (T, atk) in ((40, "drift"), (40, "none"), (10, "drift"))
            for r in (1, 4, 10) for s in (1, 2)]
    with mp.get_context("fork").Pool(6) as p:
        out = p.map(job, jobs)
    res = {}
    for n, sp in R.items():
        C.BASELINES["_t"] = build(sp)
        v = E.evaluate(name="_t", suite="design_val", procs=6)["summary"]
        g = lambda T, a: float(np.mean([o[3] for o in out if o[0] == n and o[1] == T and o[2] == a]))
        res[n] = dict(val=v["score"], val_mean=v["mean_acc"], drift40=g(40, "drift"), none40=g(40, "none"), drift10=g(10, "drift"))
        print(f"{n:34s} val={v['score']:.4f} drift40={res[n]['drift40']:.3f} drift10={res[n]['drift10']:.3f} none40={res[n]['none40']:.3f}", flush=True)
    del C.BASELINES["_t"]
    json.dump(res, open(f"{E.ROOT}/results/frontier_design.json", "w"), indent=1)
