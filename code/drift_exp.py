"""Accumulation experiment: accuracy vs number of gossip iterations T under the drift attack."""
import json, multiprocessing as mp, sys
import numpy as np
import consensus as C, evaluate as E
from arco import make_arco
from programs import load_program
METHODS = {n: (lambda n=n: C.BASELINES[n]) for n in ("ARCO-G", "ARCO-B", "ARCO-H", "ClippedGossip", "BALANCE", "GeoMed", "BRIDGE-T", "Mean")}
METHODS["LLM-A"] = lambda: load_program(f"{E.ROOT}/design_runs/A/A1.4_ReputationClippedGossip.py")
def job(a):
    name, ds, atk, T, seed = a
    agg = METHODS[name]()
    E.T_CONS = T
    r = E.run_scenario(agg, (ds, 4, atk, "er", 0.2, "random", seed))
    return dict(method=name, ds=ds, attack=atk, T=T, seed=seed, acc=r["acc"], rel=r["acc"] / r["oracle_acc"], disagree=r["disagree"])
if __name__ == "__main__":
    jobs = [(m, ds, atk, T, s) for m in METHODS for ds in ("cifar10", "fmnist_a01") for atk in ("drift", "none")
            for T in (5, 10, 20, 40, 80, 160) for s in (1, 2)]
    with mp.get_context("fork").Pool(6) as p:
        out = p.map(job, jobs)
    json.dump(out, open(f"{E.ROOT}/results/drift_T.json", "w"))
    for m in METHODS:
        print(f"{m:15s}", " ".join(f"{np.mean([o['rel'] for o in out if o['method']==m and o['T']==T and o['attack']=='drift']):.3f}" for T in (5,10,20,40,80,160)),
              "| none:", " ".join(f"{np.mean([o['rel'] for o in out if o['method']==m and o['T']==T and o['attack']=='none']):.3f}" for T in (5,10,20,40,80,160)))
