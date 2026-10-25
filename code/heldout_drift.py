"""Held-out inlier-drift evaluation (CIFAR-10 / FMNIST-a0.1 knowledge), T in {10, 40}, all rules."""
import json, multiprocessing as mp
import numpy as np
import consensus as C, evaluate as E
from programs import load_program
D = f"{E.ROOT}/design_runs"
PROGS = {"LLM-A": f"{D}/A/A1.4_ReputationClippedGossip.py", "LLM-B": f"{D}/B/B2.3_B2_3_TwoLevelRobustMean.py",
         "LLM-C": f"{D}/C/C1.3_HellingerGM.py", "LLM-H": f"{D}/H/H5.1_H5_1_PersistentBandClippedMix.py",
         "Rand24first": f"{D}/R0/rand0_012.py", "Rand200": f"{D}/R0/rand0_197.py"}
def job(a):
    name, ds, r, T, s = a
    agg = C.BASELINES[name] if name in C.BASELINES else load_program(PROGS[name])
    E.T_CONS = T
    o = E.run_scenario(agg, (ds, r, "drift", "er", 0.2, "random", s))
    return dict(method=name, ds=ds, round=r, T=T, seed=s, acc=o["acc"], rel=o["acc"] / o["oracle_acc"])
if __name__ == "__main__":
    names = list(C.BASELINES) + list(PROGS)
    jobs = [(n, ds, r, T, s) for n in names for ds in ("cifar10", "fmnist_a01") for r in (1, 4, 10) for T in (10, 40) for s in (1, 2)]
    with mp.get_context("fork").Pool(5) as p:
        out = p.map(job, jobs)
    json.dump(out, open(f"{E.ROOT}/results/heldout_drift.json", "w"))
    for n in names:
        print(f"{n:14s}", " ".join(f"T{T}={np.mean([o['rel'] for o in out if o['method']==n and o['T']==T]):.3f}" for T in (10, 40)))
