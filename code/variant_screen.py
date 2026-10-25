"""Screen ARCO variants on validation suite + drift accumulation (T=40) + no-attack (T=40)."""
import json, sys, multiprocessing as mp
import numpy as np
import consensus as C, evaluate as E
from arco import make_arco
V = {}
for loss in ("huber", "log"):
    for center in ("self", "median"):
        for q in (0.25, 0.5):
            for d in (0.5, 1.0, 2.0):
                V[f"{loss} {center} q{q} d{d}"] = dict(loss=loss, center=center, q=q, delta=d)
def drift_job(a):
    name, kappa, ds, atk, s = a
    E.T_CONS = 40
    r = E.run_scenario(make_arco(kappa=kappa, **V[name]), (ds, 4, atk, "er", 0.2, "random", s))
    return (name, kappa, atk, r["acc"] / r["oracle_acc"])
if __name__ == "__main__":
    kappas = [float(k) for k in sys.argv[1:]] or [0.0, 0.1]
    jobs = [(n, k, ds, atk, s) for n in V for k in kappas for ds in ("cifar10", "fmnist_a01") for atk in ("drift", "none", "alie") for s in (1, 2)]
    with mp.get_context("fork").Pool(6) as p:
        out = p.map(drift_job, jobs)
    for n in V:
        for k in kappas:
            C.BASELINES["_t"] = make_arco(kappa=k, **V[n])
            v = E.evaluate(name="_t", suite="design_val", procs=6)["summary"]["score"]
            d = {a: np.mean([o[3] for o in out if o[0] == n and o[1] == k and o[2] == a]) for a in ("drift", "none", "alie")}
            print(f"{n:24s} k={k:<5} val={v:.4f}  T40 drift={d['drift']:.3f} none={d['none']:.3f} alie={d['alie']:.3f}", flush=True)
