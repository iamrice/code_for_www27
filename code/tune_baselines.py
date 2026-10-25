"""Tune each handcrafted baseline's hyper-parameters on the design suite (same feedback the
designer gets), so that the comparison is against tuned baselines."""
import functools, json
import numpy as np
import consensus as C
import evaluate as E

GRIDS = {
    "BRIDGE-T": [dict(frac=f) for f in (0.1, 0.2, 0.3, 0.4)],
    "Krum": [dict(frac=f) for f in (0.1, 0.25, 0.4)],
    "ClippedGossip": None,  # radius grid below
    "BALANCE": [dict(gamma=g, kappa=k, alpha=al) for g in (0.3, 0.5, 1.0, 1.5) for k in (0.0, 1.0) for al in (0.3, 0.5)],
    "Huber-IRLS": [dict(iters=i) for i in (5, 10, 20)],
}

def clipped(x, Xn, t, T, s, q=0.5, mult=1.0):
    k = len(Xn); D = Xn - x
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    tau = mult * np.quantile(nrm, q) + 1e-12
    sc = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    return x + (sc[:, None, None] * D).sum(0) / (k + 1)

best = {}
for name, grid in GRIDS.items():
    if name == "ClippedGossip":
        cands = [(dict(q=q, mult=m), functools.partial(clipped, q=q, mult=m)) for q in (0.25, 0.5, 0.75) for m in (0.5, 1.0, 2.0)]
    else:
        cands = [(g, functools.partial(C.BASELINES[name], **g)) for g in grid]
    res = []
    for g, fn in cands:
        C.BASELINES["_tmp"] = fn
        r = E.evaluate(name="_tmp", suite="design", procs=4)["summary"]
        res.append((r["score"], g))
        print(name, g, round(r["score"], 4), flush=True)
    res.sort(key=lambda z: -z[0])
    best[name] = res[0][1]
json.dump(best, open("../results/tuned_baselines.json", "w"), indent=1)
print(best)
