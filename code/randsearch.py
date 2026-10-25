"""Random program search over a hand-written grammar of robust gossip rules.

Each sampled program = robust centre  x  dissimilarity  x  weighting rule  x  granularity
x  self-weight  x  optional reputation memory  x  optional simplex projection.
The grammar deliberately contains the building blocks of all handcrafted baselines, so the
comparison asks whether the LLM adds anything beyond combinatorial search in a sensible space.

python randsearch.py --n 200 --seed 0 --out ../design_runs/R0
"""
import argparse
import json
import os
import numpy as np
import evaluate as E

CENTRES = {
    "self": "c = x_self",
    "median": "c = np.median(A, 0)",
    "mean": "c = A.mean(0)",
    "trimmed": "S = np.sort(A, 0); b = int(np.floor({trim} * len(A))); c = S[b:len(A) - b].mean(0) if len(A) - 2 * b >= 1 else np.median(A, 0)",
}
DISTS = {  # d: (k,) dissimilarity of each neighbour to centre c   [whole-vector granularity]
    "l2": "d = np.sqrt(((X_nb - c) ** 2).reshape(k, -1).sum(1))",
    "l1": "d = np.abs(X_nb - c).reshape(k, -1).sum(1)",
    "cos": "F = X_nb.reshape(k, -1); g = c.ravel(); d = 1 - F @ g / (np.linalg.norm(F, axis=1) * np.linalg.norm(g) + 1e-12)",
    "linf": "d = np.abs(X_nb - c).reshape(k, -1).max(1)",
    "argmax": "d = (X_nb.argmax(-1) != c.argmax(-1)[None]).mean(1) + 1e-3 * np.abs(X_nb - c).reshape(k, -1).mean(1)",
}
ROWDISTS = {  # d: (k, m) per-sample dissimilarity  [row granularity]
    "l2": "d = np.sqrt(((X_nb - c) ** 2).sum(-1))",
    "l1": "d = np.abs(X_nb - c).sum(-1)",
    "linf": "d = np.abs(X_nb - c).max(-1)",
}
RULES = {  # w: weights with same shape as d, from d and its median 'med'
    "hard": "w = (d <= {q} * med).astype(float)",
    "soft": "w = np.exp(-d / ({tau} * med))",
    "clip": "w = np.minimum(1.0, {q} * med / np.maximum(d, 1e-12))",
    "topk": "r = np.argsort(np.argsort(d, 0), 0); w = (r < max(1, int(np.ceil({frac} * k)))).astype(float)",
}


def sample(rng):
    centre = rng.choice(list(CENTRES))
    gran = rng.choice(["vec", "row"])
    dist = rng.choice(list(DISTS if gran == "vec" else ROWDISTS))
    rule = rng.choice(list(RULES))
    hp = dict(trim=round(rng.uniform(0.1, 0.35), 2), q=round(rng.uniform(0.5, 3.0), 2),
              tau=round(rng.uniform(0.3, 3.0), 2), frac=round(rng.uniform(0.4, 0.9), 2))
    selfw = round(rng.uniform(0.0, 0.7), 2)
    mem = rng.choice([None, round(rng.uniform(0.5, 0.95), 2)])
    proj = bool(rng.random() < 0.5)
    step = round(rng.uniform(0.3, 1.0), 2)
    L = ["def aggregate(x_self, X_nb, t, T, state):",
         "    k = len(X_nb)",
         "    A = np.concatenate([x_self[None], X_nb])",
         "    " + CENTRES[centre].format(**hp),
         "    " + (DISTS if gran == "vec" else ROWDISTS)[dist],
         "    med = np.median(d, 0) + 1e-12",
         "    " + RULES[rule].format(**hp)]
    if mem is not None:
        L += [f"    w = {mem} * state.get('w', w) + (1 - {mem}) * w",
              "    state['w'] = w"]
    if gran == "vec":
        L += ["    W = w[:, None, None]"]
    else:
        L += ["    W = w[:, :, None]"]
    L += ["    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)",
          "    has = (W.sum(0) > 1e-12)",
          "    agg = np.where(has, agg, x_self)",
          f"    out = {selfw} * x_self + {1 - selfw} * agg",
          f"    out = x_self + {step} * (out - x_self)"]
    if proj:
        L += ["    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)"]
    L += ["    return out"]
    spec = dict(centre=centre, gran=gran, dist=dist, rule=rule, selfw=selfw, mem=mem, proj=proj, step=step, **hp)
    return "\n".join(L) + "\n", spec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    hist = []
    for i in range(a.n):
        src, spec = sample(rng)
        name = f"rand{a.seed}_{i:03d}"
        p = f"{a.out}/{name}.py"
        open(p, "w").write(src)
        d = E.evaluate(src=src, suite="design", procs=6)["summary"]
        v = E.evaluate(src=src, suite="design_val", procs=6)["summary"]
        hist.append(dict(name=name, idx=i, spec=spec, src=src, file=p, design=d, val=v))
        print(name, round(d["score"], 4), round(v["score"], 4), spec, flush=True)
        json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
