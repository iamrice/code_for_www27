"""Score aggregation programs on design (seen) or held-out (unseen) consensus scenarios.

python evaluate.py --suite design  --prog cand.py           -> JSON with per-scenario results
python evaluate.py --suite design  --baselines              -> all handcrafted baselines
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import signal
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from consensus import make_graph, place_byzantine, run_consensus, score_targets, BASELINES, \
    DESIGN_ATTACKS, HELDOUT_ATTACKS
from programs import compile_program

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ""
SNAP = {
    "fmnist": f"{ROOT}/results/snapshots/fmnist_s100",
    "cifar10": f"{ROOT}/results/snapshots/cifar_s200",
    "fmnist_a01": f"{ROOT}/results/snapshots/fmnist_a01_s300",
    "news20": f"{ROOT}/results/snapshots/news20_s400",  # Web text (Usenet), added for the WWW version
}
T_CONS = 10


def scenarios(suite):
    """Each scenario: (dataset, round, attack, topo, byz_frac, byz_mode, seed)."""
    if suite == "design":  # the ONLY scenarios the designer (LLM or random search) gets feedback on
        return [("fmnist", r, a, "er", 0.2, "random", s)
                for r in (1, 4, 10) for a in DESIGN_ATTACKS for s in (0,)]
    if suite == "design_val":  # same distribution, fresh Byzantine placements (selection set)
        return [("fmnist", r, a, "er", 0.2, "random", s)
                for r in (1, 4, 10) for a in DESIGN_ATTACKS for s in (1, 2)]
    allatk = DESIGN_ATTACKS + HELDOUT_ATTACKS
    if suite == "heldout_attack":  # unseen attacks, seen data/topology
        return [("fmnist", r, a, "er", 0.2, "random", s)
                for r in (1, 4, 10) for a in HELDOUT_ATTACKS for s in (1, 2)]
    if suite == "heldout_data":  # unseen datasets, all attacks
        return [(d, r, a, "er", 0.2, "random", s)
                for d in ("cifar10", "fmnist_a01") for r in (1, 4, 10) for a in allatk for s in (1, 2)]
    if suite == "heldout_topo":  # unseen topologies on CIFAR knowledge
        return [("cifar10", 4, a, t, 0.2, "random", s)
                for t in ("ring", "ws", "geo", "er_sparse") for a in allatk for s in (1, 2)]
    if suite == "heldout_frac":  # Byzantine fraction sweep and localized placement
        return [("cifar10", 4, a, "er", f, mode, s)
                for f in (0.1, 0.2, 0.3, 0.4) for mode in ("random", "localized")
                for a in allatk for s in (1, 2)]
    if suite == "heldout_web":  # unseen Web-text knowledge (20 Newsgroups, 20 classes), all attacks incl. drift
        return [("news20", r, a, "er", 0.2, "random", s)
                for r in (1, 4, 10) for a in allatk + ["drift"] for s in (1, 2)]
    if suite == "heldout_lite":  # cheap held-out proxy used to score every candidate
        return [(d, 4, a, "er", 0.2, "random", 1) for d in ("cifar10", "fmnist_a01") for a in allatk]
    raise ValueError(suite)


_CACHE = {}


def load_snap(ds):
    if ds not in _CACHE:
        z = np.load(SNAP[ds] + "_snaps.npz", allow_pickle=True)
        meta = json.load(open(SNAP[ds] + ".json"))
        _CACHE[ds] = (z, meta["noisy"])
    return _CACHE[ds]


def run_scenario(agg, sc):
    ds, r, atk, topo, frac, mode, seed = sc
    z, noisy = load_snap(ds)
    Y = z[f"r{r}"].astype(np.float64)
    labels = z["labels"]
    n = Y.shape[0]
    adj = make_graph(topo, n, 1000 + seed)
    # Byzantines are drawn from non-noisy nodes, so honest nodes always include the noisy ones
    rng = np.random.default_rng(seed)
    if mode == "random":
        pool = [i for i in range(n) if i not in noisy]
        byz = set(rng.choice(pool, int(round(frac * n)), replace=False).tolist())
    else:
        byz = place_byzantine(adj, frac, mode, seed)
    victim = sorted(set(noisy) - byz)[0] if set(noisy) - byz else None
    res = run_consensus(Y, adj, byz, atk, agg, T=T_CONS, seed=seed, victim=victim)
    out = score_targets(res, labels)
    # reference: honest-only average consensus ("oracle" that knows who is Byzantine)
    H = [i for i in range(n) if i not in byz]
    out["oracle_acc"] = float((Y[H].mean(0).argmax(-1) == labels).mean())
    out["local_acc"] = float(np.mean([(Y[i].argmax(-1) == labels).mean() for i in H]))
    out["time"] = res["time"]
    return out


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def _worker(args):
    src, name, sc, tlimit = args
    try:
        agg = BASELINES[name] if src is None else compile_program(src)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(tlimit)
        out = run_scenario(agg, sc)
        signal.alarm(0)
        return dict(scenario=list(sc), ok=True, **out)
    except Timeout:
        return dict(scenario=list(sc), ok=False, error="timeout", acc=0.0)
    except Exception as e:
        signal.alarm(0)
        return dict(scenario=list(sc), ok=False, error=f"{type(e).__name__}: {e}"[:300], acc=0.0)


def evaluate(src=None, name=None, suite="design", procs=6, tlimit=120):
    scs = scenarios(suite)
    with mp.get_context("fork").Pool(procs) as pool:
        rows = pool.map(_worker, [(src, name, sc, tlimit) for sc in scs])
    accs = np.array([r["acc"] for r in rows])
    summ = dict(mean_acc=float(accs.mean()), worst_acc=float(accs.min()),
                score=float(0.5 * accs.mean() + 0.5 * accs.min()),
                n_fail=int(sum(not r["ok"] for r in rows)))
    # per-attack summary (mean over the other factors)
    per = {}
    for r in rows:
        per.setdefault(r["scenario"][2], []).append(r["acc"])
    summ["per_attack"] = {k: float(np.mean(v)) for k, v in per.items()}
    return dict(summary=summ, rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="design")
    ap.add_argument("--prog", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    results = {}
    if a.baselines:
        for b in BASELINES:
            results[b] = evaluate(name=b, suite=a.suite, procs=a.procs)
            print(b, json.dumps(results[b]["summary"]), flush=True)
    for p in a.prog:
        results[p] = evaluate(src=open(p).read(), suite=a.suite, procs=a.procs)
        print(p, json.dumps(results[p]["summary"]), flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
