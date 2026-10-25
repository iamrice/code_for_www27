"""Knowledge-consensus simulator: gossip over a graph with Byzantine neighbours.

Every honest node i holds a knowledge matrix x_i (m public samples x C classes) and,
for T synchronous iterations, replaces it with  aggregate(x_i, X_nb, t, T, state_i),
where X_nb stacks the k messages it received.  The aggregation rule is the object
being designed (by hand, by random search, or by an LLM).  A message costs m*C floats.
"""
import copy
import time
import numpy as np
import networkx as nx

# ----------------------------------------------------------------------------- graphs

def make_graph(kind, n, seed):
    rng = np.random.default_rng(seed)
    for trial in range(1000):
        s = int(rng.integers(1 << 30))
        if kind == "er":
            G = nx.erdos_renyi_graph(n, 0.3, seed=s)
        elif kind == "er_sparse":
            G = nx.erdos_renyi_graph(n, 0.15, seed=s)
        elif kind == "ring":
            G = nx.watts_strogatz_graph(n, 4, 0.0, seed=s)
        elif kind == "ws":
            G = nx.watts_strogatz_graph(n, 6, 0.2, seed=s)
        elif kind == "geo":
            G = nx.random_geometric_graph(n, 0.4, seed=s)
        else:
            raise ValueError(kind)
        if nx.is_connected(G):
            return [sorted(G.neighbors(i)) for i in range(n)]
    raise RuntimeError("no connected graph")


def place_byzantine(adj, frac, mode, seed):
    n = len(adj)
    b = int(round(frac * n))
    rng = np.random.default_rng(seed)
    if mode == "random":
        return set(rng.choice(n, b, replace=False).tolist())
    if mode == "localized":  # BFS ball around a random node: Byzantines clustered together
        start = int(rng.integers(n))
        order, seen, q = [], {start}, [start]
        while q:
            u = q.pop(0)
            order.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return set(order[:b])
    raise ValueError(mode)

# ----------------------------------------------------------------------------- attacks
# An attack returns the message Byzantine node j sends to honest node i at iteration t.
# Attackers are omniscient: they see all honest states (ctx["X"]) and the receiver's
# honest neighbourhood.  Colluding Byzantines send the same message to a given receiver.

def _honest_view(ctx):
    i, X, adj, byz = ctx["i"], ctx["X"], ctx["adj"], ctx["byz"]
    hn = [j for j in adj[i] if j not in byz]
    H = X[[i] + hn]
    return H


def atk_none(ctx):
    return ctx["X"][ctx["j"]]  # behaves honestly (own current state)


def atk_gaussian(ctx):
    return ctx["rng"].normal(0.0, 1.0, ctx["X"][0].shape)


def atk_signflip(ctx):
    return -ctx["Y"][ctx["j"]]


def atk_labelflip(ctx):
    y = ctx["Y"][ctx["j"]]
    return y[:, ::-1].copy()  # class c -> C-1-c, still a valid probability simplex row


def atk_alie(ctx, z=1.0):
    H = _honest_view(ctx)
    return H.mean(0) - z * H.std(0)


def atk_ipm(ctx, eps=0.5):
    return -eps * _honest_view(ctx).mean(0)


def atk_mimic(ctx):
    return ctx["X"][ctx["victim"]]  # everyone copies the same (noisy) honest node


def atk_zerosum(ctx):
    i, adj, byz, X = ctx["i"], ctx["adj"], ctx["byz"], ctx["X"]
    nb = len([j for j in adj[i] if j in byz])
    H = _honest_view(ctx)
    return -H.sum(0) / max(nb, 1)


def atk_dissensus(ctx, eps=1.5):
    i, adj, byz, X = ctx["i"], ctx["adj"], ctx["byz"], ctx["X"]
    hn = [j for j in adj[i] if j not in byz]
    nb = len([j for j in adj[i] if j in byz])
    if not hn:
        return X[i]
    return X[i] - eps * (X[hn] - X[i]).sum(0) / max(nb, 1)


def atk_targeted(ctx, target=0):
    m, C = ctx["X"][0].shape
    out = np.zeros((m, C))
    out[:, target] = 1.0
    return out


def atk_adaptive(ctx):
    """Agnostic optimisation attack (Shejwalkar & Houmansadr, NDSS'21 style) adapted to
    gossip: the attacker queries the *frozen* aggregation rule as a black box and picks,
    among {mean + s*d}, the message that pushes the receiver's output furthest from where
    it would go with honest messages only (and, as a tie-breaker, towards a wrong class)."""
    i, adj, byz, X, agg, st = ctx["i"], ctx["adj"], ctx["byz"], ctx["X"], ctx["agg"], ctx["state"]
    hn = [j for j in adj[i] if j not in byz]
    nb = len([j for j in adj[i] if j in byz])
    if not hn:
        return X[i]
    H = X[hn]
    mu, sd = np.concatenate([H, X[i:i + 1]]).mean(0), np.concatenate([H, X[i:i + 1]]).std(0) + 1e-6
    # reference output: Byzantine slots filled with the honest mean (same input size as the attack)
    ref = agg(X[i].copy(), np.concatenate([H, np.repeat(mu[None], nb, 0)]), ctx["t"], ctx["T"], copy.deepcopy(st))
    flip = mu[:, ::-1]
    dirs = [-sd, (flip - mu), -mu, ctx["rng"].normal(0, 1, mu.shape) * sd]
    best, best_val = mu, -1.0
    for d in dirs:
        for s in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            msg = mu + s * d
            Xn = np.concatenate([H, np.repeat(msg[None], nb, 0)])
            try:
                out = agg(X[i].copy(), Xn, ctx["t"], ctx["T"], copy.deepcopy(st))
                val = float(np.abs(out - ref).sum())
            except Exception:
                val = -1.0
            if np.isfinite(val) and val > best_val:
                best, best_val = msg, val
    return best


def atk_drift(ctx, target=0, frac=0.9):
    """Bounded-influence drift (introduced in the optimisation study): stay *inside* any
    clipping radius by moving only a fraction of the closest honest neighbour's distance,
    but always in the same direction (towards a target class).  Per-iteration influence is
    tiny, yet without an anchor it accumulates over iterations."""
    i, adj, byz, X = ctx["i"], ctx["adj"], ctx["byz"], ctx["X"]
    hn = [j for j in adj[i] if j not in byz]
    x = X[i]
    e = np.zeros_like(x)
    e[:, target] = 1.0
    u = e - x
    un = np.sqrt((u ** 2).sum(-1, keepdims=True))
    u = u / np.maximum(un, 1e-12)
    if hn:
        r = np.sqrt(((X[hn] - x) ** 2).sum(-1))  # (h, m) honest row distances
        rad = frac * r.min(0)[:, None]
    else:
        rad = np.full((x.shape[0], 1), 0.05)
    return x + rad * u


ATTACKS = {
    "none": atk_none, "gaussian": atk_gaussian, "signflip": atk_signflip,
    "labelflip": atk_labelflip, "alie": atk_alie,
    # ---- held-out (pre-registered, never shown to the designer) ----
    "ipm": atk_ipm, "mimic": atk_mimic, "zerosum": atk_zerosum,
    "dissensus": atk_dissensus, "targeted": atk_targeted, "adaptive": atk_adaptive,
    "drift": atk_drift,
}
DESIGN_ATTACKS = ["none", "gaussian", "signflip", "labelflip", "alie"]
HELDOUT_ATTACKS = ["ipm", "mimic", "zerosum", "dissensus", "targeted", "adaptive"]

# ----------------------------------------------------------------------------- runner

def finalize(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    s = x.sum(-1, keepdims=True)
    C = x.shape[-1]
    return np.where(s > 1e-12, x / np.maximum(s, 1e-12), 1.0 / C)


def run_consensus(Y, adj, byz, attack, agg, T=20, seed=0, victim=None):
    """Y: (n,m,C) each node's local knowledge.  Returns dict with final honest targets."""
    n = len(adj)
    rng = np.random.default_rng(seed)
    honest = [i for i in range(n) if i not in byz]
    if victim is None:
        victim = honest[0]
    X = Y.astype(np.float64).copy()
    states = {i: {} for i in honest}
    atk = ATTACKS[attack]
    t0 = time.time()
    for t in range(T):
        Xn = X.copy()
        for i in honest:
            msgs = []
            byz_msg = None
            for j in adj[i]:
                if j in byz:
                    if byz_msg is None:
                        ctx = dict(i=i, j=j, X=X, Y=Y, adj=adj, byz=byz, rng=rng, t=t, T=T,
                                   agg=agg, state=states[i], victim=victim)
                        byz_msg = np.asarray(atk(ctx), dtype=np.float64)
                    msgs.append(byz_msg)
                else:
                    msgs.append(X[j])
            out = agg(X[i].copy(), np.stack(msgs) if msgs else np.zeros((0,) + X[i].shape), t, T, states[i])
            out = np.asarray(out, dtype=np.float64)
            if out.shape != X[i].shape or not np.all(np.isfinite(out)):
                raise ValueError("aggregator returned invalid output")
            Xn[i] = out
        X = Xn
    return {"targets": {i: finalize(X[i]) for i in honest}, "raw": {i: X[i] for i in honest},
            "time": time.time() - t0, "honest": honest}


def score_targets(res, labels, Y_honest_mean=None):
    tg = res["targets"]
    H = res["honest"]
    acc = np.mean([(tg[i].argmax(-1) == labels).mean() for i in H])
    P = np.stack([tg[i] for i in H])
    disagree = 0.5 * np.abs(P - P.mean(0)).sum(-1).mean()  # mean TV to honest centroid
    return {"acc": float(acc), "disagree": float(disagree)}

# ----------------------------------------------------------------------------- baselines

def _flat(A):
    return A.reshape(A.shape[0], -1)


def agg_mean(x, Xn, t, T, s):
    return (x + Xn.sum(0)) / (len(Xn) + 1)


def agg_median(x, Xn, t, T, s):
    return np.median(np.concatenate([x[None], Xn]), 0)


def agg_trimmed(x, Xn, t, T, s, frac=0.3):
    """BRIDGE-T (Yang & Bajwa 2019): coordinate-wise trim b largest/smallest neighbours."""
    k = len(Xn)
    b = int(np.ceil(frac * k))
    if k - 2 * b < 1:
        return np.median(np.concatenate([x[None], Xn]), 0)
    S = np.sort(Xn, 0)[b:k - b]
    return (x + S.sum(0)) / (len(S) + 1)


def agg_krum(x, Xn, t, T, s, frac=0.1):
    A = np.concatenate([x[None], Xn])
    F = _flat(A)
    d = ((F[:, None] - F[None]) ** 2).sum(-1)
    f = int(np.ceil(frac * len(Xn)))
    keep = max(len(A) - f - 2, 1)
    sc = np.sort(d, 1)[:, 1:keep + 1].sum(1)
    return 0.5 * x + 0.5 * A[int(np.argmin(sc))]


def agg_geomed(x, Xn, t, T, s, iters=20):
    A = _flat(np.concatenate([x[None], Xn]))
    z = np.median(A, 0)
    for _ in range(iters):
        w = 1.0 / np.maximum(np.linalg.norm(A - z, axis=1), 1e-8)
        z = (w[:, None] * A).sum(0) / w.sum()
    return z.reshape(x.shape)


def agg_clippedgossip(x, Xn, t, T, s):
    """ClippedGossip (He, Karimireddy & Jaggi 2022); adaptive radius tau = 0.5 * 25%-quantile
    of neighbour distances (tuned on the design suite)."""
    k = len(Xn)
    D = Xn - x
    nrm = np.linalg.norm(_flat(D), axis=1)
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    return x + (scale[:, None, None] * D).sum(0) / (k + 1)


def agg_balance(x, Xn, t, T, s, gamma=1.5, kappa=1.0, alpha=0.5):
    """BALANCE (Fang et al., CCS 2024): accept neighbour j iff ||x_j-x_i|| <= gamma e^{-kappa t/T} ||x_i||."""
    nrm = np.linalg.norm(_flat(Xn - x), axis=1)
    thr = gamma * np.exp(-kappa * t / T) * np.linalg.norm(x)
    acc = Xn[nrm <= thr]
    if len(acc) == 0:
        return x
    return alpha * x + (1 - alpha) * acc.mean(0)


def agg_huber(x, Xn, t, T, s, iters=5):
    """Metric-based consensus (M-estimator with Huber loss, IRLS) -- the closest
    analytic analogue of the original MKC formulation with a fixed metric."""
    A = _flat(np.concatenate([x[None], Xn]))
    z = np.median(A, 0)
    for _ in range(iters):
        r = np.linalg.norm(A - z, axis=1)
        delta = np.median(r) + 1e-12
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
        z = (w[:, None] * A).sum(0) / w.sum()
    return z.reshape(x.shape)


BASELINES = {
    "Mean": agg_mean, "CW-Median": agg_median, "BRIDGE-T": agg_trimmed, "Krum": agg_krum,
    "GeoMed": agg_geomed, "ClippedGossip": agg_clippedgossip, "BALANCE": agg_balance,
    "Huber-IRLS": agg_huber,
}


# ARCO (anchored robust consensus optimisation) frozen configurations
from arco_configs import CONFIGS as _ARCO_CONFIGS, get as _arco_get
for _n in _ARCO_CONFIGS:
    BASELINES[_n] = _arco_get(_n)
