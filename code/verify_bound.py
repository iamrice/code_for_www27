"""Numerical check of Theorem 1 / Proposition 2: projected-gradient ARCO with fixed-delta Huber.
Measures ||X^T - X*_H||_F under several attacks against the bound delta*||b||_2/kappa."""
import json
import numpy as np
import consensus as C, evaluate as E
from arco import proj_simplex_rows

def huber_grad(D, delta):  # D: (..., m*C) flattened differences, node-level group
    r = np.linalg.norm(D, axis=-1, keepdims=True)
    return D * np.minimum(1.0, delta / np.maximum(r, 1e-12))

def solve(Y, adj, H, byz, kappa, delta, attack, iters, m, C_, rng):
    n = len(adj)
    dmax = max(len(a) for a in adj)
    eta = 1.0 / (kappa + 2 * dmax)
    X = Y.copy()
    for t in range(iters):
        G = np.zeros_like(X)
        for i in H:
            g = kappa * (X[i] - Y[i])
            for j in adj[i]:
                if j in byz:
                    if attack == "gaussian":
                        mj = rng.normal(0, 10, X[i].shape)
                    elif attack == "signflip":
                        mj = -100 * Y[j]
                    elif attack == "drift":
                        e = np.zeros((m, C_)); e[:, 0] = 1; mj = (X[i].reshape(m, C_) + 0.5 * (e - X[i].reshape(m, C_))).ravel()
                    else:
                        continue  # 'honest-only' reference problem: Byzantine edges removed
                else:
                    mj = X[j]
                g = g + huber_grad(X[i] - mj, delta)
            G[i] = g
        for i in H:
            X[i] = proj_simplex_rows((X[i] - eta * G[i]).reshape(m, C_)).ravel()
    return X

if __name__ == "__main__":
    z, noisy = E.load_snap("cifar10")
    m, C_ = 200, 10
    Y = z["r4"][:, :m].astype(float).reshape(20, -1)
    adj = C.make_graph("er", 20, 1001)
    rng = np.random.default_rng(0)
    byz = set(rng.choice([i for i in range(20) if i not in noisy], 4, replace=False).tolist())
    H = [i for i in range(20) if i not in byz]
    b = np.array([sum(j in byz for j in adj[i]) for i in H])
    out = []
    for kappa in (0.05, 0.2, 1.0):
        for delta in (0.5, 2.0):
            ref = solve(Y, adj, H, byz, kappa, delta, "ref", 3000, m, C_, rng)
            for atk in ("gaussian", "signflip", "drift"):
                Xm = solve(Y, adj, H, byz, kappa, delta, atk, 3000, m, C_, rng)
                dev = float(np.linalg.norm(Xm[H] - ref[H]))
                bound = delta * float(np.linalg.norm(b)) / kappa
                out.append(dict(kappa=kappa, delta=delta, attack=atk, dev=dev, bound=bound))
                print(f"kappa={kappa} delta={delta} {atk:9s} dev={dev:8.3f} bound={bound:8.3f} ratio={dev/bound:.3f}", flush=True)
    json.dump(out, open(f"{E.ROOT}/results/verify_bound.json", "w"), indent=1)
