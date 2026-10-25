### Candidate A5_1_HedgeAnnealedClip
Combines the two strongest mechanisms: the scale-free multiplicative-weights (Hedge) reputation on log distance ratios from A4.4, plus the annealed self-weight from A4.1 so that late iterations mix faster once reputations have settled (this targets the weaker "none" scenario). The clipping radius uses a slightly wider base quantile (0.3) to speed up honest mixing.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    D = Y - xs[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    med = float(np.median(nrm)) + 1e-12
    loss = np.clip(np.log(np.maximum(nrm, 1e-12) / med), 0.0, 3.0)
    L = state.get('L')
    n = state.get('n', 0)
    if L is None or len(L) != k:
        L = np.zeros(k)
        n = 0
    L = L + loss
    n = n + 1
    state['L'] = L
    state['n'] = n
    Lb = L / n
    r = np.exp(-2.5 * (Lb - Lb.min()))
    tau = float(np.quantile(nrm, 0.3)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    out = xs + (w[:, None, None] * D).sum(0) / (lam + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A5_2_DecisionAgreementHedge
New idea: besides geometric distance, each neighbour is scored on a *decision-level* signal, the fraction of public samples whose argmax disagrees with a robust reference (coordinate-wise median of self and neighbours). The two relative losses (log-ratio to the median neighbour) are averaged inside the Hedge reputation. Attacks that change predictions but stay close in L2, or that are far in L2 but harmless, are then weighted by how much they would actually change the labels.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    D = Y - xs[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    medn = float(np.median(nrm)) + 1e-12
    loss_n = np.clip(np.log(np.maximum(nrm, 1e-12) / medn), 0.0, 3.0)
    ref = np.median(np.concatenate([xs[None], Y], 0), 0)
    am = np.argmax(ref, -1)
    dis = (np.argmax(Y, -1) != am[None]).mean(1)
    eps = 0.02
    loss_d = np.clip(np.log((dis + eps) / (float(np.median(dis)) + eps)), 0.0, 3.0)
    loss = 0.5 * (loss_n + loss_d)
    L = state.get('L')
    n = state.get('n', 0)
    if L is None or len(L) != k:
        L = np.zeros(k)
        n = 0
    L = L + loss
    n = n + 1
    state['L'] = L
    state['n'] = n
    Lb = L / n
    r = np.exp(-2.5 * (Lb - Lb.min()))
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A5_3_ReputationWeightedGeoMedian
New aggregator: instead of a clipped mean, each node computes a reputation-weighted geometric median (Weiszfeld, warm-started at its own state) of {self, neighbours}. Self has weight 1 and neighbours have Hedge reputations. The geometric median has breakdown point 1/2 in weight, so reputation decay and median robustness reinforce each other. The step toward it is then partly damped to keep gossip stable.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    D = Y - xs[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    med = float(np.median(nrm)) + 1e-12
    loss = np.clip(np.log(np.maximum(nrm, 1e-12) / med), 0.0, 3.0)
    L = state.get('L')
    n = state.get('n', 0)
    if L is None or len(L) != k:
        L = np.zeros(k)
        n = 0
    L = L + loss
    n = n + 1
    state['L'] = L
    state['n'] = n
    Lb = L / n
    r = np.exp(-2.5 * (Lb - Lb.min()))
    P = np.concatenate([xs[None], Y], 0).reshape(k + 1, -1)
    wt = np.concatenate([[1.0], r])
    z = xs.reshape(-1).copy()
    floor = 1e-3 * med + 1e-9
    for _ in range(8):
        d = np.linalg.norm(P - z[None], axis=1)
        a = wt / np.maximum(d, floor)
        z = (a[:, None] * P).sum(0) / a.sum()
    z = z.reshape(xs.shape)
    step = 0.6 + 0.3 * float(t) / max(T - 1, 1)
    out = xs + step * (z - xs)
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A5_4_ConfidenceRowWeightedHedgeClip
Keeps the Hedge-reputation radius clipping at message level, which gives the safety guarantees. Inside each sample row, the clipped contributions (self included) are weighted by normalised prediction confidence (1 - entropy/log C), bounded to [0.5, 2] relative to the row average. Noisy-label honest nodes, whose predictions tend to be flatter, then count less per sample, and the bound on the relative weight limits how much a Byzantine node can gain by faking confidence.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k, m, C = Y.shape
    D = Y - xs[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    med = float(np.median(nrm)) + 1e-12
    loss = np.clip(np.log(np.maximum(nrm, 1e-12) / med), 0.0, 3.0)
    L = state.get('L')
    n = state.get('n', 0)
    if L is None or len(L) != k:
        L = np.zeros(k)
        n = 0
    L = L + loss
    n = n + 1
    state['L'] = L
    state['n'] = n
    Lb = L / n
    r = np.exp(-2.5 * (Lb - Lb.min()))
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    Yc = xs[None] + scale[:, None, None] * D
    Z = np.concatenate([xs[None], Y], 0)
    H = -(Z * np.log(np.maximum(Z, 1e-12))).sum(-1)
    conf = 1.0 - H / math.log(C) + 0.1
    rho = conf / np.maximum(conf.mean(0, keepdims=True), 1e-12)
    rho = np.clip(rho, 0.5, 2.0)
    ws = rho[0]
    wn = r[:, None] * rho[1:]
    num = ws[:, None] * xs + (wn[:, :, None] * Yc).sum(0)
    den = ws + wn.sum(0)
    out = num / den[:, None]
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```
