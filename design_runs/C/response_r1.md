### Candidate SanRowClip
Project every message onto the probability simplex first, which caps how far any Byzantine row can be from an honest one. Then apply self-centred clipping separately for each sample row. The clipping radius for a row is a lower quantile of the neighbour distances on that row, so the radius adapts to how uncertain each sample is. The update is a convex combination of simplex points, so the state stays a valid distribution.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    D = X - x[None]                                  # (k, m, C)
    r = np.linalg.norm(D, axis=2)                    # (k, m) per-row distances
    tau = np.quantile(r, 0.4, axis=0) + 1e-8         # (m,) per-row radius
    scale = np.minimum(1.0, tau[None] / np.maximum(r, 1e-12))
    out = x + (scale[..., None] * D).sum(0) / (k + 1)
    return san(out)
```

### Candidate ArgmaxReputation
Each neighbour gets a score: the fraction of samples whose argmax matches the coordinate-wise median of everything received. The score is kept as an exponential moving average across iterations. The effective trust is the minimum of the remembered and the current score, so a node that turns malicious late is cut off immediately. Trust is relative to the best-agreeing participant, and the output is a trust-weighted mean in which the node's own state always has weight 1.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    med = np.median(np.concatenate([x[None], X], 0), 0)
    lab = med.argmax(1)
    agree = (X.argmax(2) == lab[None]).mean(1)       # (k,)
    a_self = float((x.argmax(1) == lab).mean())
    rep = state.get('rep')
    if rep is None or rep.shape != agree.shape:
        rep = agree.copy()
    else:
        rep = 0.6 * rep + 0.4 * agree
    state['rep'] = rep
    eff = np.minimum(rep, agree)
    ref = max(a_self, float(agree.max())) + 1e-12
    w = np.clip((eff - 0.5 * ref) / (0.4 * ref), 0.0, 1.0)
    out = (x + (w[:, None, None] * X).sum(0)) / (1.0 + w.sum())
    return san(out)
```

### Candidate HellingerGM
Map the sanitised distributions to sqrt space, where Euclidean distance is the Hellinger distance between distributions. In that space, compute a self-weighted geometric median for each sample row with a few Weiszfeld iterations, starting from the coordinate-wise median. The geometric median has a 50% breakdown point, and the extra self-weight protects nodes whose neighbourhood is mostly Byzantine.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    S = np.sqrt(np.concatenate([x[None], X], 0))     # (n, m, C)
    wts = np.ones(k + 1)
    wts[0] = 1.0 + 0.25 * k
    y = np.median(S, 0)
    for _ in range(10):
        d = np.linalg.norm(S - y[None], axis=2)      # (n, m)
        inv = wts[:, None] / np.maximum(d, 1e-6)
        y = (inv[..., None] * S).sum(0) / inv.sum(0)[..., None]
    return san(y ** 2)
```

### Candidate RowMADSuspicion
This rule filters outliers row by row. A neighbour's row is accepted when its L1 distance to the coordinate-wise median is within twice the median distance on that row. Each neighbour's rejection rate is accumulated into a suspicion score that persists across iterations but reacts at once to a change in behaviour. Accepted rows are averaged with the node's own state, each weighted by the neighbour's trust.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    med = np.median(np.concatenate([x[None], X], 0), 0)
    r = np.abs(X - med[None]).sum(2)                 # (k, m)
    r0 = np.abs(x - med).sum(1)                      # (m,)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rowacc = (r <= 2.0 * sc[None]).astype(float)     # (k, m)
    frac = 1.0 - rowacc.mean(1)                      # rejection rate (k,)
    susp = state.get('susp')
    if susp is None or susp.shape != frac.shape:
        susp = frac.copy()
    else:
        susp = np.maximum(frac, 0.8 * susp + 0.2 * frac)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    W = trust[:, None] * rowacc                      # (k, m)
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
```
