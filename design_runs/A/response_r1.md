### Candidate HellingerGeoMedian
Project every message back onto the probability simplex (non-finite -> 0, clip, renormalise; all-zero rows -> uniform), embed in Hellinger space (square-root probabilities, where all honest messages lie on a bounded sphere), and compute the geometric median of self + neighbours with Weiszfeld iterations. The node then takes a self-centred step toward that median, capped at the median neighbour distance, and maps back by squaring.

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
    P = np.sqrt(np.concatenate([xs[None], Y], 0))
    n = P.shape[0]
    F = P.reshape(n, -1)
    z = np.median(F, 0)
    for _ in range(30):
        d = np.linalg.norm(F - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * F).sum(0) / w.sum()
        if np.linalg.norm(z_new - z) < 1e-7:
            z = z_new
            break
        z = z_new
    ps = F[0]
    D = z - ps
    nd = np.linalg.norm(D)
    dn = np.linalg.norm(F[1:] - ps, axis=1)
    tau = float(np.median(dn)) + 1e-12
    step = D * min(1.0, tau / max(nd, 1e-12))
    out = (ps + step).reshape(xs.shape) ** 2
    out = out + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate AgreementTrust
Each neighbour gets a trust score equal to the fraction of samples where its argmax matches a reference built half from the node's own state and half from the coordinate-wise median. The reference leans on the node's own state, so a colluding local majority cannot capture it, and the median half helps nodes with noisy labels. Trust is kept in memory so it drops fast and recovers slowly, and weights are set relative to the best-agreeing neighbour, with a floor above chance level so near-random or permuted messages get zero weight.

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
    C = xs.shape[-1]
    Z = np.concatenate([xs[None], Y], 0)
    med = np.median(Z, 0)
    ref = (0.5 * xs + 0.5 * med).argmax(-1)
    agree = (Y.argmax(-1) == ref[None]).mean(1)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = agree
    else:
        trust = np.minimum(agree, 0.5 * prev + 0.5 * agree)
    state['trust'] = trust
    amax = float(trust.max())
    lo = max(2.0 / C, 0.5 * amax)
    hi = max(lo + 0.05, 0.85 * amax)
    w = np.clip((trust - lo) / (hi - lo), 0.0, 1.0)
    out = (xs + (w[:, None, None] * Y).sum(0)) / (1.0 + w.sum())
    return out
```

### Candidate PerSampleTrimmedHellinger
Robust selection is done separately for each public sample, not per message. For each sample, after projecting onto the simplex and embedding in Hellinger space, the rule keeps the closest half of {self, neighbours} to a reference that is half own state and half coordinate-wise median, then averages them. An attacker therefore has to win sample by sample, and its influence on any one row is bounded.

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
    A = np.sqrt(np.concatenate([xs[None], Y], 0))       # (n, m, C)
    n = A.shape[0]
    med = np.median(A, 0)
    ref = 0.5 * med + 0.5 * A[0]
    d = np.linalg.norm(A - ref[None], axis=2)           # (n, m)
    keep = max(1, (n + 1) // 2)
    idx = np.argsort(d, axis=0, kind='stable')[:keep]   # (keep, m)
    sel = np.take_along_axis(A, idx[:, :, None], axis=0)
    z = sel.mean(0)
    out = z ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate ReputationClippedGossip
This is self-centred clipped gossip run on simplex-projected messages, plus a per-neighbour reputation kept in memory. The reputation is an exponential moving average of how much each neighbour's message had to be clipped, so neighbours that are far away again and again lose influence. The clipping radius adapts to a lower quantile of neighbour distances, and normalising by total reputation keeps any one Byzantine neighbour's displacement within that radius.

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
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = scale.copy()
    else:
        r = 0.6 * r + 0.4 * scale
    state['rep'] = r
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```
