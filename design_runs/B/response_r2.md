### Candidate B2.1_TrustCore
This combines the two strongest round-1 ideas. A Hellinger-distance medoid core (from ConsistentCore) gives each node a soft membership weight, and the core's weighted centre plus the node's own labels give an argmax-agreement test (from ReputationGossip). The product of these is kept as a reputation that drops at once and recovers slowly, and the output is a trust-weighted mean with a floor on the node's own weight.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    n = k + 1
    S = np.sqrt(Z)
    H = np.sqrt(np.maximum(0.5 * ((S[:, None] - S[None]) ** 2).sum(-1), 0.0)).mean(-1)  # (n,n) in [0,1]
    vv = np.concatenate([[0.0], viol])
    Dm = H + 5.0 * (vv[:, None] + vv[None, :])
    np.fill_diagonal(Dm, 0.0)
    q = max(1, (n + 1) // 2 - 1)
    sc = np.sort(Dm, 1)[:, 1:q + 1].mean(1)
    sc[0] *= 0.9
    med = int(np.argmin(sc))
    d = Dm[med]
    r = max(float(np.sort(d)[min(q, n - 1)]), 0.02)
    w_core = 1.0 / (1.0 + (d / (2.0 * r)) ** 4)
    w_core[med] = 1.0
    center = (w_core[:, None, None] * Z).sum(0) / w_core.sum()
    lab = Z.argmax(-1)
    lab_c = center.argmax(-1)
    lab_s = lab[0]
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab_s[None]).mean(1))
    lo, hi = 2.5 / C, 5.0 / C
    w_agree = np.clip((agree - lo) / (hi - lo), 0.0, 1.0)
    w_new = (w_core * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    w_self = max(0.5, float(w_core[0]))
    out = (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + trust.sum())
    return out
```

### Candidate B2.2_EvidenceLedger
This is a sequential-testing view of reputation. At each iteration, every neighbour is scored against the leave-one-out median of all the other messages, using both distance and argmax agreement (also checked against the node's own labels). The score is added to a per-neighbour log-evidence ledger, and trust is a sigmoid of that ledger. Bad evidence counts three times as much as good evidence, and good evidence is capped, so a neighbour that turns Byzantine late loses trust within one or two steps.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    n = k + 1
    lab = Z.argmax(-1)
    d = np.empty(n)
    a = np.empty(n)
    for j in range(n):
        c = np.median(np.delete(Z, j, 0), 0)
        d[j] = np.abs(Z[j] - c).sum(-1).mean()
        a[j] = (lab[j] == c.argmax(-1)).mean()
    a_self = (lab == lab[0][None]).mean(1)
    a = 0.5 * (a + a_self)
    ref = max(float(np.median(d)), 0.02)
    g_dist = np.clip(2.0 - d / ref, -1.0, 1.0)
    g_agree = np.clip((a - 3.0 / C) / (2.0 / C), -1.0, 1.0)
    g = np.minimum(g_dist, g_agree)[1:] - viol / 0.05
    g = np.clip(g, -3.0, 1.0)
    L = state.get('L')
    if L is None or len(L) != k:
        L = np.zeros(k)
    L = np.clip(L + np.where(g < 0, 3.0 * g, g), -10.0, 4.0)
    state['L'] = L
    trust = 1.0 / (1.0 + np.exp(-3.0 * (L - 0.5)))
    return (xs + (trust[:, None, None] * P).sum(0)) / (1.0 + trust.sum())
```

### Candidate B2.3_TwoLevelRobustMean
Robustness is applied at two levels. At the node level, a reputation (distance, agreement and simplex violation, with memory) decides how much each neighbour counts overall. At the sample level, each row's weight is also reduced smoothly by its L1 distance to the row-wise median, measured relative to that row's own spread. This keeps out subtle attacks that only change some rows, while a wide margin keeps confident honest minorities from being removed.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    med = np.median(Z, 0)
    drow = np.abs(Z - med[None]).sum(-1)              # (n, m)
    dist = drow.mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist / (2.0 * d_ref)) ** 4)
    lab = Z.argmax(-1)
    lab_c = med.argmax(-1)
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab[0][None]).mean(1))
    w_agree = np.clip((agree - 2.5 / C) / (2.5 / C), 0.0, 1.0)
    w_new = (w_dist * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tr = np.concatenate([[1.0], trust])
    s = np.median(drow, 0) + 0.05                      # (m,)
    u = 1.0 / (1.0 + (drow / (3.0 * s[None])) ** 2)
    W = tr[:, None] * u
    W[0] = np.maximum(W[0], 0.5)
    return (W[:, :, None] * Z).sum(0) / W.sum(0)[:, None]
```

### Candidate B2.4_SemanticAlignment
This adds a new test for class meaning. For each neighbour, the node builds the soft confusion matrix between that neighbour's predictions and a robust reference (half the node's own state, half the median). An honest neighbour uses the same class names, so each row of the matrix peaks on its diagonal. Any relabelling, permutation or mixing attack moves the mass off the diagonal, whatever form it takes. This alignment score is combined with distance and simplex-validity weights in a memory-based reputation. The node's own weight goes up or down with how well it agrees with its trusted neighbours, so honest nodes with noisy labels move toward the consensus.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    center = np.median(Z, 0)
    ref = 0.5 * (xs + center)
    M = np.einsum('kmc,md->kcd', P, ref)               # (k, C, C) soft confusion
    rowsum = M.sum(2)
    diag = np.einsum('kcc->kc', M)
    rmax = M.max(2)
    ratio = diag / np.maximum(rmax, 1e-12)
    mass = rowsum / np.maximum(rowsum.sum(1, keepdims=True), 1e-12)
    align = (mass * ratio).sum(1)
    w_align = np.clip((align - 0.4) / 0.4, 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_new = w_align * w_dist * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum > 1e-6:
        nb = (trust[:, None, None] * P).sum(0) / tsum
        s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
        w_self = 0.5 + 0.5 * s_agree
    else:
        w_self = 1.0
    return (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)
```
