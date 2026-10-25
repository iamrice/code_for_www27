### Candidate HellingerBandMixture
Measures disagreement in square-root (Hellinger) space, where the distance between two probability rows is a bounded, principled divergence, and then forms the output as a weighted mixture of the probability rows rather than an average in raw space. Each message is accepted with a soft band that anneals over iterations, and the mixture stays on the simplex by construction.

```python
def aggregate(x_self, X_nb, t, T, state):
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / P.shape[-1])
    S = np.sqrt(P)
    ref = np.median(S, 0)
    d = np.sqrt(((S - ref) ** 2).sum(-1)).mean(-1)
    scale = np.median(d) + 1e-12
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr - d) / (0.5 * thr + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    out = (w[:, None, None] * P).sum(0) / w.sum()
    return out.reshape(x_self.shape)
```

### Candidate NeighbourhoodAgreementGeoMed
Weights each message by its local agreement with all other messages, using a Gaussian kernel on pairwise distances (a Krum-like density score that does not rely on a single reference point). The weighted geometric median of the messages is then computed, which keeps the estimate robust when a Byzantine cluster is large.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    n = k + 1
    if n == 1:
        return x_self
    G = P @ P.T
    sq = np.diag(G)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))
    off = D[~np.eye(n, dtype=bool)]
    h = np.median(off) + 1e-12
    ht = h * (1.0 + math.exp(-2.0 * t / T))
    K = np.exp(-(D / ht) ** 2)
    agree = (K.sum(1) - 1.0) / max(n - 1, 1)
    w = agree ** 2
    w[0] = 1.0
    z = (w[:, None] * P).sum(0) / w.sum()
    for _ in range(20):
        d = np.linalg.norm(P - z, axis=1)
        a = w / np.maximum(d, 1e-8)
        z_new = (a[:, None] * P).sum(0) / a.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
```

### Candidate PersistentSuspicionBand
Keeps a per-neighbour suspicion score in the persistent state, an exponential moving average of how far each slot's message lies from the robust centre. Slots that are persistently far are down-weighted even if they occasionally look plausible, which handles attackers that change behaviour over iterations. Combined with an annealed soft band on the current deviation, the output is a trust-weighted mean.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    r = d / (np.median(d) + 1e-12)
    s_prev = state.get('s')
    if s_prev is None or len(s_prev) != k:
        s_prev = np.zeros(k)
    s = 0.6 * np.asarray(s_prev, dtype=float) + 0.4 * r[1:]
    state['s'] = s
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    trust = band * np.exp(-np.maximum(s - 1.0, 0.0))
    w = np.concatenate([[1.0], trust])
    z = (w[:, None] * P).sum(0) / w.sum()
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
```

### Candidate CoordinateMADFilteredMean
Works coordinate by coordinate: each probability entry is compared with the coordinate-wise median in units of the scaled median absolute deviation, entries outside an annealed band are discarded, and the rest are averaged. This is a trimmed-mean style estimator that keeps the efficiency of averaging on coordinates where honest nodes agree, and falls back to the median where it has no inliers. The self value is always kept.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(P, 0)
    mad = 1.4826 * np.median(np.abs(P - med), 0) + 1e-6
    c = 1.5 + 2.0 * math.exp(-2.0 * t / T)
    inl = np.abs(P - med) <= c * mad
    inl[0] = True
    cnt = inl.sum(0)
    s = np.where(inl, P, 0.0).sum(0)
    out = s / np.maximum(cnt, 1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
```
