### Candidate AnnealedSoftBandGeoMedian
Keeps the annealed distance band from H2.2 but replaces the hard keep/drop cut with a soft weight that falls off linearly across the band edge, then takes a weighted geometric median (Weiszfeld) of the retained messages instead of their plain mean. This limits the influence of messages near the band boundary, which the hard cut treats inconsistently.
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    scale = np.median(d) + 1e-12
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr - d) / (0.5 * thr + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    z = (w[:, None] * P).sum(0) / w.sum()
    for _ in range(20):
        dz = np.linalg.norm(P - z, axis=1)
        ww = w / np.maximum(dz, 1e-8)
        z_new = (ww[:, None] * P).sum(0) / ww.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
```

### Candidate LogPoolTrimmedMean
A new mechanism: pool in the log domain. Each message is renormalised per row and log-transformed, then the 20% extremes are trimmed per coordinate and the rest averaged (geometric pooling). Log pooling makes a single message that assigns near-zero probability to the consensus class very costly, so it resists corrupted or noisy rows while keeping the output a probability vector.
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    n = k + 1
    Q = np.concatenate([x_self[None], X_nb]).reshape(n, m, C).astype(float)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    Q = np.clip(Q, 1e-6, None)
    Q = Q / Q.sum(-1, keepdims=True)
    L = np.log(Q)
    b = int(math.floor(0.2 * n)) if n >= 5 else 0
    S = np.sort(L, axis=0)
    M = S[b:n - b].mean(0)
    out = np.exp(M - M.max(-1, keepdims=True))
    out = out / out.sum(-1, keepdims=True)
    out = np.nan_to_num(out, nan=1.0 / C, posinf=1.0 / C, neginf=1.0 / C)
    return out.reshape(x_self.shape)
```

### Candidate DirectionGatedClip
Combines self-centred norm clipping (ClippedGossip) with a directional gate: each message's displacement from the node's own state is weighted by how well its direction agrees with the coordinate-wise median displacement. Messages pushing the state in a direction most others do not support are down-weighted, which targets sign-flipped and label-permuted messages without relying on their specific form.
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    xs = np.nan_to_num(x_self.reshape(-1).astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    Xf = np.nan_to_num(X_nb.reshape(k, -1).astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    D = Xf - xs
    nrm = np.linalg.norm(D, axis=1)
    med = np.median(np.concatenate([xs[None], Xf]), 0)
    dm = med - xs
    cos = (D @ dm) / (nrm * np.linalg.norm(dm) + 1e-12)
    gate = np.clip(cos, 0.0, 1.0) ** 2
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    w = gate * scale
    out = xs + (w[:, None] * D).sum(0) / (k + 1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
```

### Candidate DecayedTrustBand
Uses the state across iterations: each neighbour carries a reputation that is an exponential moving average of how typical its distance to the coordinate-wise median is, relative to the median distance. The reputation then scales a norm-clipped, self-anchored weighted average. Persistent outliers lose influence over the T rounds, while a single noisy-but-honest round is not decisive.
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    s = np.median(d) + 1e-12
    u = d[1:] / s
    evidence = np.exp(-np.maximum(u - 1.0, 0.0) ** 2)
    rep = state.get('rep')
    if rep is None or len(rep) != k:
        rep = np.ones(k)
    rep = 0.8 * np.asarray(rep, dtype=float) + 0.2 * evidence
    state['rep'] = rep
    w = rep * np.minimum(1.0, (1.5 * s) / np.maximum(d[1:], 1e-12))
    num = P[0] + (w[:, None] * P[1:]).sum(0)
    out = num / (1.0 + w.sum())
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
```
