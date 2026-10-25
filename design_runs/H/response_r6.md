### Candidate HellingerPersistentTrust
Hybrid of the two strongest prior rules: row-normalised Hellinger distances to the coordinate-wise median give a per-round deviation score, a persistent suspicion memory penalises neighbours that stay far across iterations, and the annealed band gates the trust weights before a self-anchored mean.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, m, C).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / C)
    S = np.sqrt(P)
    ref = np.median(S, 0)
    raw = np.sqrt(((S - ref) ** 2).sum(-1)).mean(-1) / math.sqrt(2.0)
    r = raw / (np.median(raw) + 1e-12)
    s_prev = state.get('s')
    if s_prev is None or len(s_prev) != k:
        s_prev = np.zeros(k)
    s = 0.6 * np.asarray(s_prev, dtype=float) + 0.4 * r[1:]
    state['s'] = s
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    trust = band * np.exp(-1.5 * np.maximum(s - 1.0, 0.0))
    z = (P[0] + np.tensordot(trust, P[1:], axes=1)) / (1.0 + trust.sum())
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
```

### Candidate HuberIRLSWithBanMemory
Runs a few Huber-IRLS steps anchored at the node's own state with an annealed scale, then turns persistently down-weighted neighbours (weight below 0.5 in most recent rounds) into hard exclusions via a decaying flag memory, which adds a new ban mechanism on top of soft robust weights.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, m, C).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / C)
    P0, Nb = P[0], P[1:]
    b_prev = state.get('b')
    if b_prev is None or len(b_prev) != k:
        b_prev = np.zeros(k)
    dself = np.sqrt(((Nb - P0) ** 2).sum(axis=(-1, -2)) / m)
    c = (np.median(dself) + 1e-12) * (1.0 + math.exp(-2.0 * t / T))
    z = P0.copy()
    w = np.ones(k)
    for _ in range(4):
        dz = np.sqrt(((Nb - z) ** 2).sum(axis=(-1, -2)) / m)
        w = np.minimum(1.0, c / np.maximum(dz, 1e-12))
        z = (P0 + np.tensordot(w, Nb, axes=1)) / (1.0 + w.sum())
    flag = (w < 0.5).astype(float)
    b = 0.7 * np.asarray(b_prev, dtype=float) + 0.3 * flag
    state['b'] = b
    w_eff = w * (b < 0.5)
    z = (P0 + np.tensordot(w_eff, Nb, axes=1)) / (1.0 + w_eff.sum())
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
```

### Candidate MedoidAnchoredAnnealedTrim
Finds the medoid of the received messages plus self using pairwise Hellinger distances, then averages self with the neighbours nearest to that medoid; the kept fraction grows over iterations as honest estimates converge, so early rounds are strict and late rounds recover accuracy.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, m, C).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / C)
    F = np.sqrt(P).reshape(k + 1, -1)
    G = F @ F.T
    Dm = np.sqrt(np.maximum(2.0 * m - 2.0 * G, 0.0))
    med = int(np.argmin(Dm.sum(1)))
    dm = Dm[med, 1:]
    frac = 0.5 + 0.4 * t / max(T, 1)
    n_keep = max(1, int(round(frac * k)))
    keep = np.argsort(dm, kind='stable')[:n_keep]
    z = (P[0] + P[1:][keep].sum(0)) / (1.0 + n_keep)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
```

### Candidate ConfidenceWeightedBand
New precision-weighting idea: each message is weighted by the mean confidence (one minus normalised row entropy) of its predictions, so diffuse or noisy messages count less in the average, multiplied by the annealed distance band to reject outliers; a floor keeps noisy but honest nodes from being fully discarded.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, m, C).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / C)
    denom = max(math.log(C), 1e-12)
    H = -(P * np.log(P + 1e-12)).sum(-1) / denom
    conf = np.clip(1.0 - H.mean(-1), 0.0, 1.0)
    S = np.sqrt(P)
    ref = np.median(S, 0)
    raw = np.sqrt(((S - ref) ** 2).sum(-1)).mean(-1) / math.sqrt(2.0)
    r = raw / (np.median(raw) + 1e-12)
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    w = np.concatenate([[0.1 + conf[0]], band * (0.1 + conf[1:])])
    z = np.tensordot(w, P, axes=1) / (w.sum() + 1e-12)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
```
