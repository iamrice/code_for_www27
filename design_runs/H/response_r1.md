### Candidate MedoidFilteredMean
Compute the coordinate-free robust centre of {self, neighbours} as the coordinate-wise median, rank all messages by their Frobenius distance to that centre, and average only the closest 60% (always at least one). This is a trimmed-mean-style filter that adapts its cut to the local message count and does not depend on the attack's magnitude.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    S = np.concatenate([x_self[None], X_nb])
    med = np.median(S, 0)
    d = np.linalg.norm((S - med).reshape(k + 1, -1), axis=1)
    keep = max(1, int(np.ceil(0.6 * (k + 1))))
    idx = np.argsort(d, kind='stable')[:keep]
    return S[idx].mean(0)
```

### Candidate TrustEMAGossip
Each neighbour slot keeps a persistent trust weight in `state`, updated every iteration from a soft gate on its distance relative to the median distance, smoothed by an exponential moving average over iterations. Persistent low trust lets Byzantine neighbours that behave well early, or noisy honest neighbours, be discounted over time without a hard cut.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    D = np.linalg.norm((X_nb - x_self).reshape(k, -1), axis=1)
    z = D / (np.median(D) + 1e-12)
    w_inst = 1.0 / (1.0 + (z / 1.5) ** 4)
    trust = state.get('trust')
    if trust is None or len(trust) != k:
        trust = np.ones(k)
    trust = 0.5 * np.asarray(trust, dtype=float) + 0.5 * w_inst
    state['trust'] = trust
    return (x_self + np.tensordot(trust, X_nb, axes=1)) / (1.0 + trust.sum())
```

### Candidate GeometricMedianCentre
Replace the mean by the geometric median of the flattened {self, neighbours} set, computed with Weiszfeld iterations started from the coordinate-wise median. The geometric median has a breakdown point of one half and is rotation-invariant, so it bounds the influence of any single far-away message regardless of its coordinate pattern.

```python
def aggregate(x_self, X_nb, t, T, state):
    P = np.concatenate([x_self[None], X_nb]).reshape(len(X_nb) + 1, -1)
    z = np.median(P, 0)
    for _ in range(30):
        d = np.linalg.norm(P - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * P).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)
```

### Candidate RowwiseConsensusWeighted
Work per sample rather than per message: project every message row onto the simplex (clip negatives, renormalise), then weight each row by its confidence (one minus normalised entropy) times a Gaussian-like gate on its total-variation deviation from the per-row median. Noisy honest nodes, which give high-entropy rows, and outlier rows from attackers are both down-weighted sample by sample.

```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]
    P = np.concatenate([x_self[None], X_nb])
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    s = P.sum(-1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-12), 1.0 / C)
    H = -(P * np.log(P + 1e-12)).sum(-1)
    conf = np.clip(1.0 - H / math.log(C), 0.0, 1.0)
    med = np.median(P, 0)
    dev = 0.5 * np.abs(P - med).sum(-1)
    tau = np.median(dev) + 1e-3
    wgt = conf * np.exp(-dev / tau) + 1e-6
    num = (wgt[..., None] * P).sum(0)
    den = wgt.sum(0)[..., None]
    return num / den
```
