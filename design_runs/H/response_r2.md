### Candidate HuberIRLSAnchored
Anchor an M-estimator of location at the node's own state instead of a median, so the centre is never pulled toward Byzantine values. Scale is set robustly from the MAD of the distances to the current centre, and Huber weights are recomputed each iteration. This keeps the statistical efficiency of a mean for honest, similar messages while bounding the influence of any single message.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    z = P[0].copy()
    for _ in range(8):
        d = np.linalg.norm(P - z, axis=1)
        s = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9
        c = 1.345 * max(s, 1e-6 * (1.0 + np.median(d)))
        w = np.minimum(1.0, c / np.maximum(d, 1e-12))
        z_new = (w[:, None] * P).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-8 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)
```

### Candidate AnnealedMedianBand
Build a robust reference as the coordinate-wise median of self and neighbours, then accept only messages within a distance band around it. The band is wide early (exploration while the consensus is far from converged) and tightens over iterations, following the annealing idea of BALANCE but using a median reference that does not depend on x_self. The accepted messages, including self, are averaged.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    scale = np.median(d) + 1e-12
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    keep = d <= thr
    keep[0] = True
    out = P[keep].mean(0)
    return out.reshape(x_self.shape)
```

### Candidate SimplexTVReputation
Messages are probability matrices, so compare them with a simplex-native distance instead of the Euclidean norm. Each message is clipped and row-normalised, then its mean row-wise total-variation distance to the coordinate-wise median is computed. A per-slot reputation is stored in `state`: it is decayed when a neighbour is repeatedly an outlier and recovers when it behaves, so persistent attackers lose weight across iterations while noisy but honest nodes are only downweighted, not removed. The output is the reputation-weighted mean with self weight 1.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    C = x_self.shape[-1]
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None).reshape(k + 1, -1, C)
    s = P.sum(-1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-12), 1.0 / C)
    ref = np.median(P, 0)
    tv = 0.5 * np.abs(P - ref).sum(-1).mean(-1)
    rep = state.get('rep')
    if rep is None or len(rep) != k:
        rep = np.ones(k)
    rep = np.asarray(rep, dtype=float)
    u = tv[1:] / (np.median(tv[1:]) + 1e-12)
    good = np.exp(-np.maximum(u - 1.5, 0.0) ** 2)
    rep = 0.7 * rep + 0.3 * good
    state['rep'] = rep
    tau = np.median(tv) + 1e-12
    w = rep * np.minimum(1.0, tau / np.maximum(tv[1:], 1e-12))
    Xn = P[1:]
    num = P[0] + (w[:, None, None] * Xn).sum(0)
    den = 1.0 + w.sum()
    out = num / den
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
```

### Candidate GeometricMedianFromClip
Combine the two strongest existing mechanisms in sequence. First, self-centred norm clipping (ClippedGossip) bounds each neighbour's deviation from x_self by a robust quantile radius. Then a Weiszfeld geometric-median refinement is run on the clipped set, which removes the residual bias of the clipped mean under ALIE-type shifts. Iteration is capped so the call stays well under the time limit.

```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    xs = x_self.reshape(-1)
    D = X_nb.reshape(k, -1) - xs
    nrm = np.linalg.norm(D, axis=1)
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    Q = np.concatenate([xs[None], xs + scale[:, None] * D])
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    z = Q.mean(0)
    for _ in range(20):
        d = np.linalg.norm(Q - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * Q).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)
```
