### Candidate H5.1_PersistentBandClippedMix
Combines the persistent suspicion memory and band trust of H4.3 with ClippedGossip-style norm clipping of the accepted deviations from the node's own state, so that even trusted but large deviations cannot move the output far in one iteration.
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
    x0 = P[0]
    D = P[1:] - x0
    nrm = np.linalg.norm(D, axis=1)
    tau = np.median(nrm) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    z = x0 + (trust * scale)[:, None].__mul__(D).sum(0) / (1.0 + trust.sum())
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
```

### Candidate H5.2_WelschIRLSAnnealed
A robust location estimate: a Welsch (Gaussian-kernel) M-estimator solved by a few iterative reweighted averages, started from the coordinate-wise median. The kernel scale is a MAD-based estimate of the spread of distances, inflated early in consensus and annealed towards the honest spread later; the node's own state always keeps unit weight.
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.median(P, 0)
    d0 = np.linalg.norm(P - z, axis=1)
    med0 = np.median(d0)
    mad = np.median(np.abs(d0 - med0))
    sig = (1.4826 * mad + 0.5 * med0 + 1e-12) * (1.0 + math.exp(-2.0 * t / T))
    for _ in range(10):
        d = np.linalg.norm(P - z, axis=1)
        u = np.minimum(d / sig, 1e6)
        w = np.exp(-0.5 * u * u)
        w[0] = 1.0
        z_new = (w[:, None] * P).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
```

### Candidate H5.3_PerSampleRowTrust
Changes the granularity of the trust decision: instead of judging a whole neighbour message, each row (sample) of each message is judged against the per-sample median across messages, after renormalising rows to probability vectors. A neighbour that is noisy or manipulated on some public samples is discounted only there, while its agreeing rows still contribute. Row-wise L1 deviations are compared with the per-row median deviation, with the annealed band as in H4.1.
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / P.shape[-1])
    ref = np.median(P, 0)
    d = np.abs(P - ref).sum(-1)
    scale = np.median(d, 0) + 1e-9
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr[None, :] - d) / (0.5 * thr[None, :] + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    out = (w[..., None] * P).sum(0) / w.sum(0)[:, None]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, 0.0, None)
    rs = out.sum(-1, keepdims=True)
    out = np.where(rs > 1e-12, out / np.maximum(rs, 1e-12), 1.0 / out.shape[-1])
    return out.reshape(x_self.shape)
```

### Candidate H5.4_LargestGapCutWithBanMemory
A new mechanism that avoids a fixed or annealed threshold: among the sorted distances to the coordinate-wise median, the rule looks for the largest multiplicative gap in the lower half-plus of the distribution and cuts there only if the gap is clear (ratio above 1.3), otherwise it keeps everyone. Neighbours cut in previous iterations are persistently down-weighted through a per-neighbour memory, so repeated cuts accumulate evidence while a single noisy iteration does not exclude an honest node.
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    keep = np.ones(k + 1, dtype=bool)
    if k + 1 >= 3:
        order = np.argsort(d)
        ds = d[order]
        lo = (k + 1) // 2
        ratios = (ds[lo:] + 1e-12) / (ds[lo - 1:k] + 1e-12)
        j = lo + int(np.argmax(ratios))
        if ratios.max() > 1.3:
            keep = np.zeros(k + 1, dtype=bool)
            keep[order[:j]] = True
    keep[0] = True
    ban_prev = state.get('ban')
    if ban_prev is None or len(ban_prev) != k:
        ban_prev = np.zeros(k)
    cut_nb = (~keep[1:]).astype(float)
    ban = 0.6 * np.asarray(ban_prev, dtype=float) + 0.4 * cut_nb
    state['ban'] = ban
    wnb = keep[1:].astype(float) * np.clip(1.0 - ban, 0.1, 1.0)
    w = np.concatenate([[1.0], wnb])
    z = (w[:, None] * P).sum(0) / w.sum()
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
```
