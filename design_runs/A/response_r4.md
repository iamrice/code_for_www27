### Candidate A4_1_AnnealedSelfTrustRadiusClip
Refines the best rule (trust-scaled radius clipping with asymmetric reputation). The self-weight in the denominator is annealed from 1.0 to 0.5 over the iterations. Clipping and reputation already limit Byzantine influence, so later rounds can lean more on trusted neighbours, which should help consensus and noisy-label nodes, especially in the "none" scenario where we are furthest from the oracle.
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
    base = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = base.copy()
    else:
        r = np.where(base < r, 0.4 * r + 0.6 * base, 0.7 * r + 0.3 * base)
    state['rep'] = r
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    out = xs + (w[:, None, None] * D).sum(0) / (lam + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A4_2_RowwiseTrustClip
Keeps a global per-neighbour reputation, which captures who is reliable over time, but clips at the level of single samples (rows). Each row gets its own radius: a quantile across neighbours of that row's deviation. Rows where this node is wrong and everyone disagrees with it are then not over-clipped, while a small coordinated shift on rows where everyone agrees (ALIE-style) is still bounded.
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
    base = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = base.copy()
    else:
        r = np.where(base < r, 0.4 * r + 0.6 * base, 0.7 * r + 0.3 * base)
    state['rep'] = r
    rn = np.linalg.norm(D, axis=2)                      # (k, m)
    tau_row = np.quantile(rn, 0.25, axis=0) + 1e-12     # (m,)
    tau_jr = tau_row[None, :] * (1.0 + 2.0 * r[:, None] ** 2)
    sc = np.minimum(1.0, tau_jr / np.maximum(rn, 1e-12))
    gsc = np.minimum(1.0, tau * (1.0 + 2.0 * r ** 2) / np.maximum(nrm, 1e-12))
    w = (r * np.sqrt(gsc))[:, None] * sc                # (k, m)
    out = xs + (w[:, :, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A4_3_PeerCorroboratedRepClip
New idea: a neighbour's evidence comes from how close it is to this node and also from how well its peers corroborate it. The score is the median distance to all other received messages, self included. A noisy-label honest node then does not wrongly distrust good neighbours just because they disagree with it. Byzantines, as long as they are a minority locally, still look far from the honest majority. Reputation and trust-scaled clipping on the actual displacement are as before.
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
    if k >= 2:
        F = np.concatenate([xs[None], Y], 0).reshape(k + 1, -1)
        sq = (F * F).sum(1)
        G = F @ F.T
        Dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))
        dpeer = np.empty(k)
        for j in range(k):
            row = np.delete(Dm[j + 1], j + 1)
            dpeer[j] = np.median(row)
        s = 0.5 * nrm + 0.5 * dpeer
    else:
        s = nrm.copy()
    ts = float(np.quantile(s, 0.25)) + 1e-12
    base = np.minimum(1.0, ts / np.maximum(s, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = base.copy()
    else:
        r = np.where(base < r, 0.4 * r + 0.6 * base, 0.7 * r + 0.3 * base)
    state['rep'] = r
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A4_4_RelativeHedgeTrustClip
New idea: reputation as a relative multiplicative-weights (Hedge) learner. Each round, a neighbour suffers a loss equal to how far its log-distance exceeds the median log-distance. Its trust is exp(-eta * (average loss - best average loss)), so the most consistent neighbour always keeps trust 1. Neighbours that are persistently outlying decay exponentially, while honest heterogeneity costs only a bounded amount. The trust then drives the trust-scaled clipping radius.
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
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```
