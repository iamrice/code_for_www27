### Candidate C5_1_WeightedMedianAnchoredAgreement
Refines the two best rules (C3.2/C4.1). The peer-agreement reputation stays the same, but the row-gating anchor becomes a reputation-weighted coordinate-wise median instead of a plain median. The residual scale is also computed only over trusted nodes. Together these keep the anchor honest even when distrusted neighbours are a local majority or the degree is small, and C4.1's self-distrust on outlier rows is kept.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    def wmedian(P, wts):
        idx = np.argsort(P, axis=0, kind='stable')
        Ps = np.take_along_axis(P, idx, 0)
        cw = np.cumsum(wts[idx], 0)
        j = (cw < 0.5 * cw[-1][None]).sum(0)
        j = np.minimum(j, P.shape[0] - 1)
        return np.take_along_axis(Ps, j[None], 0)[0]

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    P = np.concatenate([x[None], X], 0)
    n = k + 1
    L = P.argmax(2)
    A = (L[:, None, :] == L[None, :, :]).mean(2)
    D = 0.5 * np.abs(P[:, None] - P[None]).sum(3).mean(2)
    S = 0.5 * A + 0.5 * (1.0 - D)
    So = S.copy()
    np.fill_diagonal(So, np.nan)
    peer = np.nanmedian(So, axis=0) if n > 1 else np.ones(n)
    score = 0.5 * S[0, 1:] + 0.5 * peer[1:]
    ref = max(float(peer[0]), float(score.max())) + 1e-12
    rel = score / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    # reputation-weighted median anchor
    wts = np.concatenate([[1.0], np.maximum(w, 0.02)])
    med = wmedian(P, wts)
    r = np.abs(X - med[None]).sum(2)                          # (k, m)
    r0 = np.abs(x - med).sum(1)                               # (m,)
    R = np.concatenate([r0[None], r], 0)
    mask = np.concatenate([[True], w >= 0.5])
    Rm = np.where(mask[:, None], R, np.nan)
    sc = np.nanmedian(Rm, axis=0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = w[:, None] * rw
    if k >= 3:
        ws = np.clip((3.0 * sc - r0) / (2.0 * sc), 0.35, 1.0)
        ws = np.where(W.sum(0) >= 1.5, ws, 1.0)
    else:
        ws = np.ones_like(r0)
    out = (ws[:, None] * x + (W[..., None] * X).sum(0)) / (ws + W.sum(0))[..., None]
    return san(out)
```

### Candidate C5_2_QualityTemperedReputation
Keeps C3.2's Byzantine reputation and row gating, then adds a second, continuous *quality* weight for every node, the node itself included. The quality weight is a softmax-tempered measure of how often the node's argmax agrees with the robust (median) consensus label. It aims at the remaining gap to the oracle, which comes from noisy-label honest nodes: they are down-weighted without being excluded, and a noisy node trusts its own state less, so it moves to consensus faster.
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
    P = np.concatenate([x[None], X], 0)
    n = k + 1
    L = P.argmax(2)
    A = (L[:, None, :] == L[None, :, :]).mean(2)
    Ao = A.astype(float).copy()
    np.fill_diagonal(Ao, np.nan)
    peer = np.nanmedian(Ao, axis=0)
    score = 0.5 * A[0, 1:] + 0.5 * peer[1:]
    ref = max(float(peer[0]), float(score.max())) + 1e-12
    rel = score / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    # robust anchor (self-padded for small degree)
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    cons = med.argmax(1)
    # quality: agreement with robust consensus label, softmax-tempered
    agr = (L == cons[None]).mean(1)                           # (n,)
    trusted = np.concatenate([[True], w >= 0.5])
    amax = float(agr[trusted].max())
    q = np.exp(np.minimum(agr - amax, 0.0) / 0.08)
    r = np.abs(X - med[None]).sum(2)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = (w * q[1:])[:, None] * rw
    ws0 = max(float(q[0]), 0.3) if k >= 2 else 1.0
    out = (ws0 * x + (W[..., None] * X).sum(0)) / (ws0 + W.sum(0))[..., None]
    return san(out)
```

### Candidate C5_3_GeoMedianCenteredClipping
This candidate brings in a new robust anchor. For each row it computes a reputation-weighted geometric median of the simplex points (vectorised Weiszfeld), which has a 50% breakdown point and is rotation-invariant, so it does not depend on the coordinate-wise structure the attacks exploit. It then applies centred clipping around that anchor: deviations are clipped to the median trusted radius and averaged with reputation weights. This recovers mean-like efficiency on honest nodes while bounding the influence of any single message.
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
    P = np.concatenate([x[None], X], 0)
    n = k + 1
    L = P.argmax(2)
    A = (L[:, None, :] == L[None, :, :]).mean(2)
    D = 0.5 * np.abs(P[:, None] - P[None]).sum(3).mean(2)
    S = 0.5 * A + 0.5 * (1.0 - D)
    So = S.copy()
    np.fill_diagonal(So, np.nan)
    peer = np.nanmedian(So, axis=0)
    score = 0.5 * S[0, 1:] + 0.5 * peer[1:]
    ref = max(float(peer[0]), float(score.max())) + 1e-12
    rel = score / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    # weighted geometric median per row (Weiszfeld)
    gw = np.concatenate([[1.0], 0.25 + 0.75 * w])             # (n,)
    if k <= 3:
        gw[0] = 2.0
    z = np.median(P, 0)
    for _ in range(8):
        d = np.sqrt(((P - z[None]) ** 2).sum(2)) + 1e-6       # (n, m)
        a = gw[:, None] / d
        z = (a[..., None] * P).sum(0) / a.sum(0)[..., None]
    # centred clipping around the geometric median
    dev = P - z[None]
    dn = np.sqrt((dev ** 2).sum(2))                           # (n, m)
    mask = np.concatenate([[True], w >= 0.5])
    tau = np.nanmedian(np.where(mask[:, None], dn, np.nan), axis=0) + 0.02
    c = np.minimum(1.0, tau[None] / np.maximum(dn, 1e-12))   # (n, m)
    v = np.concatenate([[1.0], w])
    V = v[:, None] * c
    out = z + (V[..., None] * dev).sum(0) / v.sum()
    return san(out)
```

### Candidate C5_4_LeaveOneOutConsistencyTrimmed
This candidate uses a new reputation signal. Each node is scored by how well it agrees with the leave-one-out consensus, meaning the coordinate-wise median of everyone except itself. This stops a node, or a colluding group, from pulling the reference it is judged against, and it applies the same test to the node itself. The temporally memorised reputation decides which nodes are kept. Aggregation is a coordinate-wise trimmed mean over the kept set, so a node that fools the reputation still has bounded influence.
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
    P = np.concatenate([x[None], X], 0)
    n = k + 1
    sc = np.zeros(n)
    for j in range(n):
        others = np.delete(P, j, 0)
        if j != 0 and k <= 3:
            others = np.concatenate([x[None], others], 0)     # self-pad small degree
        mj = np.median(others, 0)
        agr = float((P[j].argmax(1) == mj.argmax(1)).mean())
        tv = float(0.5 * np.abs(P[j] - mj).sum(1).mean())
        sc[j] = 0.5 * agr + 0.5 * (1.0 - tv)
    ref = max(float(sc[0]), float(sc[1:].max())) + 1e-12
    rel = sc[1:] / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    keep = np.concatenate([[True], w >= 0.5])
    K = P[keep]
    nk = K.shape[0]
    if nk == 1:
        return x
    b = 1 if nk >= 5 else 0
    Ks = np.sort(K, axis=0)
    tm = Ks[b:nk - b].mean(0)
    # partial weight for borderline neighbours: blend them in softly
    part = (w > 0.0) & (w < 0.5)
    if part.any():
        pw = w[part]
        tm = (nk * tm + (pw[:, None, None] * X[part]).sum(0)) / (nk + pw.sum())
    return san(tm)
```
