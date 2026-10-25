### Candidate C6_1_AgreementRepSoftTV
Refines the two best rules (C3.2 and C4.1): node-level reputation from a mix of hard argmax agreement and soft TV similarity, scored against both self and the peer median, with a reputation memory that drops fast and recovers slowly. Row-wise gating against the coordinate-wise median stays. Self-distrust is milder and only kicks in when there are enough neighbours and enough trusted mass to take over, so noisy honest nodes get pulled toward consensus without exposing small-degree nodes.
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
    S = 0.65 * A + 0.35 * (1.0 - D)
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
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    r = np.abs(X - med[None]).sum(2)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = w[:, None] * rw
    if k >= 4:
        ws = np.clip((3.0 * sc - r0) / (2.0 * sc), 0.5, 1.0)
        ws = np.where(W.sum(0) >= 2.0, ws, 1.0)
    else:
        ws = np.ones_like(r0)
    out = (ws[:, None] * x + (W[..., None] * X).sum(0)) / (ws + W.sum(0))[..., None]
    return san(out)
```

### Candidate C6_2_ReputationWeightedLabelVote
New idea: aggregate in label space first. Each row gets a reputation-weighted plurality vote over the senders' argmax labels (self counts with weight 1). Senders whose argmax matches the winning label get full weight on that row; dissenters are down-weighted. An attacker therefore has to win the vote on a row before it can move that row's prediction, and argmax correctness is exactly what the objective measures. When the vote margin is weak, it falls back to plain trusted averaging.
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
    m, C = x.shape
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
    wts = np.concatenate([[1.0], w])                          # (n,)
    # row gating against coordinate-wise median
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    R = np.abs(P - med[None]).sum(2)                          # (n, m)
    sc = np.median(R, 0) + 0.05
    rw = np.clip((3.0 * sc[None] - R) / sc[None], 0.0, 1.0)
    rw[0] = 1.0
    G = wts[:, None] * rw                                     # (n, m)
    # weighted label vote
    onehot = (L[..., None] == np.arange(C)[None, None, :]).astype(float)  # (n, m, C)
    votes = (G[..., None] * onehot).sum(0)                    # (m, C)
    tot = votes.sum(1) + 1e-12
    win = votes.argmax(1)
    margin = votes.max(1) / tot                               # (m,)
    agree = (L == win[None]).astype(float)                    # (n, m)
    strength = np.clip((margin - 0.4) / 0.3, 0.0, 1.0)        # 0 -> no vote effect
    dis = 1.0 - 0.8 * strength                                # weight of dissenters
    F = agree + (1.0 - agree) * dis[None]
    Wn = G * F
    # never let self vanish entirely
    Wn[0] = np.maximum(Wn[0], 0.15)
    out = (Wn[..., None] * P).sum(0) / Wn.sum(0)[..., None]
    return san(out)
```

### Candidate C6_3_ContractionTestReputation
New idea: a temporal contraction test. Under honest gossip, the distance between a neighbour's message and our own state should shrink from one iteration to the next because states converge. A sender that stays far away and does not contract is suspicious, whether it sends fresh noise, a fixed adversarial target, or a drifting attack. The non-contraction flag is accumulated with max-memory and multiplies the agreement-based reputation from C3.2/C4.1. Neighbours that are already close, and so cannot contract further, are never flagged.
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
    # contraction test
    d = D[0, 1:]                                              # mean TV distance to self, (k,)
    prev = state.get('dprev')
    cfl = state.get('cfl')
    if cfl is None or cfl.shape != d.shape:
        cfl = np.zeros(k)
    if prev is not None and prev.shape == d.shape:
        ratio = d / (prev + 0.02)
        base = float(np.quantile(ratio, 0.25)) if k >= 3 else float(min(ratio.min(), 0.9))
        excess = np.clip((ratio - max(base, 0.5) - 0.1) / 0.3, 0.0, 1.0)
        dmed = float(np.median(d))
        far = np.clip((d - dmed) / (dmed + 0.02), 0.0, 1.0)
        far = np.maximum(far, np.clip((d - 0.3) / 0.2, 0.0, 1.0))
        flag = excess * far
        cfl = np.maximum(flag, 0.8 * cfl + 0.2 * flag)
    state['dprev'] = d.copy()
    state['cfl'] = cfl
    rel = rel * (1.0 - 0.6 * cfl)
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    r = np.abs(X - med[None]).sum(2)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
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

### Candidate C6_4_ReputationIRLSBiweight
An M-estimator view. The center starts at a reputation-weighted coordinate median, then a few rounds of iteratively reweighted least squares refine it. Each round recomputes per-row L1 residuals, sets the scale from trusted senders only, and gives each sender a Tukey biweight times its node reputation. Gross outliers get exactly zero influence, while close senders are smoothly averaged. Self keeps a small minimum weight so the estimate is always defined.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    def wmedian(Q, wv):
        idx = np.argsort(Q, axis=0, kind='stable')
        Qs = np.take_along_axis(Q, idx, 0)
        cw = np.cumsum(wv[idx], 0)
        j = (cw < 0.5 * cw[-1][None]).sum(0)
        j = np.minimum(j, Q.shape[0] - 1)
        return np.take_along_axis(Qs, j[None], 0)[0]

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
    wts = np.concatenate([[1.0], w])                          # (n,)
    selfw = 1.0 if k >= 3 else 2.0
    wts[0] = selfw
    center = wmedian(P, np.maximum(wts, 0.02))
    trusted = np.concatenate([[True], w >= 0.5])
    Wt = None
    for _ in range(3):
        R = np.abs(P - center[None]).sum(2)                   # (n, m)
        Rm = np.where(trusted[:, None], R, np.nan)
        s = np.nanmedian(Rm, axis=0) + 0.05                   # (m,)
        u = R / (3.0 * s[None])
        bw = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        Wt = wts[:, None] * bw
        Wt[0] = np.maximum(Wt[0], 0.3)
        center = (Wt[..., None] * P).sum(0) / Wt.sum(0)[..., None]
    return san(center)
```
