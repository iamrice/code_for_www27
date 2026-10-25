### Candidate C4_1_PeerAgreementSelfDistrust
Refines the best rule (C3.2): neighbour reputation now mixes hard argmax agreement with soft total-variation similarity (harder to game with one-hot or near-honest forgeries), keeps the min-type memory, and adds a new *self-distrust* mechanism: on rows where the node's own prediction is far from the robust median (typical for noisy-label honest nodes), its self-weight is reduced so it leans on trusted neighbours. Self-distrust is only enabled when k >= 3 so small neighbourhoods cannot be hijacked.
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
    P = np.concatenate([x[None], X], 0)                       # (n, m, C)
    n = k + 1
    L = P.argmax(2)
    A = (L[:, None, :] == L[None, :, :]).mean(2)              # hard agreement
    D = 0.5 * np.abs(P[:, None] - P[None]).sum(3).mean(2)     # mean TV distance
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
    # row gating against a (self-padded for small k) coordinate-wise median
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    r = np.abs(X - med[None]).sum(2)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = w[:, None] * rw                                       # (k, m)
    # self-distrust on rows where own state is an outlier
    if k >= 3:
        ws = np.clip((3.0 * sc - r0) / (2.0 * sc), 0.35, 1.0)
        # only distrust self where enough trusted mass exists
        ws = np.where(W.sum(0) >= 1.5, ws, 1.0)
    else:
        ws = np.ones_like(r0)
    out = (ws[:, None] * x + (W[..., None] * X).sum(0)) / (ws + W.sum(0))[..., None]
    return san(out)
```

### Candidate C4_2_TukeyIRLSReputation
New estimator: a row-wise Tukey-biweight M-estimator of location computed by iteratively re-weighted least squares starting from the coordinate-wise median, with an L1 MAD-type scale per row. Redescending weights give exactly zero influence to gross outliers (gaussian/signflip/labelflip rows) while honest rows get near-equal weights, approaching the oracle mean; node-level reputation with min-memory multiplies the weights to resist adaptive attackers, and the self row keeps a floor weight as an anchor.
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
    P = np.concatenate([x[None], X], 0)                       # (n, m, C)
    if k <= 3:
        Pm = np.concatenate([x[None], P], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    r = np.abs(P - med[None]).sum(2)                          # (n, m)
    sc = np.median(r, 0) + 0.05                               # (m,)
    # node reputation: fraction of rows inside 2*scale, plus argmax agreement
    inside = (r[1:] <= 2.0 * sc[None]).mean(1)
    lab = med.argmax(1)
    ag = (X.argmax(2) == lab[None]).mean(1)
    ag0 = float((x.argmax(1) == lab).mean())
    agr = np.clip(ag / (max(ag0, float(ag.max())) + 1e-12), 0.0, 1.0)
    ev = np.minimum(inside, agr)
    rep = state.get('rep')
    if rep is None or rep.shape != ev.shape:
        rep = ev.copy()
    else:
        rep = np.minimum(ev, 0.75 * rep + 0.25 * ev)
    state['rep'] = rep
    nw = np.concatenate([[1.0], np.clip((rep - 0.45) / 0.35, 0.0, 1.0)])   # (n,)
    mu = med
    c = 3.5
    for _ in range(4):
        rr = np.abs(P - mu[None]).sum(2)                      # (n, m)
        s = np.median(rr, 0) + 0.05
        u = rr / (c * s[None])
        bw = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        Wt = nw[:, None] * bw
        Wt[0] = np.maximum(Wt[0], 0.5)                        # self anchor
        mu = (Wt[..., None] * P).sum(0) / Wt.sum(0)[..., None]
    return san(mu)
```

### Candidate C4_3_PersonalizedPageRankTrust
New idea: trust propagates through the neighbourhood similarity graph. A personalized PageRank seeded at the node itself runs over the sharpened soft-similarity matrix of {self} ∪ neighbours, so honest neighbours that agree with each other reinforce one another (which helps noisy-label nodes whose own view is poor), while a colluding Byzantine clique that is dissimilar to the seed gets little mass. The PPR trust is combined (min) with direct self-similarity and a min-memory, then used with soft row gating.
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
    S = np.clip(0.5 * A + 0.5 * (1.0 - D), 0.0, 1.0)
    K = S ** 6
    np.fill_diagonal(K, 0.0)
    colsum = K.sum(0, keepdims=True)
    M = K / np.maximum(colsum, 1e-12)                         # column-stochastic
    e0 = np.zeros(n)
    e0[0] = 1.0
    v = e0.copy()
    alpha = 0.7
    for _ in range(30):
        v = (1.0 - alpha) * e0 + alpha * (M @ v)
    vn = v[1:]
    ppr = vn / (vn.max() + 1e-12)
    direct = S[0, 1:] / (max(float(S[0, 1:].max()), float(np.median(S[1:, 1:] if k > 1 else S[0, 1:]))) + 1e-12)
    rel = np.minimum(np.sqrt(ppr), np.clip(direct, 0.0, 1.0))
    rel = 0.5 * rel + 0.5 * np.clip(direct, 0.0, 1.0) * (ppr > 0.15)
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.5) / 0.3, 0.0, 1.0)
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
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
```

### Candidate C4_4_BetaReputationHybrid
Treats each neighbour's per-iteration behaviour as Bernoulli evidence (min of row-acceptance rate and relative argmax agreement with the robust median) and maintains a Beta posterior with asymmetric penalty, so trust is earned slowly and lost quickly; an instantaneous evidence gate is multiplied in to catch attackers who switch behaviour late, and a temporal drift test flags neighbours whose messages jump far more than everyone else's. Combined with soft row gating and the self-distrust idea for noisy-label nodes.
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
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = np.concatenate([x[None], X], 0)
    med = np.median(Pm, 0)
    r = np.abs(X - med[None]).sum(2)                          # (k, m)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    acc = (r <= 2.0 * sc[None]).mean(1)
    lab = med.argmax(1)
    ag = (X.argmax(2) == lab[None]).mean(1)
    ag0 = float((x.argmax(1) == lab).mean())
    agr = np.clip(ag / (max(ag0, float(ag.max())) + 1e-12), 0.0, 1.0)
    ev = np.minimum(acc, agr)
    # temporal drift evidence
    prevX = state.get('prevX')
    prevx = state.get('prevx')
    if prevX is not None and prevX.shape == X.shape and prevx is not None:
        dr = np.abs(X - prevX).sum(2).mean(1)
        own = float(np.abs(x - prevx).sum(1).mean())
        base = float(np.median(np.concatenate([[own], dr]))) + 0.05
        dflag = np.clip((dr - 2.5 * base) / (2.5 * base), 0.0, 1.0)
        ev = ev * (1.0 - dflag)
    state['prevX'] = X.copy()
    state['prevx'] = x.copy()
    a = state.get('a')
    b = state.get('b')
    if a is None or a.shape != ev.shape:
        a = np.full(k, 2.0)
        b = np.full(k, 0.5)
    a = a + ev
    b = b + 2.0 * (1.0 - ev)
    state['a'] = a
    state['b'] = b
    mpost = a / (a + b)
    w = np.clip((mpost - 0.5) / 0.3, 0.0, 1.0) * np.clip((ev - 0.35) / 0.3, 0.0, 1.0)
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = w[:, None] * rw
    if k >= 3:
        ws = np.clip((3.0 * sc - r0) / (2.0 * sc), 0.4, 1.0)
        ws = np.where(W.sum(0) >= 1.5, ws, 1.0)
    else:
        ws = np.ones_like(r0)
    out = (ws[:, None] * x + (W[..., None] * X).sum(0)) / (ws + W.sum(0))[..., None]
    return san(out)
```
