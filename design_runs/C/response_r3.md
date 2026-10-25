### Candidate C3_1_RowMADSuspicionV2
This refines the strongest candidate so far (row-wise MAD gating plus node-level suspicion memory). When the degree is small (k <= 3), the reference median uses a doubled self-anchor so that one Byzantine neighbour cannot pull it. Rows near the threshold get soft weights that taper off instead of a hard cut, and suspicion is forgiven more slowly so that an attacker cannot regain trust within a few iterations.
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
    r = np.abs(X - med[None]).sum(2)                         # (k, m)
    r0 = np.abs(x - med).sum(1)                              # (m,)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    frac = 1.0 - (r <= 2.0 * sc[None]).mean(1)               # row rejection rate
    rw = np.clip((2.5 * sc[None] - r) / sc[None], 0.0, 1.0)  # 1 below 1.5sc, 0 above 2.5sc
    susp = state.get('susp')
    if susp is None or susp.shape != frac.shape:
        susp = frac.copy()
    else:
        susp = np.maximum(frac, 0.85 * susp + 0.15 * frac)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    W = trust[:, None] * rw
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
```

### Candidate C3_2_RobustPeerAgreement
New idea: each neighbour is scored by how many other peers vouch for it. We compute the pairwise argmax-agreement matrix over all messages. A neighbour's credibility is the average of (a) its agreement with self and (b) the median of its agreement with every other peer. Because of the median, a colluding minority cannot vouch for its own members. The scores are normalised against the most credible peer, kept in memory as the worst value seen so far (with slow forgiveness), and combined with row-wise MAD gating.
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
    P = np.concatenate([x[None], X], 0)                      # (n, m, C)
    n = k + 1
    L = P.argmax(2)                                          # (n, m)
    A = (L[:, None, :] == L[None, :, :]).mean(2)             # (n, n)
    Ao = A.astype(float).copy()
    np.fill_diagonal(Ao, np.nan)
    peer = np.nanmedian(Ao, axis=0)                          # (n,) median agreement with others
    score = 0.5 * A[0, 1:] + 0.5 * peer[1:]
    s0 = peer[0]
    ref = max(float(s0), float(score.max())) + 1e-12
    rel = score / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)                # (k,)
    # row-wise gating against coordinate-wise median
    med = np.median(P, 0)
    r = np.abs(X - med[None]).sum(2)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = w[:, None] * rw
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
```

### Candidate C3_3_TemporalDriftSuspicion
This is RowMAD suspicion with a second, attack-agnostic signal: temporal consistency. In gossip, an honest neighbour's message changes smoothly from one iteration to the next, by about as much as our own state does. A neighbour whose message jumps much more than the median drift (fresh noise, or switching behaviour mid-run) is flagged. Its suspicion is the larger of the row-rejection rate and the drift flag.
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
    med = np.median(np.concatenate([x[None], X], 0), 0)
    r = np.abs(X - med[None]).sum(2)                         # (k, m)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rowacc = (r <= 2.0 * sc[None]).astype(float)
    frac = 1.0 - rowacc.mean(1)
    # temporal drift test
    prevX = state.get('prevX')
    prevx = state.get('prevx')
    if prevX is not None and prevX.shape == X.shape and prevx is not None:
        dr = np.abs(X - prevX).sum(2).mean(1)                # (k,)
        own = float(np.abs(x - prevx).sum(1).mean())
        base = float(np.median(np.concatenate([[own], dr]))) + 0.05
        dflag = np.clip((dr - 2.0 * base) / (2.0 * base), 0.0, 1.0)
    else:
        dflag = np.zeros(k)
    state['prevX'] = X.copy()
    state['prevx'] = x.copy()
    raw = np.maximum(frac, dflag)
    susp = state.get('susp')
    if susp is None or susp.shape != raw.shape:
        susp = raw.copy()
    else:
        susp = np.maximum(raw, 0.8 * susp + 0.2 * raw)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    W = trust[:, None] * rowacc
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
```

### Candidate C3_4_RowClippedTrustGossip
This combines self-centred clipping (ClippedGossip) with RowMAD suspicion, applied per row. Each neighbour's per-row update toward it, x_j - x_self, is clipped in L1 to the median per-row neighbour displacement, so any single message has bounded influence even before it is detected. Node-level suspicion memory then removes neighbours that are persistently inconsistent.
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
    med = np.median(np.concatenate([x[None], X], 0), 0)
    r = np.abs(X - med[None]).sum(2)                         # (k, m)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    frac = 1.0 - (r <= 2.0 * sc[None]).mean(1)
    susp = state.get('susp')
    if susp is None or susp.shape != frac.shape:
        susp = frac.copy()
    else:
        susp = np.maximum(frac, 0.8 * susp + 0.2 * frac)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)            # (k,)
    D = X - x[None]                                          # (k, m, C)
    dn = np.abs(D).sum(2)                                    # (k, m)
    tau = np.maximum(np.median(dn, 0), 2.0 * sc) + 0.02      # (m,)
    c = np.minimum(1.0, tau[None] / np.maximum(dn, 1e-12))   # (k, m)
    W = trust[:, None] * c
    upd = (W[..., None] * D).sum(0) / (1.0 + trust.sum())
    return san(x + upd)
```
