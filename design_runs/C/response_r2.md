### Candidate C2.1_DualSuspicion
Refines the best rule (C1.4 row-MAD suspicion) by fusing it with the complementary argmax-agreement signal of C1.2: a neighbour's raw suspicion is the max of its row-rejection rate and its argmax disagreement with the robust median (this catches confident-but-wrong attacks like labelflip that the L1 test only partially sees). Row acceptance is made soft (linear ramp between 2 and 3 robust scales) and suspicion has sticky memory.
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
    r0 = np.abs(x - med).sum(1)                              # (m,)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)  # soft row weights
    frac = 1.0 - (r <= 2.0 * sc[None]).mean(1)               # row rejection rate
    lab = med.argmax(1)
    agree = (X.argmax(2) == lab[None]).mean(1)
    ref = max(float((x.argmax(1) == lab).mean()), float(agree.max())) + 1e-12
    dis = np.clip(1.0 - agree / ref, 0.0, 1.0)               # relative argmax disagreement
    raw = np.maximum(frac, dis)
    susp = state.get('susp')
    if susp is None or susp.shape != raw.shape:
        susp = raw.copy()
    else:
        susp = np.maximum(raw, 0.8 * susp + 0.2 * raw)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    W = trust[:, None] * rw                                  # (k, m)
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
```

### Candidate C2.2_HellingerMeanShift
Genuinely new: per-row mode seeking. In Hellinger (sqrt-probability) space we run a few Gaussian-kernel mean-shift steps from the coordinate-wise median with a data-driven bandwidth (median squared distance), so the estimate locks onto the densest cluster of messages rather than the centre of all of them; per-node average kernel mass then gives a reputation (with memory) that multiplies the kernel weights in the final probability-space average.
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
    S = np.sqrt(P)
    y = np.median(S, 0)
    w = None
    for _ in range(4):
        d2 = ((S - y[None]) ** 2).sum(2)                     # (n, m)
        h2 = np.median(d2, 0) + 1e-3
        w = np.exp(-0.5 * d2 / h2[None])
        y = (w[..., None] * S).sum(0) / (w.sum(0)[..., None] + 1e-12)
    d2 = ((S - y[None]) ** 2).sum(2)
    h2 = np.median(d2, 0) + 1e-3
    w = np.exp(-0.5 * d2 / h2[None])                         # final kernel weights
    g = w.mean(1)                                            # node-level kernel mass
    ratio = g[1:] / (np.median(g) + 1e-12)
    rep = state.get('rep')
    if rep is None or rep.shape != ratio.shape:
        rep = ratio.copy()
    else:
        rep = np.minimum(ratio, 0.7 * rep + 0.3 * ratio)
    state['rep'] = rep
    trust = np.clip((rep - 0.2) / 0.4, 0.0, 1.0)
    Wn = w * np.concatenate([[1.0], trust])[:, None]
    Wn[0] += 0.05                                            # small self anchor
    out = (Wn[..., None] * P).sum(0) / (Wn.sum(0)[..., None] + 1e-12)
    return san(out)
```

### Candidate C2.3_CohesionTrust
Node-level outlier detection by cohesion: compute the full pairwise mean-row Hellinger distance matrix among self and neighbours, score each node by its *median* distance to the others (robust to colluding identical attackers, unlike Krum-style nearest-neighbour sums), and down-weight nodes whose score exceeds 1.2x the median score, with sticky suspicion memory; the final step is a trust-weighted mean with self.
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
    S = np.sqrt(P)
    n = k + 1
    Dm = np.zeros((n, n))
    for a in range(n):
        Dm[a] = np.sqrt(((S - S[a][None]) ** 2).sum(2)).mean(1)
    score = np.array([np.median(np.delete(Dm[a], a)) for a in range(n)])
    ms = float(np.median(score)) + 1e-3
    raw = np.clip((score[1:] - 1.2 * ms) / (0.8 * ms), 0.0, 1.0)
    # also penalise neighbours far from self relative to typical cohesion
    raw = np.maximum(raw, np.clip((Dm[0, 1:] - 3.0 * ms) / (2.0 * ms), 0.0, 1.0))
    susp = state.get('susp')
    if susp is None or susp.shape != raw.shape:
        susp = raw.copy()
    else:
        susp = np.maximum(raw, 0.8 * susp + 0.2 * raw)
    state['susp'] = susp
    trust = 1.0 - susp
    out = (x + (trust[:, None, None] * X).sum(0)) / (1.0 + trust.sum())
    return san(out)
```

### Candidate C2.4_TrustedReferenceRowMAD
C1.4 with a self-purifying reference: the per-row median and robust scale are computed only from self plus neighbours trusted in the previous iteration, so persistent attackers stop biasing the reference that judges everyone; the acceptance multiplier anneals from 2.5 to 1.5 scales as honest messages contract during gossip.
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
    prev = state.get('trust')
    if prev is None or prev.shape != (k,):
        prev = np.ones(k)
    sel = prev >= 0.5
    if sel.sum() == 0:
        sel = np.ones(k, dtype=bool)
    ref_set = np.concatenate([x[None], X[sel]], 0)
    med = np.median(ref_set, 0)
    r = np.abs(X - med[None]).sum(2)                         # (k, m)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r[sel]], 0), 0) + 0.04
    mult = 2.5 - 1.0 * t / max(T - 1, 1)
    rowacc = np.clip((mult + 0.5) - r / sc[None], 0.0, 1.0)  # soft ramp around mult
    frac = 1.0 - (r <= mult * sc[None]).mean(1)
    susp = state.get('susp')
    if susp is None or susp.shape != frac.shape:
        susp = frac.copy()
    else:
        susp = np.maximum(frac, 0.8 * susp + 0.2 * frac)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    state['trust'] = trust
    W = trust[:, None] * rowacc
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
```
