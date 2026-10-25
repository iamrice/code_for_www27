### Candidate R5_1_QualityWeightedRowClip
This keeps the strongest rule so far (B4.1: semantic-alignment gate, distance gate, monotone trust and row-wise clipping) and adds a continuous *reliability* weight. The gates only decide whether a neighbour is admitted. The new weight also favours cleaner honest neighbours: it is an EMA of how often each neighbour's argmax agrees with the trusted consensus. Noisy-label honest nodes therefore get less weight, without needing a hard threshold.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    center = proj(np.median(Z, 0))
    a = 0.5 if t == 0 else 0.3
    ref = a * xs + (1.0 - a) * center
    M = np.einsum('kmc,md->kcd', P, ref)
    rowsum = M.sum(2)
    diag = np.einsum('kcc->kc', M)
    ratio = diag / np.maximum(M.max(2), 1e-12)
    mass = rowsum / np.maximum(rowsum.sum(1, keepdims=True), 1e-12)
    align = (mass * ratio).sum(1)
    w_align = np.clip((align - 0.4) / 0.4, 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_new = w_align * w_dist * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum <= 1e-6:
        return xs
    nb = (trust[:, None, None] * P).sum(0) / tsum
    lab_nb = nb.argmax(-1)
    s_agree = float((xs.argmax(-1) == lab_nb).mean())
    # reliability: agreement with trusted consensus, smoothed over time
    q_now = (P.argmax(-1) == lab_nb[None]).mean(1)
    good = trust > 0.5
    q_top = float(np.max(q_now[good])) if good.any() else float(np.max(q_now))
    q_rel = np.clip(q_now / max(q_top, 1e-3), 0.0, 1.0)
    qp = state.get('qual')
    if qp is None or len(qp) != k:
        qual = q_rel
    else:
        qual = 0.6 * qp + 0.4 * q_rel
    state['qual'] = qual
    rel = 0.4 + 0.6 * qual ** 2
    w_self = 0.35 + 0.65 * s_agree
    wt = trust * rel
    wsum = float(wt.sum())
    cons = (w_self * xs + (wt[:, None, None] * P).sum(0)) / (w_self + wsum)
    rd = np.abs(P - cons[None]).sum(-1)
    base = rd[good] if good.any() else rd
    tau = 2.0 * float(np.median(base)) + 0.05
    rw = np.minimum(1.0, tau / np.maximum(rd, 1e-12))
    W = wt[:, None] * rw
    num = w_self * xs + (W[:, :, None] * P).sum(0)
    den = w_self + W.sum(0)
    return num / den[:, None]
```

### Candidate R5_2_OverRelaxedTrustGossip
This is the new idea. Trust filtering already removes most Byzantine influence, so most of the remaining gap to the oracle comes from slow mixing on a sparse graph within 10 iterations. Each node applies successive over-relaxation, x <- x + omega*(agg - x) with omega = 1.5, to its own robust update. On a lazy gossip operator (eigenvalues in [0,1]) this stays stable and shrinks the slow modes faster. It is switched off at t = 0, when trust is still unreliable, and at the last step. The output is projected back onto the simplex.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    center = proj(np.median(Z, 0))
    a = 0.5 if t == 0 else 0.3
    ref = a * xs + (1.0 - a) * center
    M = np.einsum('kmc,md->kcd', P, ref)
    rowsum = M.sum(2)
    diag = np.einsum('kcc->kc', M)
    ratio = diag / np.maximum(M.max(2), 1e-12)
    mass = rowsum / np.maximum(rowsum.sum(1, keepdims=True), 1e-12)
    align = (mass * ratio).sum(1)
    w_align = np.clip((align - 0.4) / 0.4, 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_new = w_align * w_dist * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum <= 1e-6:
        return xs
    nb = (trust[:, None, None] * P).sum(0) / tsum
    s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
    w_self = 0.35 + 0.65 * s_agree
    cons = (w_self * xs + tsum * nb) / (w_self + tsum)
    rd = np.abs(P - cons[None]).sum(-1)
    good = trust > 0.5
    base = rd[good] if good.any() else rd
    tau = 2.0 * float(np.median(base)) + 0.05
    rw = np.minimum(1.0, tau / np.maximum(rd, 1e-12))
    W = trust[:, None] * rw
    num = w_self * xs + (W[:, :, None] * P).sum(0)
    den = w_self + W.sum(0)
    agg = num / den[:, None]
    if 1 <= t <= T - 2 and good.any():
        omega = 1.5
        agg = proj(xs + omega * (agg - xs))
    return agg
```

### Candidate R5_3_CoreHybridOpinionPool
This uses the trust mechanism from B4.2: a Hellinger medoid core, a semantic-alignment gate and monotone trust. The fusion step changes to an equal mix of linear and log-linear (geometric) opinion pooling. The geometric pool gives extra weight to classes that the trusted peers *jointly* support. This helps argmax accuracy when individual knowledge is weak, as in the early FL snapshot that sets the worst case, while the linear half keeps averaging efficient. Probabilities are floored at 1e-4 so that no single peer can veto a class.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    n = k + 1
    S = np.sqrt(Z)
    BC = np.einsum('imc,jmc->ijm', S, S)
    H = np.sqrt(np.clip(1.0 - BC, 0.0, 1.0)).mean(-1)
    vv = np.concatenate([[0.0], viol])
    Dm = H + 5.0 * (vv[:, None] + vv[None, :])
    np.fill_diagonal(Dm, 0.0)
    q = max(1, (n + 1) // 2 - 1)
    sc = np.sort(Dm, 1)[:, 1:q + 1].mean(1)
    sc[0] *= 0.9
    med = int(np.argmin(sc))
    d = Dm[med]
    r = max(float(np.sort(d)[min(q, n - 1)]), 0.02)
    w_core = 1.0 / (1.0 + (d / (2.0 * r)) ** 4)
    w_core[med] = 1.0
    center = (w_core[:, None, None] * Z).sum(0) / w_core.sum()
    ref = 0.5 * (xs + center)
    Mx = np.einsum('kmc,md->kcd', P, ref)
    rowsum = Mx.sum(2)
    diag = np.einsum('kcc->kc', Mx)
    ratio = diag / np.maximum(Mx.max(2), 1e-12)
    mass = rowsum / np.maximum(rowsum.sum(1, keepdims=True), 1e-12)
    align = (mass * ratio).sum(1)
    w_align = np.clip((align - 0.4) / 0.4, 0.0, 1.0)
    w_new = w_core[1:] * w_align * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum <= 1e-6:
        return xs
    nb = (trust[:, None, None] * P).sum(0) / tsum
    s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
    w_self = max(0.5, 0.5 * float(w_core[0]) + 0.5 * s_agree)
    wall = np.concatenate([[w_self], trust])
    wsum = float(wall.sum())
    lin = (wall[:, None, None] * Z).sum(0) / wsum
    L = np.log(np.maximum(Z, 1e-4))
    g = (wall[:, None, None] * L).sum(0) / wsum
    g = g - g.max(-1, keepdims=True)
    geo = np.exp(g)
    geo = geo / np.maximum(geo.sum(-1, keepdims=True), 1e-12)
    return 0.5 * lin + 0.5 * geo
```

### Candidate R5_4_TrustPriorRowHuberIRLS
This keeps B2.4's semantic-alignment trust, used as a prior weight per neighbour. Instead of B4.1's single clipping step, it computes a per-sample (row-wise) Huber M-estimate in total-variation distance by iteratively reweighted least squares (IRLS), 3 iterations, with the node's own row as an anchor. A colluding attacker that survives the gate can then shift each sample's estimate by only a bounded amount, set by tau, while consistent honest rows keep their full weight. A slow-decaying EMA of each neighbour's agreement with the robust output also adjusts its trust over time.
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    center = np.median(Z, 0)
    ref = 0.5 * (xs + center)
    M = np.einsum('kmc,md->kcd', P, ref)
    rowsum = M.sum(2)
    diag = np.einsum('kcc->kc', M)
    ratio = diag / np.maximum(M.max(2), 1e-12)
    mass = rowsum / np.maximum(rowsum.sum(1, keepdims=True), 1e-12)
    align = (mass * ratio).sum(1)
    w_align = np.clip((align - 0.4) / 0.4, 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_new = w_align * w_dist * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    rep = state.get('rep')
    if rep is None or len(rep) != k:
        rep = np.ones(k)
    prior = trust * (0.5 + 0.5 * rep)
    psum = float(prior.sum())
    if psum <= 1e-6:
        return xs
    nb = (prior[:, None, None] * P).sum(0) / psum
    s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
    w_self = 0.4 + 0.6 * s_agree
    y = (w_self * xs + psum * nb) / (w_self + psum)
    good = trust > 0.5
    for _ in range(3):
        rd = np.abs(P - y[None]).sum(-1)                  # (k, m)
        base = rd[good] if good.any() else rd
        tau = 2.0 * float(np.median(base)) + 0.05
        hw = np.minimum(1.0, tau / np.maximum(rd, 1e-12))
        W = prior[:, None] * hw
        y = (w_self * xs + (W[:, :, None] * P).sum(0)) / (w_self + W.sum(0))[:, None]
    agr = (P.argmax(-1) == y.argmax(-1)[None]).mean(1)
    top = float(np.max(agr[good])) if good.any() else float(np.max(agr))
    rel = np.clip(agr / max(top, 1e-3), 0.0, 1.0)
    state['rep'] = 0.7 * rep + 0.3 * rel
    return y
```
