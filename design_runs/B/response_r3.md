### Candidate R3_1_ReliabilityCore
TrustCore's Hellinger medoid core with viol-gating, plus a new step: each node's reliability is estimated as its argmax agreement with a *leave-one-out* core centre on the confident half of the samples. That agreement is turned into the optimal weighted-voting weight log(p(C-1)/(1-p)), normalised by the core median. Honest nodes with noisy labels are then down-weighted (including self), so consensus can do better than the uniform honest average, and Byzantine nodes are still gated out.
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
    Wz = w_core[:, None, None] * Z
    tot = Wz.sum(0)
    wsum = float(w_core.sum())
    center = tot / wsum
    den = wsum - w_core
    loo = np.where(den[:, None, None] > 1e-6,
                   (tot[None] - Wz) / np.maximum(den, 1e-6)[:, None, None],
                   center[None])
    lab = Z.argmax(-1)
    lab_l = loo.argmax(-1)
    lab_c = center.argmax(-1)
    srt = np.sort(center, -1)
    marg = srt[:, -1] - srt[:, -2]
    conf = marg >= np.median(marg)
    nc = float(conf.sum())
    hits = ((lab == lab_l) & conf[None]).sum(1).astype(np.float64)
    p = (hits + 1.0) / (nc + 2.0)
    lo = np.maximum(np.log(p * (C - 1) / np.maximum(1.0 - p, 1e-6)), 0.0)
    members = w_core > 0.5
    ref = max(float(np.median(lo[members])), 1e-6)
    rel = np.clip(lo / ref, 0.0, 1.5)
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab[0][None]).mean(1))
    w_agree = np.clip((agree - 2.5 / C) / (2.5 / C), 0.0, 1.0)
    w_new = (w_core * w_agree * rel)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    w_self = max(0.5, float(w_core[0])) * float(np.clip(rel[0], 0.5, 1.5))
    return (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + trust.sum())
```

### Candidate R3_2_ClassConditionalReliability
New idea for non-IID data: a node is usually reliable on the classes it trained on and unreliable on the others. After the robust core gate, I estimate each node's per-class recall against its leave-one-out centre, shrunk toward its overall accuracy, and convert it to per-class log-odds weights. Each sample is then weighted per node by those weights, taken in expectation under the centre's class distribution, so each sample listens to the nodes that are competent on its class.
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
    Wz = w_core[:, None, None] * Z
    tot = Wz.sum(0)
    wsum = float(w_core.sum())
    center = tot / wsum
    den = wsum - w_core
    loo = np.where(den[:, None, None] > 1e-6,
                   (tot[None] - Wz) / np.maximum(den, 1e-6)[:, None, None],
                   center[None])
    lab = Z.argmax(-1)
    lab_l = loo.argmax(-1)
    lab_c = center.argmax(-1)
    hitm = (lab == lab_l)
    L = np.eye(C)[lab_l]                                   # (n, m, C)
    cnt = L.sum(1)                                         # (n, C)
    hit = (L * hitm[..., None]).sum(1)                     # (n, C)
    pa = hitm.mean(1)
    rc = (hit + 4.0 * pa[:, None]) / (cnt + 4.0)
    rc = np.clip(rc, 1e-3, 1.0 - 1e-3)
    lw = np.clip(np.log(rc * (C - 1) / (1.0 - rc)), 0.0, 6.0)   # (n, C)
    u = (center @ lw.T).T                                  # (n, m)
    members = w_core > 0.5
    ref = max(float(np.median(u.mean(1)[members])), 1e-6)
    u = np.clip(u / ref, 0.0, 2.0)
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab[0][None]).mean(1))
    w_agree = np.clip((agree - 2.5 / C) / (2.5 / C), 0.0, 1.0)
    w_new = (w_core * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    g = np.concatenate([[max(0.5, float(w_core[0]))], trust])
    W = g[:, None] * u
    W[0] = np.maximum(W[0], 0.25)
    return (W[:, :, None] * Z).sum(0) / W.sum(0)[:, None]
```

### Candidate R3_3_SemanticAlignmentMomentum
This keeps the best previous rule (soft-confusion semantic alignment, distance and validity gating, fast-down trust memory) and adds heavy-ball acceleration, x_{t+1} = A(x_t) + beta (x_t - x_{t-1}). With only T=10 gossip steps on a sparse graph, consensus is far from mixed, and momentum brings honest states closer to the network-wide honest average. Gating is applied before the momentum step, so Byzantine messages are not amplified.
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
    rmax = M.max(2)
    ratio = diag / np.maximum(rmax, 1e-12)
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
    if tsum > 1e-6:
        nb = (trust[:, None, None] * P).sum(0) / tsum
        s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
        w_self = 0.5 + 0.5 * s_agree
    else:
        w_self = 1.0
    y = (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)
    xp = state.get('x_prev')
    if xp is not None and xp.shape == xs.shape and tsum > 1e-6:
        beta = 0.3 if t < T - 1 else 0.15
        y = y + beta * (xs - xp)
    state['x_prev'] = xs.copy()
    return proj(y)
```

### Candidate R3_4_LogLinearPooling
The robust core gate and log-odds reliability weights are the same as in R3_1, but the fusion of the accepted messages changes: it is a 50/50 mix of linear pooling and reliability-weighted log-linear (geometric, product-of-experts) pooling with an epsilon floor. Geometric pooling counts a class as supported only when most trusted experts give it mass. This sharpens consensus on ambiguous samples and suppresses diffuse or stray mass from noisy-label nodes and any residual Byzantine leakage.
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
    Wz = w_core[:, None, None] * Z
    tot = Wz.sum(0)
    wsum = float(w_core.sum())
    center = tot / wsum
    den = wsum - w_core
    loo = np.where(den[:, None, None] > 1e-6,
                   (tot[None] - Wz) / np.maximum(den, 1e-6)[:, None, None],
                   center[None])
    lab = Z.argmax(-1)
    lab_l = loo.argmax(-1)
    lab_c = center.argmax(-1)
    srt = np.sort(center, -1)
    marg = srt[:, -1] - srt[:, -2]
    conf = marg >= np.median(marg)
    nc = float(conf.sum())
    hits = ((lab == lab_l) & conf[None]).sum(1).astype(np.float64)
    p = (hits + 1.0) / (nc + 2.0)
    lo = np.maximum(np.log(p * (C - 1) / np.maximum(1.0 - p, 1e-6)), 0.0)
    members = w_core > 0.5
    ref = max(float(np.median(lo[members])), 1e-6)
    rel = np.clip(lo / ref, 0.0, 1.5)
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab[0][None]).mean(1))
    w_agree = np.clip((agree - 2.5 / C) / (2.5 / C), 0.0, 1.0)
    w_new = (w_core * w_agree * rel)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    w_self = max(0.5, float(w_core[0])) * float(np.clip(rel[0], 0.5, 1.5))
    w = np.concatenate([[w_self], trust])
    ws = float(w.sum())
    lin = (w[:, None, None] * Z).sum(0) / ws
    eps = 0.01
    lg = (w[:, None, None] * np.log(Z + eps)).sum(0) / ws
    lg = lg - lg.max(-1, keepdims=True)
    geo = np.exp(lg)
    geo = geo / geo.sum(-1, keepdims=True)
    return proj(0.5 * lin + 0.5 * geo)
```
