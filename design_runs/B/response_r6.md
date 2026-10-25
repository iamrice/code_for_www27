### Candidate TemporalStepTrustRowClip
Refines the best rule (B4.1) with a temporal check. Honest gossip is a contraction, so each honest neighbour's message changes smoothly and by about the same amount as everyone else's from one iteration to the next. A neighbour whose step size is far above the robust median step (for example one that switches behaviour, or sends fresh noise) loses trust. The distance gate also tightens gradually over time, so a neighbour that never converges toward the consensus loses weight in late iterations.
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
    cf = 2.0 - 0.6 * t / max(T - 1, 1)
    w_dist = 1.0 / (1.0 + (dist[1:] / (cf * d_ref)) ** 4)
    w_new = w_align * w_dist * np.exp(-viol / 0.05)
    prevP = state.get('prevP')
    prevx = state.get('prevx')
    if prevP is not None and prevP.shape == P.shape and prevx is not None:
        step = np.abs(P - prevP).sum(-1).mean(1)
        sstep = float(np.abs(xs - prevx).sum(-1).mean())
        s_ref = float(np.median(np.concatenate([[sstep], step]))) + 0.02
        w_temp = 1.0 / (1.0 + (step / (3.0 * s_ref)) ** 4)
        w_new = w_new * w_temp
    state['prevP'] = P.copy()
    state['prevx'] = xs.copy()
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
    return num / den[:, None]
```

### Candidate ClassConditionalReliability
New idea for non-IID data. Each node is reliable on some classes and not others. From the soft confusion of each neighbour against the robust consensus we estimate a per-class precision (smoothed toward a prior). Each row of a neighbour's message is then weighted by that neighbour's relative precision on the class it predicts for that row, so a neighbour counts where it is competent and counts less elsewhere. This works on top of B4.1's trust gating and row clipping, and the node applies the same rule to its own message.
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
    # class-conditional precision against consensus (self judged against neighbours only)
    pc, p0 = 5.0, 0.8
    Mc = np.einsum('kmc,md->kcd', P, cons)
    prec = (np.einsum('kcc->kc', Mc) + pc * p0) / (Mc.sum(2) + pc)          # (k, C)
    Ms = np.einsum('mc,md->cd', xs, nb)
    prec_s = (np.diag(Ms) + pc * p0) / (Ms.sum(1) + pc)                     # (C,)
    allp = np.concatenate([prec_s[None], prec], 0)
    allw = np.concatenate([[w_self], trust])
    pref = (allw[:, None] * allp).sum(0) / max(float(allw.sum()), 1e-12)
    rel = np.clip(allp / np.maximum(pref[None], 1e-3), 0.3, 1.7)            # (k+1, C)
    f = np.einsum('kmc,kc->km', Z, rel)                                     # (k+1, m)
    fs, fn = f[0], f[1:]
    rd = np.abs(P - cons[None]).sum(-1)
    good = trust > 0.5
    base = rd[good] if good.any() else rd
    tau = 2.0 * float(np.median(base)) + 0.05
    rw = np.minimum(1.0, tau / np.maximum(rd, 1e-12))
    W = trust[:, None] * rw * fn
    ws = w_self * fs
    num = ws[:, None] * xs + (W[:, :, None] * P).sum(0)
    den = ws + W.sum(0)
    return num / np.maximum(den, 1e-12)[:, None]
```

### Candidate RobustDawidSkeneFusion
New idea that treats the neighbourhood as a crowd of annotators. Starting from the B4.1 robust linear consensus, it runs two EM steps of a trust-weighted, tempered Dawid–Skene model. Each trusted source (self plus neighbours) gets a smoothed confusion matrix against the current latent labels, and a row's posterior combines the sources' averaged soft log-likelihoods with a class prior. A neighbour that is systematically confused is then partly corrected instead of only being down-weighted. The output is half posterior and half the robust linear pool, so the state stays calibrated and the gossip stays stable.
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
    y_lin = num / den[:, None]
    # trust-weighted tempered Dawid-Skene refinement
    act = trust > 0.05
    A = np.concatenate([xs[None], P[act]], 0)                       # (a, m, C)
    wa = np.concatenate([[w_self], trust[act]])
    wsum = float(wa.sum())
    eye = np.eye(C)
    q = y_lin
    for _ in range(2):
        Cm = np.einsum('mc,amd->acd', q, A) + 2.0 * eye[None] + 0.2
        Cm = Cm / Cm.sum(2, keepdims=True)
        L = np.einsum('amd,acd->amc', A, np.log(Cm))                # (a, m, C)
        prior = q.mean(0) + 1e-3
        S = (wa[:, None, None] * L).sum(0) / wsum + 0.5 * np.log(prior / prior.sum())[None]
        S = S - S.max(1, keepdims=True)
        E = np.exp(S)
        q = E / E.sum(1, keepdims=True)
    return 0.5 * y_lin + 0.5 * q
```

### Candidate AnnealedMixingHuberIRLS
Combines B4.1's single-pass row clip with B5.4's iterative Huber refinement, but drops the reputation term that did not help. It adds annealed mixing: the self-weight shrinks as t grows, so a node that has already filtered its neighbours mixes faster with them in later iterations, moving it closer to the honest-average oracle on sparse graphs. Meanwhile the distance gate tightens over time, which bounds the extra exposure to attackers.
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
    prog = t / max(T - 1, 1)
    cf = 2.0 - 0.5 * prog
    w_dist = 1.0 / (1.0 + (dist[1:] / (cf * d_ref)) ** 4)
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
    w_self = (0.35 + 0.65 * s_agree) * (1.0 - 0.4 * prog)
    y = (w_self * xs + tsum * nb) / (w_self + tsum)
    good = trust > 0.5
    for _ in range(2):
        rd = np.abs(P - y[None]).sum(-1)
        base = rd[good] if good.any() else rd
        tau = 2.0 * float(np.median(base)) + 0.05
        rw = np.minimum(1.0, tau / np.maximum(rd, 1e-12))
        W = trust[:, None] * rw
        y = (w_self * xs + (W[:, :, None] * P).sum(0)) / (w_self + W.sum(0))[:, None]
    return y
```
