### Candidate R4_1_SemanticAlignmentRowClip
Refines the best rule (SemanticAlignment): after t = 0 the alignment reference leans on the robust centre rather than on the node's own (possibly noisy-label) state, a node that disagrees with its trusted neighbourhood defers more (w_self down to 0.35), and a new per-sample (row-level) soft clipping stage down-weights a trusted neighbour only on the samples where it deviates strongly from the provisional consensus, which limits the damage of partially-malicious or time-varying neighbours.
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
    if tsum <= 1e-6:
        return xs
    nb = (trust[:, None, None] * P).sum(0) / tsum
    s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
    w_self = 0.35 + 0.65 * s_agree
    cons = (w_self * xs + tsum * nb) / (w_self + tsum)
    rd = np.abs(P - cons[None]).sum(-1)                     # (k, m)
    good = trust > 0.5
    base = rd[good] if good.any() else rd
    tau = 2.0 * float(np.median(base)) + 0.05
    rw = np.minimum(1.0, tau / np.maximum(rd, 1e-12))
    W = trust[:, None] * rw                                 # (k, m)
    num = w_self * xs + (W[:, :, None] * P).sum(0)
    den = w_self + W.sum(0)
    return num / den[:, None]
```

### Candidate R4_2_CoreAlignmentFusion
Combines the two strongest complementary signals: a Hellinger-medoid core (TrustCore) gives a geometric, collusion-resistant centre and radius-based weights, while the soft-confusion alignment score (SemanticAlignment) computed against that core detects semantically permuted or uninformative senders that may be geometrically close; the asymmetric (fast-down, slow-up) trust memory is kept.
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
    rmax = Mx.max(2)
    ratio = diag / np.maximum(rmax, 1e-12)
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
    return (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)
```

### Candidate R4_3_PersonalizedPageRankTrust
New idea: build an agreement graph over {self, neighbours} (chance-corrected argmax agreement times Hellinger affinity, with simplex-violation penalty) and score each neighbour by personalized PageRank restarted at the node itself. Trust then flows from the node through mutually consistent senders, so a colluding Byzantine clique that agrees internally but is weakly linked to the node's own view gets little mass, even when it is a local majority; the result is combined with direct affinity and the asymmetric trust memory.
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
    lab = Z.argmax(-1)
    AG = (lab[:, None, :] == lab[None, :, :]).mean(-1)
    kappa = np.clip((AG - 1.0 / C) / (1.0 - 1.0 / C), 0.0, 1.0)
    vv = np.concatenate([[0.0], viol])
    A = (kappa ** 2) * np.clip(1.0 - H, 0.0, 1.0) * np.exp(-(vv[:, None] + vv[None, :]) / 0.05)
    np.fill_diagonal(A, 0.0)
    rs = A.sum(1, keepdims=True)
    Pm = np.where(rs > 1e-12, A / np.maximum(rs, 1e-12), 0.0)
    alpha = 0.25
    e0 = np.zeros(n)
    e0[0] = 1.0
    pi = e0.copy()
    for _ in range(40):
        nxt = alpha * e0 + (1.0 - alpha) * (pi @ Pm)
        nxt[0] += max(0.0, 1.0 - float(nxt.sum()))
        pi = nxt
    r = pi[1:]
    rmx = float(r.max()) if k > 0 else 0.0
    if rmx <= 1e-12:
        w_ppr = np.zeros(k)
    else:
        w_ppr = np.clip((r / rmx - 0.2) / 0.5, 0.0, 1.0)
    a0 = A[0, 1:]
    amx = float(a0.max()) if k > 0 else 0.0
    if amx <= 1e-12:
        w_dir = np.zeros(k)
    else:
        w_dir = np.clip(a0 / (0.6 * amx), 0.0, 1.0)
    w_new = w_ppr * w_dir * np.exp(-viol / 0.05)
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
    w_self = 0.4 + 0.6 * s_agree
    return (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)
```

### Candidate R4_4_OneCoinEMWeighting
New idea grounded in crowdsourcing theory: treat every sender (self included) as a one-coin Dawid-Skene annotator, estimate its accuracy against a leave-one-out consensus by a few EM-style reweighting passes started from a self-anchored median, and weight it by the Bayes-optimal log-odds log(p(C-1)/(1-p)) minus a margin. Honest but noisy-label nodes are smoothly down-weighted (not rejected), which helps accuracy, while chance-level or permuted senders get zero weight. A fast-down/slow-up memory protects against senders that change behaviour.
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
    m = Z.shape[1]
    gate = np.exp(-np.concatenate([[0.0], viol]) / 0.05)
    lab = Z.argmax(-1)
    y0 = proj(0.5 * xs + 0.5 * np.median(Z, 0))
    w = gate.copy()
    for it in range(4):
        if it == 0:
            ref_lab = np.broadcast_to(y0.argmax(-1), (n, m))
        else:
            Wz = w[:, None, None] * Z
            tot = Wz.sum(0)
            ws = float(w.sum())
            den = ws - w
            loo = np.where(den[:, None, None] > 1e-6,
                           (tot[None] - Wz) / np.maximum(den, 1e-6)[:, None, None],
                           y0[None])
            ref_lab = loo.argmax(-1)
        p = (lab == ref_lab).mean(1)
        p = (p * m + 1.0) / (m + 2.0)
        p = np.clip(p, 1e-3, 1.0 - 1e-3)
        lw = np.clip(np.log(p * (C - 1) / (1.0 - p)), 0.0, 8.0)
        w = np.clip(lw - 1.5, 0.0, None) * gate
        w[0] = max(w[0], 0.5 * float(np.median(w)), 1e-3)
    wmax = float(w.max())
    wn = w / max(wmax, 1e-12)
    w_new = wn[1:]
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum <= 1e-6:
        return xs
    w_self = max(0.4, float(wn[0]))
    return (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)
```
