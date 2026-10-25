### Candidate A6_1_TrustedRadiusHedgeClip
Refines the best hedge-trust clipping rule: the clipping radius is no longer the global 25% quantile of all distances (which stays very conservative in benign runs) but is estimated from the neighbours the hedge currently trusts, blended with the global quantile and capped by the global median. Self-weight is annealed so honest nodes mix more in late iterations once trust has separated.
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
    q25 = float(np.quantile(nrm, 0.25))
    trusted = r >= 0.5
    if trusted.sum() >= 1:
        tq = float(np.median(nrm[trusted]))
    else:
        tq = q25
    tau = min(0.5 * q25 + 0.5 * tq, med) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    out = xs + (w[:, None, None] * D).sum(0) / (lam + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A6_2_IteratedTrustCenteredClip
Combines hedge reputation with Karimireddy-style iterated centred clipping: the first fixed-point step reproduces the hedge-clip rule (centred at self), then two more steps re-centre the clipping ball at the current estimate, so the result moves toward the trust-weighted robust centre of the honest cluster instead of staying anchored to a possibly noisy self state. Self contributes unclipped with annealed weight, which bounds how far any step can be dragged.
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
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    v = xs.copy()
    for it in range(3):
        Dv = Y - v[None]
        nv = np.linalg.norm(Dv.reshape(k, -1), axis=1)
        s = np.minimum(1.0, tau_j / np.maximum(nv, 1e-12))
        step = lam * (xs - v) + ((r * s)[:, None, None] * Dv).sum(0)
        v = v + step / (lam + r.sum())
    out = np.clip(v, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A6_3_PeerConsensusHedgeClip
New idea: the hedge loss of a neighbour is computed not only from its distance to self (which penalises honest neighbours when self is a noisy-label node) but also from its median distance to all other received messages (a Krum-like peer-consensus score using the full pairwise Gram matrix). The two log-ratio losses are averaged, accumulated over time, and drive the same trust-scaled self-centred clipping.
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
    loss_self = np.clip(np.log(np.maximum(nrm, 1e-12) / med), 0.0, 3.0)
    Z = np.concatenate([xs[None], Y]).reshape(k + 1, -1)
    G = Z @ Z.T
    dg = np.diag(G)
    P = np.sqrt(np.maximum(dg[:, None] + dg[None, :] - 2.0 * G, 0.0))
    if k >= 2:
        sc = np.zeros(k)
        for j in range(k):
            row = np.delete(P[j + 1], j + 1)
            sc[j] = np.median(row)
        msc = float(np.median(sc)) + 1e-12
        loss_peer = np.clip(np.log(np.maximum(sc, 1e-12) / msc), 0.0, 3.0)
        loss = 0.5 * loss_self + 0.5 * loss_peer
    else:
        loss = loss_self
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
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    out = xs + (w[:, None, None] * D).sum(0) / (lam + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate A6_4_SampleLevelHedgeClip
New idea: node-level hedge trust is kept, but clipping is done per public sample (row) instead of on the whole matrix: each row deviation is clipped to a radius set by the median row deviation across neighbours for that sample, widened for trusted neighbours. This lets honest neighbours fully correct the samples where self is wrong while capping any neighbour's influence on each individual sample, which is the unit the final argmax accuracy is measured on.
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
    Rn = np.linalg.norm(D, axis=2)
    tau_row = np.median(Rn, axis=0) + 1e-6
    gmed = float(np.median(tau_row)) + 1e-6
    tau_row = np.minimum(tau_row, 3.0 * gmed)
    tau_jr = (0.5 + 1.0 * r ** 2)[:, None] * tau_row[None, :]
    srow = np.minimum(1.0, tau_jr / np.maximum(Rn, 1e-12))
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    sg = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    sg = np.sqrt(sg)
    W = (r * sg)[:, None] * srow
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    out = xs + (W[:, :, None] * D).sum(0) / (lam + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
```
