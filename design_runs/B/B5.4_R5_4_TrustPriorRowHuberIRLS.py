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
