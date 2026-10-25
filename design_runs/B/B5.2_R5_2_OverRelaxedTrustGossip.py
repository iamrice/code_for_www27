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
