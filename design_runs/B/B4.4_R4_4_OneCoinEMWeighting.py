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
