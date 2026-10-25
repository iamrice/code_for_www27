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
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    Z = np.concatenate([xs[None], Y], 0)
    med = np.median(Z, 0)
    ref = (0.5 * xs + 0.5 * med).argmax(-1)
    agree = (Y.argmax(-1) == ref[None]).mean(1)
    amax = float(agree.max())
    lo = 0.3
    hi = max(lo + 0.05, 0.8 * amax)
    ag = np.clip((agree - lo) / (hi - lo), 0.0, 1.0)
    ev = np.sqrt(scale) * ag
    r = state.get('rep')
    if r is None or len(r) != k:
        r = ev.copy()
    else:
        r = np.where(ev < r, 0.4 * r + 0.6 * ev, 0.7 * r + 0.3 * ev)
    state['rep'] = r
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
