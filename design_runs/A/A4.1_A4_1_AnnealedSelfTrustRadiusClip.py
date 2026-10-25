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
    base = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = base.copy()
    else:
        r = np.where(base < r, 0.4 * r + 0.6 * base, 0.7 * r + 0.3 * base)
    state['rep'] = r
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    out = xs + (w[:, None, None] * D).sum(0) / (lam + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
