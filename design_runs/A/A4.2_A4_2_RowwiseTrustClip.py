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
    rn = np.linalg.norm(D, axis=2)                      # (k, m)
    tau_row = np.quantile(rn, 0.25, axis=0) + 1e-12     # (m,)
    tau_jr = tau_row[None, :] * (1.0 + 2.0 * r[:, None] ** 2)
    sc = np.minimum(1.0, tau_jr / np.maximum(rn, 1e-12))
    gsc = np.minimum(1.0, tau * (1.0 + 2.0 * r ** 2) / np.maximum(nrm, 1e-12))
    w = (r * np.sqrt(gsc))[:, None] * sc                # (k, m)
    out = xs + (w[:, :, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
