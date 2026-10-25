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
    c = xs
    if k >= 3:
        Z = np.concatenate([xs[None], Y], 0)
        med = np.median(Z, 0)
        ds = float(np.linalg.norm(xs - med))
        dn = np.linalg.norm((Y - med[None]).reshape(k, -1), axis=1)
        ref = float(np.median(dn)) + 1e-12
        alpha = float(np.clip(0.5 * (ds / ref - 1.0), 0.0, 0.5))
        prev = state.get('alpha')
        if prev is not None:
            alpha = 0.5 * prev + 0.5 * alpha
        state['alpha'] = alpha
        c = xs + alpha * (med - xs)
    D = Y - c[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = scale.copy()
    else:
        r = 0.6 * r + 0.4 * scale
    state['rep'] = r
    w = r * scale
    out = c + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
