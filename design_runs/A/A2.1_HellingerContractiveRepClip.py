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
    S = np.sqrt(xs)
    R = np.sqrt(Y)
    D = R - S[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    q = 0.25 if k >= 4 else 0.5
    tau = float(np.quantile(nrm, q)) + 1e-12
    prev_tau = state.get('tau')
    if prev_tau is not None:
        tau = min(tau, 1.1 * prev_tau)
    state['tau'] = tau
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = scale.copy()
    else:
        # fast to lose trust, slow to regain it
        r = np.where(scale < r, 0.4 * r + 0.6 * scale, 0.7 * r + 0.3 * scale)
    state['rep'] = r
    w = r * scale
    out = S + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
