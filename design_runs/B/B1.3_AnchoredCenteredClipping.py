def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    xs = proj(clean(x_self))
    Z = np.concatenate([xs[None], R], 0)
    n = Z.shape[0]
    v = np.median(np.concatenate([xs[None], proj(R)], 0), 0)
    prev_tau = state.get('tau')
    tau_used = None
    for _ in range(3):
        D = Z - v[None]
        nrm = np.linalg.norm(D.reshape(n, -1), axis=1)
        tau = float(np.median(nrm)) + 1e-12
        if prev_tau is not None:
            tau = min(tau, 1.5 * prev_tau)
        if tau_used is None:
            tau_used = tau
        sc = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
        v = v + (sc[:, None, None] * D).mean(0)
    state['tau'] = tau_used
    return proj(v)
