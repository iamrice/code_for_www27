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
    Z = np.sqrt(np.concatenate([xs[None], Y], 0))     # (n, m, C)
    n = k + 1
    F = Z.reshape(n, -1)
    r = state.get('rep')
    if r is None or len(r) != k:
        r = np.ones(k)
    prior = np.concatenate([[1.0], r])
    c = 0.5 * F[0] + 0.5 * np.median(F, 0)
    scale = np.ones(n)
    for _ in range(3):
        Dc = F - c[None]
        dn = np.linalg.norm(Dc, axis=1)
        tau = float(np.median(dn[1:])) + 1e-12
        scale = np.minimum(1.0, tau / np.maximum(dn, 1e-12))
        w = prior * scale
        c = c + (w[:, None] * Dc).sum(0) / prior.sum()
    r = 0.6 * r + 0.4 * scale[1:] if 'rep' in state else scale[1:].copy()
    state['rep'] = r
    # bounded step from self toward the robust centre
    step = c - F[0]
    sn = np.linalg.norm(step)
    dself = np.linalg.norm(F[1:] - F[0][None], axis=1)
    lim = float(np.quantile(dself, 0.5)) + 1e-12
    alpha = (0.5 + 0.5 * (t + 1) / T)
    out = F[0] + alpha * step * min(1.0, lim / max(sn, 1e-12))
    out = np.clip(out.reshape(xs.shape), 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
