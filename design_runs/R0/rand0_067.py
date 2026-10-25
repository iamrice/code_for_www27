def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    F = X_nb.reshape(k, -1); g = c.ravel(); d = 1 - F @ g / (np.linalg.norm(F, axis=1) * np.linalg.norm(g) + 1e-12)
    med = np.median(d, 0) + 1e-12
    w = (d <= 1.0 * med).astype(float)
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.6 * x_self + 0.4 * agg
    out = x_self + 0.67 * (out - x_self)
    return out
