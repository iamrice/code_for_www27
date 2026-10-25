def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.abs(X_nb - c).reshape(k, -1).max(1)
    med = np.median(d, 0) + 1e-12
    w = np.exp(-d / (2.83 * med))
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.12 * x_self + 0.88 * agg
    out = x_self + 0.33 * (out - x_self)
    return out
