def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.abs(X_nb - c).reshape(k, -1).sum(1)
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 2.83 * med / np.maximum(d, 1e-12))
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.63 * x_self + 0.37 * agg
    out = x_self + 0.7 * (out - x_self)
    return out
