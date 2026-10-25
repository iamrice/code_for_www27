def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.sqrt(((X_nb - c) ** 2).sum(-1))
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 2.99 * med / np.maximum(d, 1e-12))
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.62 * x_self + 0.38 * agg
    out = x_self + 0.95 * (out - x_self)
    return out
