def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = np.exp(-d / (2.52 * med))
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.16 * x_self + 0.84 * agg
    out = x_self + 0.49 * (out - x_self)
    return out
