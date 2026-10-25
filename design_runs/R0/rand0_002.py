def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.sqrt(((X_nb - c) ** 2).reshape(k, -1).sum(1))
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 2.04 * med / np.maximum(d, 1e-12))
    w = 0.81 * state.get('w', w) + (1 - 0.81) * w
    state['w'] = w
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.69 * x_self + 0.31000000000000005 * agg
    out = x_self + 0.57 * (out - x_self)
    return out
