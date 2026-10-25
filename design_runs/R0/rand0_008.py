def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.abs(X_nb - c).reshape(k, -1).sum(1)
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 2.83 * med / np.maximum(d, 1e-12))
    w = 0.94 * state.get('w', w) + (1 - 0.94) * w
    state['w'] = w
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.65 * x_self + 0.35 * agg
    out = x_self + 0.99 * (out - x_self)
    return out
