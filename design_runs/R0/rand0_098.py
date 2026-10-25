def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.sqrt(((X_nb - c) ** 2).sum(-1))
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 2.69 * med / np.maximum(d, 1e-12))
    w = 0.64 * state.get('w', w) + (1 - 0.64) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.18 * x_self + 0.8200000000000001 * agg
    out = x_self + 0.61 * (out - x_self)
    return out
