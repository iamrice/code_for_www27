def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = np.median(A, 0)
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 1.79 * med / np.maximum(d, 1e-12))
    w = 0.83 * state.get('w', w) + (1 - 0.83) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.33 * x_self + 0.6699999999999999 * agg
    out = x_self + 0.98 * (out - x_self)
    return out
