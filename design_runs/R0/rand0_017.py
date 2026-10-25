def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 0.85 * med / np.maximum(d, 1e-12))
    w = 0.68 * state.get('w', w) + (1 - 0.68) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.12 * x_self + 0.88 * agg
    out = x_self + 0.69 * (out - x_self)
    return out
