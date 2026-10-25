def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = np.median(A, 0)
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = np.exp(-d / (2.68 * med))
    w = 0.7 * state.get('w', w) + (1 - 0.7) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.49 * x_self + 0.51 * agg
    out = x_self + 0.83 * (out - x_self)
    return out
