def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = np.exp(-d / (1.47 * med))
    w = 0.92 * state.get('w', w) + (1 - 0.92) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.19 * x_self + 0.81 * agg
    out = x_self + 0.31 * (out - x_self)
    return out
