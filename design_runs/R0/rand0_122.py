def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = (d <= 1.32 * med).astype(float)
    w = 0.79 * state.get('w', w) + (1 - 0.79) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.7 * x_self + 0.30000000000000004 * agg
    out = x_self + 0.32 * (out - x_self)
    return out
