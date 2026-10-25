def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = (d <= 0.93 * med).astype(float)
    w = 0.84 * state.get('w', w) + (1 - 0.84) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.47 * x_self + 0.53 * agg
    out = x_self + 0.59 * (out - x_self)
    return out
