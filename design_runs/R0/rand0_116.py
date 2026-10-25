def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.sqrt(((X_nb - c) ** 2).sum(-1))
    med = np.median(d, 0) + 1e-12
    w = (d <= 1.26 * med).astype(float)
    w = 0.81 * state.get('w', w) + (1 - 0.81) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.49 * x_self + 0.51 * agg
    out = x_self + 0.53 * (out - x_self)
    return out
