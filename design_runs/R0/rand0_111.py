def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.sqrt(((X_nb - c) ** 2).sum(-1))
    med = np.median(d, 0) + 1e-12
    r = np.argsort(np.argsort(d, 0), 0); w = (r < max(1, int(np.ceil(0.79 * k)))).astype(float)
    w = 0.91 * state.get('w', w) + (1 - 0.91) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.64 * x_self + 0.36 * agg
    out = x_self + 0.53 * (out - x_self)
    return out
