def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = np.median(A, 0)
    d = np.abs(X_nb - c).sum(-1)
    med = np.median(d, 0) + 1e-12
    w = (d <= 0.93 * med).astype(float)
    w = 0.71 * state.get('w', w) + (1 - 0.71) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.58 * x_self + 0.42000000000000004 * agg
    out = x_self + 0.83 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
