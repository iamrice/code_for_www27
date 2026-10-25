def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = np.median(A, 0)
    d = np.sqrt(((X_nb - c) ** 2).sum(-1))
    med = np.median(d, 0) + 1e-12
    w = np.exp(-d / (2.19 * med))
    w = 0.8 * state.get('w', w) + (1 - 0.8) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.03 * x_self + 0.97 * agg
    out = x_self + 0.83 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
