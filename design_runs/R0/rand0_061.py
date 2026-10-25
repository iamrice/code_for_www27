def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.sqrt(((X_nb - c) ** 2).sum(-1))
    med = np.median(d, 0) + 1e-12
    w = np.exp(-d / (1.92 * med))
    w = 0.64 * state.get('w', w) + (1 - 0.64) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.13 * x_self + 0.87 * agg
    out = x_self + 0.65 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
