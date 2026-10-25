def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = np.abs(X_nb - c).max(-1)
    med = np.median(d, 0) + 1e-12
    w = (d <= 2.12 * med).astype(float)
    w = 0.67 * state.get('w', w) + (1 - 0.67) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.19 * x_self + 0.81 * agg
    out = x_self + 0.92 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
