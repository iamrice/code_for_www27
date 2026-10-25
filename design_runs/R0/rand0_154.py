def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = np.median(A, 0)
    d = np.abs(X_nb - c).reshape(k, -1).max(1)
    med = np.median(d, 0) + 1e-12
    r = np.argsort(np.argsort(d, 0), 0); w = (r < max(1, int(np.ceil(0.66 * k)))).astype(float)
    w = 0.79 * state.get('w', w) + (1 - 0.79) * w
    state['w'] = w
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.17 * x_self + 0.83 * agg
    out = x_self + 0.4 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
