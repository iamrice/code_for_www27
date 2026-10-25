def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = x_self
    d = (X_nb.argmax(-1) != c.argmax(-1)[None]).mean(1) + 1e-3 * np.abs(X_nb - c).reshape(k, -1).mean(1)
    med = np.median(d, 0) + 1e-12
    r = np.argsort(np.argsort(d, 0), 0); w = (r < max(1, int(np.ceil(0.49 * k)))).astype(float)
    w = 0.84 * state.get('w', w) + (1 - 0.84) * w
    state['w'] = w
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.27 * x_self + 0.73 * agg
    out = x_self + 0.71 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
