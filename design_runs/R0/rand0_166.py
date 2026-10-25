def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    c = A.mean(0)
    d = np.sqrt(((X_nb - c) ** 2).reshape(k, -1).sum(1))
    med = np.median(d, 0) + 1e-12
    r = np.argsort(np.argsort(d, 0), 0); w = (r < max(1, int(np.ceil(0.53 * k)))).astype(float)
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.55 * x_self + 0.44999999999999996 * agg
    out = x_self + 0.99 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
