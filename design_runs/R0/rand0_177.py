def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    S = np.sort(A, 0); b = int(np.floor(0.21 * len(A))); c = S[b:len(A) - b].mean(0) if len(A) - 2 * b >= 1 else np.median(A, 0)
    d = np.abs(X_nb - c).reshape(k, -1).sum(1)
    med = np.median(d, 0) + 1e-12
    w = np.exp(-d / (0.8 * med))
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.27 * x_self + 0.73 * agg
    out = x_self + 0.72 * (out - x_self)
    return out
