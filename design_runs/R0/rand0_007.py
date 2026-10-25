def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    S = np.sort(A, 0); b = int(np.floor(0.26 * len(A))); c = S[b:len(A) - b].mean(0) if len(A) - 2 * b >= 1 else np.median(A, 0)
    F = X_nb.reshape(k, -1); g = c.ravel(); d = 1 - F @ g / (np.linalg.norm(F, axis=1) * np.linalg.norm(g) + 1e-12)
    med = np.median(d, 0) + 1e-12
    w = (d <= 2.99 * med).astype(float)
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.53 * x_self + 0.47 * agg
    out = x_self + 0.85 * (out - x_self)
    return out
