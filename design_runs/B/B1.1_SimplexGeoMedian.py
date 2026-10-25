def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)              # (k,) distance from the simplex
    w0 = np.concatenate([[1.0], np.exp(-viol / 0.05)])  # protocol-violation down-weighting
    Z = np.concatenate([xs[None], P], 0)              # (n, m, C)
    y = np.median(Z, 0)
    for _ in range(10):
        d = np.sqrt(((Z - y[None]) ** 2).sum(-1)) + 1e-6   # (n, m)
        w = w0[:, None] / d
        y = (w[:, :, None] * Z).sum(0) / w.sum(0)[:, None]
    return y
