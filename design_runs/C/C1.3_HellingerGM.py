def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    S = np.sqrt(np.concatenate([x[None], X], 0))     # (n, m, C)
    wts = np.ones(k + 1)
    wts[0] = 1.0 + 0.25 * k
    y = np.median(S, 0)
    for _ in range(10):
        d = np.linalg.norm(S - y[None], axis=2)      # (n, m)
        inv = wts[:, None] / np.maximum(d, 1e-6)
        y = (inv[..., None] * S).sum(0) / inv.sum(0)[..., None]
    return san(y ** 2)
