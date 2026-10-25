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
    D = X - x[None]                                  # (k, m, C)
    r = np.linalg.norm(D, axis=2)                    # (k, m) per-row distances
    tau = np.quantile(r, 0.4, axis=0) + 1e-8         # (m,) per-row radius
    scale = np.minimum(1.0, tau[None] / np.maximum(r, 1e-12))
    out = x + (scale[..., None] * D).sum(0) / (k + 1)
    return san(out)
