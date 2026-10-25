def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    A = np.sqrt(np.concatenate([xs[None], Y], 0))       # (n, m, C)
    n = A.shape[0]
    med = np.median(A, 0)
    ref = 0.5 * med + 0.5 * A[0]
    d = np.linalg.norm(A - ref[None], axis=2)           # (n, m)
    keep = max(1, (n + 1) // 2)
    idx = np.argsort(d, axis=0, kind='stable')[:keep]   # (keep, m)
    sel = np.take_along_axis(A, idx[:, :, None], axis=0)
    z = sel.mean(0)
    out = z ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
