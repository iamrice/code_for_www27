def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    P = np.sqrt(np.concatenate([xs[None], Y], 0))
    n = P.shape[0]
    F = P.reshape(n, -1)
    z = np.median(F, 0)
    for _ in range(30):
        d = np.linalg.norm(F - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * F).sum(0) / w.sum()
        if np.linalg.norm(z_new - z) < 1e-7:
            z = z_new
            break
        z = z_new
    ps = F[0]
    D = z - ps
    nd = np.linalg.norm(D)
    dn = np.linalg.norm(F[1:] - ps, axis=1)
    tau = float(np.median(dn)) + 1e-12
    step = D * min(1.0, tau / max(nd, 1e-12))
    out = (ps + step).reshape(xs.shape) ** 2
    out = out + 1e-12
    return out / out.sum(-1, keepdims=True)
