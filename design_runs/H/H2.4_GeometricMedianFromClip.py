def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    xs = x_self.reshape(-1)
    D = X_nb.reshape(k, -1) - xs
    nrm = np.linalg.norm(D, axis=1)
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    Q = np.concatenate([xs[None], xs + scale[:, None] * D])
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    z = Q.mean(0)
    for _ in range(20):
        d = np.linalg.norm(Q - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * Q).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)
