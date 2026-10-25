def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.median(P, 0)
    d0 = np.linalg.norm(P - z, axis=1)
    med0 = np.median(d0)
    mad = np.median(np.abs(d0 - med0))
    sig = (1.4826 * mad + 0.5 * med0 + 1e-12) * (1.0 + math.exp(-2.0 * t / T))
    for _ in range(10):
        d = np.linalg.norm(P - z, axis=1)
        u = np.minimum(d / sig, 1e6)
        w = np.exp(-0.5 * u * u)
        w[0] = 1.0
        z_new = (w[:, None] * P).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
