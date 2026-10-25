def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    z = P[0].copy()
    for _ in range(8):
        d = np.linalg.norm(P - z, axis=1)
        s = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9
        c = 1.345 * max(s, 1e-6 * (1.0 + np.median(d)))
        w = np.minimum(1.0, c / np.maximum(d, 1e-12))
        z_new = (w[:, None] * P).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-8 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)
