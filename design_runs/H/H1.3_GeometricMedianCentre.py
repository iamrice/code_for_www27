def aggregate(x_self, X_nb, t, T, state):
    P = np.concatenate([x_self[None], X_nb]).reshape(len(X_nb) + 1, -1)
    z = np.median(P, 0)
    for _ in range(30):
        d = np.linalg.norm(P - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * P).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)
