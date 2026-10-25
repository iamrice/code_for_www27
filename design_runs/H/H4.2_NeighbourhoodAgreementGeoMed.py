def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    n = k + 1
    if n == 1:
        return x_self
    G = P @ P.T
    sq = np.diag(G)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))
    off = D[~np.eye(n, dtype=bool)]
    h = np.median(off) + 1e-12
    ht = h * (1.0 + math.exp(-2.0 * t / T))
    K = np.exp(-(D / ht) ** 2)
    agree = (K.sum(1) - 1.0) / max(n - 1, 1)
    w = agree ** 2
    w[0] = 1.0
    z = (w[:, None] * P).sum(0) / w.sum()
    for _ in range(20):
        d = np.linalg.norm(P - z, axis=1)
        a = w / np.maximum(d, 1e-8)
        z_new = (a[:, None] * P).sum(0) / a.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
