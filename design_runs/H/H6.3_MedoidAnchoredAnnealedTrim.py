def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, m, C).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / C)
    F = np.sqrt(P).reshape(k + 1, -1)
    G = F @ F.T
    Dm = np.sqrt(np.maximum(2.0 * m - 2.0 * G, 0.0))
    med = int(np.argmin(Dm.sum(1)))
    dm = Dm[med, 1:]
    frac = 0.5 + 0.4 * t / max(T, 1)
    n_keep = max(1, int(round(frac * k)))
    keep = np.argsort(dm, kind='stable')[:n_keep]
    z = (P[0] + P[1:][keep].sum(0)) / (1.0 + n_keep)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
