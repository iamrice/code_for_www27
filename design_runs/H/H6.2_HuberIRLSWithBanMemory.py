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
    P0, Nb = P[0], P[1:]
    b_prev = state.get('b')
    if b_prev is None or len(b_prev) != k:
        b_prev = np.zeros(k)
    dself = np.sqrt(((Nb - P0) ** 2).sum(axis=(-1, -2)) / m)
    c = (np.median(dself) + 1e-12) * (1.0 + math.exp(-2.0 * t / T))
    z = P0.copy()
    w = np.ones(k)
    for _ in range(4):
        dz = np.sqrt(((Nb - z) ** 2).sum(axis=(-1, -2)) / m)
        w = np.minimum(1.0, c / np.maximum(dz, 1e-12))
        z = (P0 + np.tensordot(w, Nb, axes=1)) / (1.0 + w.sum())
    flag = (w < 0.5).astype(float)
    b = 0.7 * np.asarray(b_prev, dtype=float) + 0.3 * flag
    state['b'] = b
    w_eff = w * (b < 0.5)
    z = (P0 + np.tensordot(w_eff, Nb, axes=1)) / (1.0 + w_eff.sum())
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
