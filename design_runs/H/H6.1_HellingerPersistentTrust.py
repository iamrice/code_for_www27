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
    S = np.sqrt(P)
    ref = np.median(S, 0)
    raw = np.sqrt(((S - ref) ** 2).sum(-1)).mean(-1) / math.sqrt(2.0)
    r = raw / (np.median(raw) + 1e-12)
    s_prev = state.get('s')
    if s_prev is None or len(s_prev) != k:
        s_prev = np.zeros(k)
    s = 0.6 * np.asarray(s_prev, dtype=float) + 0.4 * r[1:]
    state['s'] = s
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    trust = band * np.exp(-1.5 * np.maximum(s - 1.0, 0.0))
    z = (P[0] + np.tensordot(trust, P[1:], axes=1)) / (1.0 + trust.sum())
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
