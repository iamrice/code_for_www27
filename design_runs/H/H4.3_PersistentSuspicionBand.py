def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    r = d / (np.median(d) + 1e-12)
    s_prev = state.get('s')
    if s_prev is None or len(s_prev) != k:
        s_prev = np.zeros(k)
    s = 0.6 * np.asarray(s_prev, dtype=float) + 0.4 * r[1:]
    state['s'] = s
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    trust = band * np.exp(-np.maximum(s - 1.0, 0.0))
    w = np.concatenate([[1.0], trust])
    z = (w[:, None] * P).sum(0) / w.sum()
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
