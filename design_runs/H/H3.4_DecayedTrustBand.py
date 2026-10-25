def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    s = np.median(d) + 1e-12
    u = d[1:] / s
    evidence = np.exp(-np.maximum(u - 1.0, 0.0) ** 2)
    rep = state.get('rep')
    if rep is None or len(rep) != k:
        rep = np.ones(k)
    rep = 0.8 * np.asarray(rep, dtype=float) + 0.2 * evidence
    state['rep'] = rep
    w = rep * np.minimum(1.0, (1.5 * s) / np.maximum(d[1:], 1e-12))
    num = P[0] + (w[:, None] * P[1:]).sum(0)
    out = num / (1.0 + w.sum())
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
