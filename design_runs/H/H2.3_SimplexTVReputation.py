def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    C = x_self.shape[-1]
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None).reshape(k + 1, -1, C)
    s = P.sum(-1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-12), 1.0 / C)
    ref = np.median(P, 0)
    tv = 0.5 * np.abs(P - ref).sum(-1).mean(-1)
    rep = state.get('rep')
    if rep is None or len(rep) != k:
        rep = np.ones(k)
    rep = np.asarray(rep, dtype=float)
    u = tv[1:] / (np.median(tv[1:]) + 1e-12)
    good = np.exp(-np.maximum(u - 1.5, 0.0) ** 2)
    rep = 0.7 * rep + 0.3 * good
    state['rep'] = rep
    tau = np.median(tv) + 1e-12
    w = rep * np.minimum(1.0, tau / np.maximum(tv[1:], 1e-12))
    Xn = P[1:]
    num = P[0] + (w[:, None, None] * Xn).sum(0)
    den = 1.0 + w.sum()
    out = num / den
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
