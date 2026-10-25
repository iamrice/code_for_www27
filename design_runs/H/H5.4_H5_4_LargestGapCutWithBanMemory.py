def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    keep = np.ones(k + 1, dtype=bool)
    if k + 1 >= 3:
        order = np.argsort(d)
        ds = d[order]
        lo = (k + 1) // 2
        ratios = (ds[lo:] + 1e-12) / (ds[lo - 1:k] + 1e-12)
        j = lo + int(np.argmax(ratios))
        if ratios.max() > 1.3:
            keep = np.zeros(k + 1, dtype=bool)
            keep[order[:j]] = True
    keep[0] = True
    ban_prev = state.get('ban')
    if ban_prev is None or len(ban_prev) != k:
        ban_prev = np.zeros(k)
    cut_nb = (~keep[1:]).astype(float)
    ban = 0.6 * np.asarray(ban_prev, dtype=float) + 0.4 * cut_nb
    state['ban'] = ban
    wnb = keep[1:].astype(float) * np.clip(1.0 - ban, 0.1, 1.0)
    w = np.concatenate([[1.0], wnb])
    z = (w[:, None] * P).sum(0) / w.sum()
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)
