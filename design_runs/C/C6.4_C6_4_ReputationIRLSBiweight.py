def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    def wmedian(Q, wv):
        idx = np.argsort(Q, axis=0, kind='stable')
        Qs = np.take_along_axis(Q, idx, 0)
        cw = np.cumsum(wv[idx], 0)
        j = (cw < 0.5 * cw[-1][None]).sum(0)
        j = np.minimum(j, Q.shape[0] - 1)
        return np.take_along_axis(Qs, j[None], 0)[0]

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    P = np.concatenate([x[None], X], 0)
    n = k + 1
    L = P.argmax(2)
    A = (L[:, None, :] == L[None, :, :]).mean(2)
    D = 0.5 * np.abs(P[:, None] - P[None]).sum(3).mean(2)
    S = 0.5 * A + 0.5 * (1.0 - D)
    So = S.copy()
    np.fill_diagonal(So, np.nan)
    peer = np.nanmedian(So, axis=0)
    score = 0.5 * S[0, 1:] + 0.5 * peer[1:]
    ref = max(float(peer[0]), float(score.max())) + 1e-12
    rel = score / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    wts = np.concatenate([[1.0], w])                          # (n,)
    selfw = 1.0 if k >= 3 else 2.0
    wts[0] = selfw
    center = wmedian(P, np.maximum(wts, 0.02))
    trusted = np.concatenate([[True], w >= 0.5])
    Wt = None
    for _ in range(3):
        R = np.abs(P - center[None]).sum(2)                   # (n, m)
        Rm = np.where(trusted[:, None], R, np.nan)
        s = np.nanmedian(Rm, axis=0) + 0.05                   # (m,)
        u = R / (3.0 * s[None])
        bw = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        Wt = wts[:, None] * bw
        Wt[0] = np.maximum(Wt[0], 0.3)
        center = (Wt[..., None] * P).sum(0) / Wt.sum(0)[..., None]
    return san(center)
