def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

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
    # weighted geometric median per row (Weiszfeld)
    gw = np.concatenate([[1.0], 0.25 + 0.75 * w])             # (n,)
    if k <= 3:
        gw[0] = 2.0
    z = np.median(P, 0)
    for _ in range(8):
        d = np.sqrt(((P - z[None]) ** 2).sum(2)) + 1e-6       # (n, m)
        a = gw[:, None] / d
        z = (a[..., None] * P).sum(0) / a.sum(0)[..., None]
    # centred clipping around the geometric median
    dev = P - z[None]
    dn = np.sqrt((dev ** 2).sum(2))                           # (n, m)
    mask = np.concatenate([[True], w >= 0.5])
    tau = np.nanmedian(np.where(mask[:, None], dn, np.nan), axis=0) + 0.02
    c = np.minimum(1.0, tau[None] / np.maximum(dn, 1e-12))   # (n, m)
    v = np.concatenate([[1.0], w])
    V = v[:, None] * c
    out = z + (V[..., None] * dev).sum(0) / v.sum()
    return san(out)
