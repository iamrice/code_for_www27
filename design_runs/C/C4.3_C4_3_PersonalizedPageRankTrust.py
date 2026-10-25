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
    S = np.clip(0.5 * A + 0.5 * (1.0 - D), 0.0, 1.0)
    K = S ** 6
    np.fill_diagonal(K, 0.0)
    colsum = K.sum(0, keepdims=True)
    M = K / np.maximum(colsum, 1e-12)                         # column-stochastic
    e0 = np.zeros(n)
    e0[0] = 1.0
    v = e0.copy()
    alpha = 0.7
    for _ in range(30):
        v = (1.0 - alpha) * e0 + alpha * (M @ v)
    vn = v[1:]
    ppr = vn / (vn.max() + 1e-12)
    direct = S[0, 1:] / (max(float(S[0, 1:].max()), float(np.median(S[1:, 1:] if k > 1 else S[0, 1:]))) + 1e-12)
    rel = np.minimum(np.sqrt(ppr), np.clip(direct, 0.0, 1.0))
    rel = 0.5 * rel + 0.5 * np.clip(direct, 0.0, 1.0) * (ppr > 0.15)
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.5) / 0.3, 0.0, 1.0)
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    r = np.abs(X - med[None]).sum(2)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = w[:, None] * rw
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
