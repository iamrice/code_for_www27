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
    # contraction test
    d = D[0, 1:]                                              # mean TV distance to self, (k,)
    prev = state.get('dprev')
    cfl = state.get('cfl')
    if cfl is None or cfl.shape != d.shape:
        cfl = np.zeros(k)
    if prev is not None and prev.shape == d.shape:
        ratio = d / (prev + 0.02)
        base = float(np.quantile(ratio, 0.25)) if k >= 3 else float(min(ratio.min(), 0.9))
        excess = np.clip((ratio - max(base, 0.5) - 0.1) / 0.3, 0.0, 1.0)
        dmed = float(np.median(d))
        far = np.clip((d - dmed) / (dmed + 0.02), 0.0, 1.0)
        far = np.maximum(far, np.clip((d - 0.3) / 0.2, 0.0, 1.0))
        flag = excess * far
        cfl = np.maximum(flag, 0.8 * cfl + 0.2 * flag)
    state['dprev'] = d.copy()
    state['cfl'] = cfl
    rel = rel * (1.0 - 0.6 * cfl)
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
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
    if k >= 3:
        ws = np.clip((3.0 * sc - r0) / (2.0 * sc), 0.35, 1.0)
        ws = np.where(W.sum(0) >= 1.5, ws, 1.0)
    else:
        ws = np.ones_like(r0)
    out = (ws[:, None] * x + (W[..., None] * X).sum(0)) / (ws + W.sum(0))[..., None]
    return san(out)
