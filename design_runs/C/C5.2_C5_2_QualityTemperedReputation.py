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
    Ao = A.astype(float).copy()
    np.fill_diagonal(Ao, np.nan)
    peer = np.nanmedian(Ao, axis=0)
    score = 0.5 * A[0, 1:] + 0.5 * peer[1:]
    ref = max(float(peer[0]), float(score.max())) + 1e-12
    rel = score / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    # robust anchor (self-padded for small degree)
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    cons = med.argmax(1)
    # quality: agreement with robust consensus label, softmax-tempered
    agr = (L == cons[None]).mean(1)                           # (n,)
    trusted = np.concatenate([[True], w >= 0.5])
    amax = float(agr[trusted].max())
    q = np.exp(np.minimum(agr - amax, 0.0) / 0.08)
    r = np.abs(X - med[None]).sum(2)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = (w * q[1:])[:, None] * rw
    ws0 = max(float(q[0]), 0.3) if k >= 2 else 1.0
    out = (ws0 * x + (W[..., None] * X).sum(0)) / (ws0 + W.sum(0))[..., None]
    return san(out)
