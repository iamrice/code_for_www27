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
    m, C = x.shape
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
    # row gating against coordinate-wise median
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    R = np.abs(P - med[None]).sum(2)                          # (n, m)
    sc = np.median(R, 0) + 0.05
    rw = np.clip((3.0 * sc[None] - R) / sc[None], 0.0, 1.0)
    rw[0] = 1.0
    G = wts[:, None] * rw                                     # (n, m)
    # weighted label vote
    onehot = (L[..., None] == np.arange(C)[None, None, :]).astype(float)  # (n, m, C)
    votes = (G[..., None] * onehot).sum(0)                    # (m, C)
    tot = votes.sum(1) + 1e-12
    win = votes.argmax(1)
    margin = votes.max(1) / tot                               # (m,)
    agree = (L == win[None]).astype(float)                    # (n, m)
    strength = np.clip((margin - 0.4) / 0.3, 0.0, 1.0)        # 0 -> no vote effect
    dis = 1.0 - 0.8 * strength                                # weight of dissenters
    F = agree + (1.0 - agree) * dis[None]
    Wn = G * F
    # never let self vanish entirely
    Wn[0] = np.maximum(Wn[0], 0.15)
    out = (Wn[..., None] * P).sum(0) / Wn.sum(0)[..., None]
    return san(out)
