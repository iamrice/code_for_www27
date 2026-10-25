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
    sc = np.zeros(n)
    for j in range(n):
        others = np.delete(P, j, 0)
        if j != 0 and k <= 3:
            others = np.concatenate([x[None], others], 0)     # self-pad small degree
        mj = np.median(others, 0)
        agr = float((P[j].argmax(1) == mj.argmax(1)).mean())
        tv = float(0.5 * np.abs(P[j] - mj).sum(1).mean())
        sc[j] = 0.5 * agr + 0.5 * (1.0 - tv)
    ref = max(float(sc[0]), float(sc[1:].max())) + 1e-12
    rel = sc[1:] / ref
    rep = state.get('rep')
    if rep is None or rep.shape != rel.shape:
        rep = rel.copy()
    else:
        rep = np.minimum(rel, 0.7 * rep + 0.3 * rel)
    state['rep'] = rep
    w = np.clip((rep - 0.55) / 0.3, 0.0, 1.0)
    keep = np.concatenate([[True], w >= 0.5])
    K = P[keep]
    nk = K.shape[0]
    if nk == 1:
        return x
    b = 1 if nk >= 5 else 0
    Ks = np.sort(K, axis=0)
    tm = Ks[b:nk - b].mean(0)
    # partial weight for borderline neighbours: blend them in softly
    part = (w > 0.0) & (w < 0.5)
    if part.any():
        pw = w[part]
        tm = (nk * tm + (pw[:, None, None] * X[part]).sum(0)) / (nk + pw.sum())
    return san(tm)
