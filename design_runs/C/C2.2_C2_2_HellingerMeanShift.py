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
    P = np.concatenate([x[None], X], 0)                      # (n, m, C)
    S = np.sqrt(P)
    y = np.median(S, 0)
    w = None
    for _ in range(4):
        d2 = ((S - y[None]) ** 2).sum(2)                     # (n, m)
        h2 = np.median(d2, 0) + 1e-3
        w = np.exp(-0.5 * d2 / h2[None])
        y = (w[..., None] * S).sum(0) / (w.sum(0)[..., None] + 1e-12)
    d2 = ((S - y[None]) ** 2).sum(2)
    h2 = np.median(d2, 0) + 1e-3
    w = np.exp(-0.5 * d2 / h2[None])                         # final kernel weights
    g = w.mean(1)                                            # node-level kernel mass
    ratio = g[1:] / (np.median(g) + 1e-12)
    rep = state.get('rep')
    if rep is None or rep.shape != ratio.shape:
        rep = ratio.copy()
    else:
        rep = np.minimum(ratio, 0.7 * rep + 0.3 * ratio)
    state['rep'] = rep
    trust = np.clip((rep - 0.2) / 0.4, 0.0, 1.0)
    Wn = w * np.concatenate([[1.0], trust])[:, None]
    Wn[0] += 0.05                                            # small self anchor
    out = (Wn[..., None] * P).sum(0) / (Wn.sum(0)[..., None] + 1e-12)
    return san(out)
