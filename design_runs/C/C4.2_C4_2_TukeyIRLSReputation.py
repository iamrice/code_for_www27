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
    P = np.concatenate([x[None], X], 0)                       # (n, m, C)
    if k <= 3:
        Pm = np.concatenate([x[None], P], 0)
    else:
        Pm = P
    med = np.median(Pm, 0)
    r = np.abs(P - med[None]).sum(2)                          # (n, m)
    sc = np.median(r, 0) + 0.05                               # (m,)
    # node reputation: fraction of rows inside 2*scale, plus argmax agreement
    inside = (r[1:] <= 2.0 * sc[None]).mean(1)
    lab = med.argmax(1)
    ag = (X.argmax(2) == lab[None]).mean(1)
    ag0 = float((x.argmax(1) == lab).mean())
    agr = np.clip(ag / (max(ag0, float(ag.max())) + 1e-12), 0.0, 1.0)
    ev = np.minimum(inside, agr)
    rep = state.get('rep')
    if rep is None or rep.shape != ev.shape:
        rep = ev.copy()
    else:
        rep = np.minimum(ev, 0.75 * rep + 0.25 * ev)
    state['rep'] = rep
    nw = np.concatenate([[1.0], np.clip((rep - 0.45) / 0.35, 0.0, 1.0)])   # (n,)
    mu = med
    c = 3.5
    for _ in range(4):
        rr = np.abs(P - mu[None]).sum(2)                      # (n, m)
        s = np.median(rr, 0) + 0.05
        u = rr / (c * s[None])
        bw = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        Wt = nw[:, None] * bw
        Wt[0] = np.maximum(Wt[0], 0.5)                        # self anchor
        mu = (Wt[..., None] * P).sum(0) / Wt.sum(0)[..., None]
    return san(mu)
