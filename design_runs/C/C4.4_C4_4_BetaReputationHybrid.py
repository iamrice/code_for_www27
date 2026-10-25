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
    if k <= 3:
        Pm = np.concatenate([x[None], x[None], X], 0)
    else:
        Pm = np.concatenate([x[None], X], 0)
    med = np.median(Pm, 0)
    r = np.abs(X - med[None]).sum(2)                          # (k, m)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    acc = (r <= 2.0 * sc[None]).mean(1)
    lab = med.argmax(1)
    ag = (X.argmax(2) == lab[None]).mean(1)
    ag0 = float((x.argmax(1) == lab).mean())
    agr = np.clip(ag / (max(ag0, float(ag.max())) + 1e-12), 0.0, 1.0)
    ev = np.minimum(acc, agr)
    # temporal drift evidence
    prevX = state.get('prevX')
    prevx = state.get('prevx')
    if prevX is not None and prevX.shape == X.shape and prevx is not None:
        dr = np.abs(X - prevX).sum(2).mean(1)
        own = float(np.abs(x - prevx).sum(1).mean())
        base = float(np.median(np.concatenate([[own], dr]))) + 0.05
        dflag = np.clip((dr - 2.5 * base) / (2.5 * base), 0.0, 1.0)
        ev = ev * (1.0 - dflag)
    state['prevX'] = X.copy()
    state['prevx'] = x.copy()
    a = state.get('a')
    b = state.get('b')
    if a is None or a.shape != ev.shape:
        a = np.full(k, 2.0)
        b = np.full(k, 0.5)
    a = a + ev
    b = b + 2.0 * (1.0 - ev)
    state['a'] = a
    state['b'] = b
    mpost = a / (a + b)
    w = np.clip((mpost - 0.5) / 0.3, 0.0, 1.0) * np.clip((ev - 0.35) / 0.3, 0.0, 1.0)
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)
    W = w[:, None] * rw
    if k >= 3:
        ws = np.clip((3.0 * sc - r0) / (2.0 * sc), 0.4, 1.0)
        ws = np.where(W.sum(0) >= 1.5, ws, 1.0)
    else:
        ws = np.ones_like(r0)
    out = (ws[:, None] * x + (W[..., None] * X).sum(0)) / (ws + W.sum(0))[..., None]
    return san(out)
