def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    m, C = xs.shape
    Z = np.concatenate([xs[None], Y], 0)
    n = k + 1
    A = Z.argmax(-1)
    OH = np.eye(C)[A]                       # (n, m, C)
    w = np.ones(n)
    for _ in range(4):
        votes = np.tensordot(w, OH, axes=(0, 0))   # (m, C)
        maj = votes.argmax(-1)
        acc = (A == maj[None]).mean(1)
        a = np.clip(acc, 1.0 / C + 1e-3, 1.0 - 1e-3)
        w_new = np.maximum(0.0, np.log(a * (C - 1) / (1.0 - a)))
        if w_new.sum() <= 1e-12:
            break
        w = w_new
    wmax = max(float(w.max()), 1e-12)
    ev = w[1:] / wmax
    selfrel = float(w[0] / wmax)
    r = state.get('rep')
    if r is None or len(r) != k:
        r = ev.copy()
    else:
        r = np.where(ev < r, 0.4 * r + 0.6 * ev, 0.7 * r + 0.3 * ev)
    state['rep'] = r
    D = Y - xs[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    trusted = r > 0.5
    if trusted.any():
        tau = float(np.median(nrm[trusted]))
    else:
        tau = float(np.quantile(nrm, 0.25))
    tau += 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    g = np.clip((r - 0.25) / 0.5, 0.0, 1.0)
    sw = max(0.3, min(1.0, selfrel))
    cw = g * scale
    out = xs + (cw[:, None, None] * D).sum(0) / (sw + g.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
