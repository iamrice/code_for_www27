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
    D = Y - xs[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    if k >= 2:
        F = np.concatenate([xs[None], Y], 0).reshape(k + 1, -1)
        sq = (F * F).sum(1)
        G = F @ F.T
        Dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))
        dpeer = np.empty(k)
        for j in range(k):
            row = np.delete(Dm[j + 1], j + 1)
            dpeer[j] = np.median(row)
        s = 0.5 * nrm + 0.5 * dpeer
    else:
        s = nrm.copy()
    ts = float(np.quantile(s, 0.25)) + 1e-12
    base = np.minimum(1.0, ts / np.maximum(s, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = base.copy()
    else:
        r = np.where(base < r, 0.4 * r + 0.6 * base, 0.7 * r + 0.3 * base)
    state['rep'] = r
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
