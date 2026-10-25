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
    med = float(np.median(nrm)) + 1e-12
    loss = np.clip(np.log(np.maximum(nrm, 1e-12) / med), 0.0, 3.0)
    L = state.get('L')
    n = state.get('n', 0)
    if L is None or len(L) != k:
        L = np.zeros(k)
        n = 0
    L = L + loss
    n = n + 1
    state['L'] = L
    state['n'] = n
    Lb = L / n
    r = np.exp(-2.5 * (Lb - Lb.min()))
    lam = 1.0 - 0.5 * float(t) / max(T - 1, 1)
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    v = xs.copy()
    for it in range(3):
        Dv = Y - v[None]
        nv = np.linalg.norm(Dv.reshape(k, -1), axis=1)
        s = np.minimum(1.0, tau_j / np.maximum(nv, 1e-12))
        step = lam * (xs - v) + ((r * s)[:, None, None] * Dv).sum(0)
        v = v + step / (lam + r.sum())
    out = np.clip(v, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
