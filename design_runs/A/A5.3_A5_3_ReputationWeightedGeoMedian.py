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
    P = np.concatenate([xs[None], Y], 0).reshape(k + 1, -1)
    wt = np.concatenate([[1.0], r])
    z = xs.reshape(-1).copy()
    floor = 1e-3 * med + 1e-9
    for _ in range(8):
        d = np.linalg.norm(P - z[None], axis=1)
        a = wt / np.maximum(d, floor)
        z = (a[:, None] * P).sum(0) / a.sum()
    z = z.reshape(xs.shape)
    step = 0.6 + 0.3 * float(t) / max(T - 1, 1)
    out = xs + step * (z - xs)
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)
