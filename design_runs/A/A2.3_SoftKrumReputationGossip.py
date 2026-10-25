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
    n = k + 1
    F = np.sqrt(np.concatenate([xs[None], Y], 0)).reshape(n, -1)
    G = F @ F.T
    sq = np.diag(G)
    Dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))
    h = max(1, (n - 1) // 2)
    Ds = np.sort(Dm + np.diag(np.full(n, np.inf)), axis=1)[:, :h]
    score = Ds.mean(1)
    base = float(np.median(score)) + 1e-12
    dens = np.exp(-4.0 * np.maximum(score - score.min(), 0.0) / base)
    cur = dens[1:]
    r = state.get('rep')
    if r is None or len(r) != k:
        r = cur.copy()
    else:
        r = np.minimum(cur + 0.1, 0.5 * r + 0.5 * cur)
    r = np.clip(r, 0.0, 1.0)
    state['rep'] = r
    D = F[1:] - F[0][None]
    nrm = Dm[0, 1:]
    tau = float(np.quantile(nrm, 0.5)) + 1e-12
    sc = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    w = r * sc
    out = F[0] + (w[:, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out.reshape(xs.shape), 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
