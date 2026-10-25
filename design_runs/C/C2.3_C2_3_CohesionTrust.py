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
    S = np.sqrt(P)
    n = k + 1
    Dm = np.zeros((n, n))
    for a in range(n):
        Dm[a] = np.sqrt(((S - S[a][None]) ** 2).sum(2)).mean(1)
    score = np.array([np.median(np.delete(Dm[a], a)) for a in range(n)])
    ms = float(np.median(score)) + 1e-3
    raw = np.clip((score[1:] - 1.2 * ms) / (0.8 * ms), 0.0, 1.0)
    # also penalise neighbours far from self relative to typical cohesion
    raw = np.maximum(raw, np.clip((Dm[0, 1:] - 3.0 * ms) / (2.0 * ms), 0.0, 1.0))
    susp = state.get('susp')
    if susp is None or susp.shape != raw.shape:
        susp = raw.copy()
    else:
        susp = np.maximum(raw, 0.8 * susp + 0.2 * raw)
    state['susp'] = susp
    trust = 1.0 - susp
    out = (x + (trust[:, None, None] * X).sum(0)) / (1.0 + trust.sum())
    return san(out)
