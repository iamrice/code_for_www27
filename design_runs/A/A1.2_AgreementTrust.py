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
    C = xs.shape[-1]
    Z = np.concatenate([xs[None], Y], 0)
    med = np.median(Z, 0)
    ref = (0.5 * xs + 0.5 * med).argmax(-1)
    agree = (Y.argmax(-1) == ref[None]).mean(1)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = agree
    else:
        trust = np.minimum(agree, 0.5 * prev + 0.5 * agree)
    state['trust'] = trust
    amax = float(trust.max())
    lo = max(2.0 / C, 0.5 * amax)
    hi = max(lo + 0.05, 0.85 * amax)
    w = np.clip((trust - lo) / (hi - lo), 0.0, 1.0)
    out = (xs + (w[:, None, None] * Y).sum(0)) / (1.0 + w.sum())
    return out
