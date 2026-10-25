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
    med = np.median(np.concatenate([x[None], X], 0), 0)
    lab = med.argmax(1)
    agree = (X.argmax(2) == lab[None]).mean(1)       # (k,)
    a_self = float((x.argmax(1) == lab).mean())
    rep = state.get('rep')
    if rep is None or rep.shape != agree.shape:
        rep = agree.copy()
    else:
        rep = 0.6 * rep + 0.4 * agree
    state['rep'] = rep
    eff = np.minimum(rep, agree)
    ref = max(a_self, float(agree.max())) + 1e-12
    w = np.clip((eff - 0.5 * ref) / (0.4 * ref), 0.0, 1.0)
    out = (x + (w[:, None, None] * X).sum(0)) / (1.0 + w.sum())
    return san(out)
