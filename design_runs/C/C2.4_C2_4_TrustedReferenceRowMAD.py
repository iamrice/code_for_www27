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
    prev = state.get('trust')
    if prev is None or prev.shape != (k,):
        prev = np.ones(k)
    sel = prev >= 0.5
    if sel.sum() == 0:
        sel = np.ones(k, dtype=bool)
    ref_set = np.concatenate([x[None], X[sel]], 0)
    med = np.median(ref_set, 0)
    r = np.abs(X - med[None]).sum(2)                         # (k, m)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r[sel]], 0), 0) + 0.04
    mult = 2.5 - 1.0 * t / max(T - 1, 1)
    rowacc = np.clip((mult + 0.5) - r / sc[None], 0.0, 1.0)  # soft ramp around mult
    frac = 1.0 - (r <= mult * sc[None]).mean(1)
    susp = state.get('susp')
    if susp is None or susp.shape != frac.shape:
        susp = frac.copy()
    else:
        susp = np.maximum(frac, 0.8 * susp + 0.2 * frac)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    state['trust'] = trust
    W = trust[:, None] * rowacc
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
