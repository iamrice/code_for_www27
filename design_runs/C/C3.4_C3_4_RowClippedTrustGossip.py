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
    r = np.abs(X - med[None]).sum(2)                         # (k, m)
    r0 = np.abs(x - med).sum(1)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    frac = 1.0 - (r <= 2.0 * sc[None]).mean(1)
    susp = state.get('susp')
    if susp is None or susp.shape != frac.shape:
        susp = frac.copy()
    else:
        susp = np.maximum(frac, 0.8 * susp + 0.2 * frac)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)            # (k,)
    D = X - x[None]                                          # (k, m, C)
    dn = np.abs(D).sum(2)                                    # (k, m)
    tau = np.maximum(np.median(dn, 0), 2.0 * sc) + 0.02      # (m,)
    c = np.minimum(1.0, tau[None] / np.maximum(dn, 1e-12))   # (k, m)
    W = trust[:, None] * c
    upd = (W[..., None] * D).sum(0) / (1.0 + trust.sum())
    return san(x + upd)
