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
    r0 = np.abs(x - med).sum(1)                              # (m,)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rw = np.clip((3.0 * sc[None] - r) / sc[None], 0.0, 1.0)  # soft row weights
    frac = 1.0 - (r <= 2.0 * sc[None]).mean(1)               # row rejection rate
    lab = med.argmax(1)
    agree = (X.argmax(2) == lab[None]).mean(1)
    ref = max(float((x.argmax(1) == lab).mean()), float(agree.max())) + 1e-12
    dis = np.clip(1.0 - agree / ref, 0.0, 1.0)               # relative argmax disagreement
    raw = np.maximum(frac, dis)
    susp = state.get('susp')
    if susp is None or susp.shape != raw.shape:
        susp = raw.copy()
    else:
        susp = np.maximum(raw, 0.8 * susp + 0.2 * raw)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    W = trust[:, None] * rw                                  # (k, m)
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)
