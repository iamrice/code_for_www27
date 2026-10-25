def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    med = np.median(Z, 0)
    drow = np.abs(Z - med[None]).sum(-1)              # (n, m)
    dist = drow.mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist / (2.0 * d_ref)) ** 4)
    lab = Z.argmax(-1)
    lab_c = med.argmax(-1)
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab[0][None]).mean(1))
    w_agree = np.clip((agree - 2.5 / C) / (2.5 / C), 0.0, 1.0)
    w_new = (w_dist * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tr = np.concatenate([[1.0], trust])
    s = np.median(drow, 0) + 0.05                      # (m,)
    u = 1.0 / (1.0 + (drow / (3.0 * s[None])) ** 2)
    W = tr[:, None] * u
    W[0] = np.maximum(W[0], 0.5)
    return (W[:, :, None] * Z).sum(0) / W.sum(0)[:, None]
