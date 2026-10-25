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
    n = k + 1
    S = np.sqrt(Z)
    BC = np.einsum('imc,jmc->ijm', S, S)
    H = np.sqrt(np.clip(1.0 - BC, 0.0, 1.0)).mean(-1)
    vv = np.concatenate([[0.0], viol])
    Dm = H + 5.0 * (vv[:, None] + vv[None, :])
    np.fill_diagonal(Dm, 0.0)
    q = max(1, (n + 1) // 2 - 1)
    sc = np.sort(Dm, 1)[:, 1:q + 1].mean(1)
    sc[0] *= 0.9
    med = int(np.argmin(sc))
    d = Dm[med]
    r = max(float(np.sort(d)[min(q, n - 1)]), 0.02)
    w_core = 1.0 / (1.0 + (d / (2.0 * r)) ** 4)
    w_core[med] = 1.0
    Wz = w_core[:, None, None] * Z
    tot = Wz.sum(0)
    wsum = float(w_core.sum())
    center = tot / wsum
    den = wsum - w_core
    loo = np.where(den[:, None, None] > 1e-6,
                   (tot[None] - Wz) / np.maximum(den, 1e-6)[:, None, None],
                   center[None])
    lab = Z.argmax(-1)
    lab_l = loo.argmax(-1)
    lab_c = center.argmax(-1)
    hitm = (lab == lab_l)
    L = np.eye(C)[lab_l]                                   # (n, m, C)
    cnt = L.sum(1)                                         # (n, C)
    hit = (L * hitm[..., None]).sum(1)                     # (n, C)
    pa = hitm.mean(1)
    rc = (hit + 4.0 * pa[:, None]) / (cnt + 4.0)
    rc = np.clip(rc, 1e-3, 1.0 - 1e-3)
    lw = np.clip(np.log(rc * (C - 1) / (1.0 - rc)), 0.0, 6.0)   # (n, C)
    u = (center @ lw.T).T                                  # (n, m)
    members = w_core > 0.5
    ref = max(float(np.median(u.mean(1)[members])), 1e-6)
    u = np.clip(u / ref, 0.0, 2.0)
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab[0][None]).mean(1))
    w_agree = np.clip((agree - 2.5 / C) / (2.5 / C), 0.0, 1.0)
    w_new = (w_core * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    g = np.concatenate([[max(0.5, float(w_core[0]))], trust])
    W = g[:, None] * u
    W[0] = np.maximum(W[0], 0.25)
    return (W[:, :, None] * Z).sum(0) / W.sum(0)[:, None]
