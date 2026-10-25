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
    H = np.sqrt(np.maximum(0.5 * ((S[:, None] - S[None]) ** 2).sum(-1), 0.0)).mean(-1)  # (n,n) in [0,1]
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
    center = (w_core[:, None, None] * Z).sum(0) / w_core.sum()
    lab = Z.argmax(-1)
    lab_c = center.argmax(-1)
    lab_s = lab[0]
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab_s[None]).mean(1))
    lo, hi = 2.5 / C, 5.0 / C
    w_agree = np.clip((agree - lo) / (hi - lo), 0.0, 1.0)
    w_new = (w_core * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    w_self = max(0.5, float(w_core[0]))
    out = (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + trust.sum())
    return out
