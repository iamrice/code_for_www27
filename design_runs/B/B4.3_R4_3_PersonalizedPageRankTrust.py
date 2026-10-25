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
    lab = Z.argmax(-1)
    AG = (lab[:, None, :] == lab[None, :, :]).mean(-1)
    kappa = np.clip((AG - 1.0 / C) / (1.0 - 1.0 / C), 0.0, 1.0)
    vv = np.concatenate([[0.0], viol])
    A = (kappa ** 2) * np.clip(1.0 - H, 0.0, 1.0) * np.exp(-(vv[:, None] + vv[None, :]) / 0.05)
    np.fill_diagonal(A, 0.0)
    rs = A.sum(1, keepdims=True)
    Pm = np.where(rs > 1e-12, A / np.maximum(rs, 1e-12), 0.0)
    alpha = 0.25
    e0 = np.zeros(n)
    e0[0] = 1.0
    pi = e0.copy()
    for _ in range(40):
        nxt = alpha * e0 + (1.0 - alpha) * (pi @ Pm)
        nxt[0] += max(0.0, 1.0 - float(nxt.sum()))
        pi = nxt
    r = pi[1:]
    rmx = float(r.max()) if k > 0 else 0.0
    if rmx <= 1e-12:
        w_ppr = np.zeros(k)
    else:
        w_ppr = np.clip((r / rmx - 0.2) / 0.5, 0.0, 1.0)
    a0 = A[0, 1:]
    amx = float(a0.max()) if k > 0 else 0.0
    if amx <= 1e-12:
        w_dir = np.zeros(k)
    else:
        w_dir = np.clip(a0 / (0.6 * amx), 0.0, 1.0)
    w_new = w_ppr * w_dir * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum <= 1e-6:
        return xs
    nb = (trust[:, None, None] * P).sum(0) / tsum
    s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
    w_self = 0.4 + 0.6 * s_agree
    return (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)
