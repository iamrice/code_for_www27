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
    lab = Z.argmax(-1)
    d = np.empty(n)
    a = np.empty(n)
    for j in range(n):
        c = np.median(np.delete(Z, j, 0), 0)
        d[j] = np.abs(Z[j] - c).sum(-1).mean()
        a[j] = (lab[j] == c.argmax(-1)).mean()
    a_self = (lab == lab[0][None]).mean(1)
    a = 0.5 * (a + a_self)
    ref = max(float(np.median(d)), 0.02)
    g_dist = np.clip(2.0 - d / ref, -1.0, 1.0)
    g_agree = np.clip((a - 3.0 / C) / (2.0 / C), -1.0, 1.0)
    g = np.minimum(g_dist, g_agree)[1:] - viol / 0.05
    g = np.clip(g, -3.0, 1.0)
    L = state.get('L')
    if L is None or len(L) != k:
        L = np.zeros(k)
    L = np.clip(L + np.where(g < 0, 3.0 * g, g), -10.0, 4.0)
    state['L'] = L
    trust = 1.0 / (1.0 + np.exp(-3.0 * (L - 0.5)))
    return (xs + (trust[:, None, None] * P).sum(0)) / (1.0 + trust.sum())
