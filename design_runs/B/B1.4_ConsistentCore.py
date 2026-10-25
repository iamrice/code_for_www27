def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.concatenate([[0.0], np.abs(R - P).sum(-1).mean(1)])
    Z = np.concatenate([xs[None], P], 0)
    n = k + 1
    Dm = np.abs(Z[:, None] - Z[None]).sum(-1).mean(-1)   # (n, n) mean row L1 in [0, 2]
    Dm = Dm + 10.0 * (viol[:, None] + viol[None, :])
    np.fill_diagonal(Dm, 0.0)
    q = max(1, (n + 1) // 2 - 1)
    score = np.sort(Dm, 1)[:, 1:q + 1].sum(1)
    score[0] *= 0.9
    med = int(np.argmin(score))
    d = Dm[med]
    radius = min(1.4, max(2.0 * float(np.sort(d)[q]), 0.05))
    members = d <= radius
    members[med] = True
    return Z[members].mean(0)
