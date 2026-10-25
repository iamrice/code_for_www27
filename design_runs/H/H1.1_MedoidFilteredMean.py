def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    S = np.concatenate([x_self[None], X_nb])
    med = np.median(S, 0)
    d = np.linalg.norm((S - med).reshape(k + 1, -1), axis=1)
    keep = max(1, int(np.ceil(0.6 * (k + 1))))
    idx = np.argsort(d, kind='stable')[:keep]
    return S[idx].mean(0)
