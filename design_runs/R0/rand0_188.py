def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    S = np.sort(A, 0); b = int(np.floor(0.28 * len(A))); c = S[b:len(A) - b].mean(0) if len(A) - 2 * b >= 1 else np.median(A, 0)
    d = np.abs(X_nb - c).sum(-1)
    med = np.median(d, 0) + 1e-12
    w = np.minimum(1.0, 0.61 * med / np.maximum(d, 1e-12))
    w = 0.76 * state.get('w', w) + (1 - 0.76) * w
    state['w'] = w
    W = w[:, :, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.33 * x_self + 0.6699999999999999 * agg
    out = x_self + 0.7 * (out - x_self)
    return out
