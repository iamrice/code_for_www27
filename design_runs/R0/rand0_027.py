def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    A = np.concatenate([x_self[None], X_nb])
    S = np.sort(A, 0); b = int(np.floor(0.17 * len(A))); c = S[b:len(A) - b].mean(0) if len(A) - 2 * b >= 1 else np.median(A, 0)
    d = (X_nb.argmax(-1) != c.argmax(-1)[None]).mean(1) + 1e-3 * np.abs(X_nb - c).reshape(k, -1).mean(1)
    med = np.median(d, 0) + 1e-12
    w = (d <= 2.42 * med).astype(float)
    w = 0.53 * state.get('w', w) + (1 - 0.53) * w
    state['w'] = w
    W = w[:, None, None]
    agg = (W * X_nb).sum(0) / np.maximum(W.sum(0), 1e-12)
    has = (W.sum(0) > 1e-12)
    agg = np.where(has, agg, x_self)
    out = 0.27 * x_self + 0.73 * agg
    out = x_self + 0.6 * (out - x_self)
    out = np.clip(out, 0, None); out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
    return out
