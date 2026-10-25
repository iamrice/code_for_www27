def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    S = np.sqrt(xs)
    R = np.sqrt(Y)
    D = R - S[None]                                   # (k, m, C)
    dr = np.linalg.norm(D, axis=2)                    # (k, m)
    q = 0.4 if k >= 3 else 0.5
    tau_row = np.quantile(dr, q, axis=0)              # (m,)
    tau_glob = float(np.median(dr))
    tau = 0.5 * tau_row + 0.5 * tau_glob + 1e-12
    sr = np.minimum(1.0, tau[None, :] / np.maximum(dr, 1e-12))   # (k, m)
    acc = sr.mean(1)
    r = state.get('rep')
    if r is None or len(r) != k:
        r = acc.copy()
    else:
        r = np.where(acc < r, 0.4 * r + 0.6 * acc, 0.7 * r + 0.3 * acc)
    state['rep'] = r
    rw = np.clip((r - 0.3) / 0.5, 0.0, 1.0)            # hard-gate low-reputation neighbours
    w = rw[:, None] * sr                               # (k, m)
    out = S + (w[:, :, None] * D).sum(0) / (1.0 + rw.sum())
    out = np.clip(out, 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
