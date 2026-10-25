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
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    center = np.median(Z, 0)
    lab_c = center.argmax(-1)
    lab_s = xs.argmax(-1)
    lab_n = P.argmax(-1)
    agree = 0.5 * ((lab_n == lab_c[None]).mean(1) + (lab_n == lab_s[None]).mean(1))
    lo, hi = 2.5 / C, 5.0 / C                       # chance-relative agreement thresholds
    w_agree = np.clip((agree - lo) / (hi - lo), 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)   # mean row L1 to centre, (n,)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_valid = np.exp(-viol / 0.05)
    w_new = w_agree * w_dist * w_valid
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    num = xs + (trust[:, None, None] * P).sum(0)
    return num / (1.0 + trust.sum())
