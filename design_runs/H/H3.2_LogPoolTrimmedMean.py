def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    n = k + 1
    Q = np.concatenate([x_self[None], X_nb]).reshape(n, m, C).astype(float)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    Q = np.clip(Q, 1e-6, None)
    Q = Q / Q.sum(-1, keepdims=True)
    L = np.log(Q)
    b = int(math.floor(0.2 * n)) if n >= 5 else 0
    S = np.sort(L, axis=0)
    M = S[b:n - b].mean(0)
    out = np.exp(M - M.max(-1, keepdims=True))
    out = out / out.sum(-1, keepdims=True)
    out = np.nan_to_num(out, nan=1.0 / C, posinf=1.0 / C, neginf=1.0 / C)
    return out.reshape(x_self.shape)
