def aggregate(x_self, X_nb, t, T, state):
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / P.shape[-1])
    S = np.sqrt(P)
    ref = np.median(S, 0)
    d = np.sqrt(((S - ref) ** 2).sum(-1)).mean(-1)
    scale = np.median(d) + 1e-12
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr - d) / (0.5 * thr + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    out = (w[:, None, None] * P).sum(0) / w.sum()
    return out.reshape(x_self.shape)
