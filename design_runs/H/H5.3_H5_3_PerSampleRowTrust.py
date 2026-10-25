def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / P.shape[-1])
    ref = np.median(P, 0)
    d = np.abs(P - ref).sum(-1)
    scale = np.median(d, 0) + 1e-9
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr[None, :] - d) / (0.5 * thr[None, :] + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    out = (w[..., None] * P).sum(0) / w.sum(0)[:, None]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, 0.0, None)
    rs = out.sum(-1, keepdims=True)
    out = np.where(rs > 1e-12, out / np.maximum(rs, 1e-12), 1.0 / out.shape[-1])
    return out.reshape(x_self.shape)
