def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    m, C = x_self.shape
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, m, C).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / C)
    denom = max(math.log(C), 1e-12)
    H = -(P * np.log(P + 1e-12)).sum(-1) / denom
    conf = np.clip(1.0 - H.mean(-1), 0.0, 1.0)
    S = np.sqrt(P)
    ref = np.median(S, 0)
    raw = np.sqrt(((S - ref) ** 2).sum(-1)).mean(-1) / math.sqrt(2.0)
    r = raw / (np.median(raw) + 1e-12)
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    w = np.concatenate([[0.1 + conf[0]], band * (0.1 + conf[1:])])
    z = np.tensordot(w, P, axes=1) / (w.sum() + 1e-12)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, None)
    zs = z.sum(-1, keepdims=True)
    z = np.where(zs > 1e-12, z / np.maximum(zs, 1e-12), 1.0 / C)
    return z.reshape(x_self.shape)
