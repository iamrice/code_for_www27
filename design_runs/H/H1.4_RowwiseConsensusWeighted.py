def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]
    P = np.concatenate([x_self[None], X_nb])
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    s = P.sum(-1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-12), 1.0 / C)
    H = -(P * np.log(P + 1e-12)).sum(-1)
    conf = np.clip(1.0 - H / math.log(C), 0.0, 1.0)
    med = np.median(P, 0)
    dev = 0.5 * np.abs(P - med).sum(-1)
    tau = np.median(dev) + 1e-3
    wgt = conf * np.exp(-dev / tau) + 1e-6
    num = (wgt[..., None] * P).sum(0)
    den = wgt.sum(0)[..., None]
    return num / den
