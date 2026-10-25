def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    xs = np.nan_to_num(x_self.reshape(-1).astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    Xf = np.nan_to_num(X_nb.reshape(k, -1).astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    D = Xf - xs
    nrm = np.linalg.norm(D, axis=1)
    med = np.median(np.concatenate([xs[None], Xf]), 0)
    dm = med - xs
    cos = (D @ dm) / (nrm * np.linalg.norm(dm) + 1e-12)
    gate = np.clip(cos, 0.0, 1.0) ** 2
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    w = gate * scale
    out = xs + (w[:, None] * D).sum(0) / (k + 1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
