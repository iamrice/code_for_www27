def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(P, 0)
    mad = 1.4826 * np.median(np.abs(P - med), 0) + 1e-6
    c = 1.5 + 2.0 * math.exp(-2.0 * t / T)
    inl = np.abs(P - med) <= c * mad
    inl[0] = True
    cnt = inl.sum(0)
    s = np.where(inl, P, 0.0).sum(0)
    out = s / np.maximum(cnt, 1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)
