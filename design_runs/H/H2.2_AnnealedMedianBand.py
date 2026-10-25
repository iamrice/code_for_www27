def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    scale = np.median(d) + 1e-12
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    keep = d <= thr
    keep[0] = True
    out = P[keep].mean(0)
    return out.reshape(x_self.shape)
