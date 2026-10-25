def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    D = np.linalg.norm((X_nb - x_self).reshape(k, -1), axis=1)
    z = D / (np.median(D) + 1e-12)
    w_inst = 1.0 / (1.0 + (z / 1.5) ** 4)
    trust = state.get('trust')
    if trust is None or len(trust) != k:
        trust = np.ones(k)
    trust = 0.5 * np.asarray(trust, dtype=float) + 0.5 * w_inst
    state['trust'] = trust
    return (x_self + np.tensordot(trust, X_nb, axes=1)) / (1.0 + trust.sum())
