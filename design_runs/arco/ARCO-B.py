"""ARCO: Anchored Robust Consensus Optimization.

Honest nodes jointly minimise
    F(X) = sum_i  alpha/2 ||x_i - y_i||^2  +  lambda/2 sum_i sum_{j in N_i} sum_s rho_delta(||x_i^s - x_j^s||)
over row-stochastic matrices x_i (one probability row per public sample s), by projected
(preconditioned) gradient descent.  One iteration = one message per neighbour.

    x_i <- Proj_simplex( x_i + gamma_i [ sum_j psi_delta(m_j - x_i) - kappa (x_i - y_i) ] )

psi_delta = gradient of rho_delta, applied row-wise:
    huber : psi(z) = z * min(1, delta/||z||)        (convex,   |psi| <= delta)
    log   : psi(z) = z * min(1, delta/||z||)^2      (redescending, weakly convex, |psi| <= delta)
kappa = alpha/lambda is the anchor strength; kappa = 0, huber, gamma=1/(k+1) is row-wise ClippedGossip.
"""


def proj_simplex_rows(V):
    """Euclidean projection of every row (last axis) onto the probability simplex."""
    shp = V.shape
    V = V.reshape(-1, shp[-1])
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    idx = np.arange(1, V.shape[1] + 1)
    cond = U - css / idx > 0
    rho = cond.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(len(V)), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0.0).reshape(shp)


def psi(D, delta, loss):
    r = np.sqrt((D ** 2).sum(-1, keepdims=True))  # (k, m, 1) row distances
    w = np.minimum(1.0, delta / np.maximum(r, 1e-12))
    if loss == "log":
        w = w * w
    elif loss == "l2":
        w = np.ones_like(w)
    return w * D


def make_arco(kappa=0.1, delta=1.0, loss="log", solver="mm", group="node", adaptive=True, q=0.25, project=True, center="self"):
    """solver: 'mm'  = proximal majorize-minimize step (exact minimiser of the local quadratic majoriser)
               'grad'= projected gradient step with step 1/(k+1+kappa)
       group : 'node' = robust loss on whole-message distance, 'row' = per public sample
       adaptive: delta_t = delta * quantile_q(neighbour distances) (scale-free); else delta is absolute."""
    def aggregate(x_self, X_nb, t, T, state):
        if "y" not in state:
            state["y"] = proj_simplex_rows(np.nan_to_num(x_self)) if project else x_self.copy()
        x, y = x_self, state["y"]
        k = len(X_nb)
        M = np.nan_to_num(np.asarray(X_nb, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        if center == "median":  # high-breakdown reference point for measuring distances
            c = np.median(np.concatenate([x[None], M]), 0)
        else:
            c = x
        D = M - c
        if group == "node":
            r = np.sqrt((D ** 2).reshape(k, -1).sum(1)).reshape(k, 1, 1)
        else:
            r = np.sqrt((D ** 2).sum(-1, keepdims=True))
        dl = delta * np.quantile(r, q, axis=0, keepdims=True) if adaptive else delta
        w = np.minimum(1.0, dl / np.maximum(r, 1e-12))
        if loss == "log":
            w = w * w
        elif loss == "l2":
            w = np.ones_like(w)
        if solver == "mm":
            x = (x + kappa * y + (w * M).sum(0)) / (1.0 + kappa + w.sum(0))
        else:
            x = x + ((w * (M - x)).sum(0) - kappa * (x - y)) / (k + 1.0 + kappa)
        return proj_simplex_rows(x) if project else x
    return aggregate

aggregate = make_arco(loss='log', center='self', q=0.25, delta=2.0, kappa=0.1)
