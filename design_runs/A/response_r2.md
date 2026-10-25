### Candidate HellingerContractiveRepClip
Refines the best rule so far (A1.4). It does reputation-weighted self-centred clipping in Hellinger (sqrt-probability) space, where distances are bounded and more meaningful for probability rows. It adds a contraction prior: honest gossip should shrink disagreement over time, so the clipping radius may never grow by more than 10% from one iteration to the next. This limits adaptive attackers and low-degree nodes (k = 1) where the quantile is the attacker itself. Reputation falls quickly and recovers slowly.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    S = np.sqrt(xs)
    R = np.sqrt(Y)
    D = R - S[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    q = 0.25 if k >= 4 else 0.5
    tau = float(np.quantile(nrm, q)) + 1e-12
    prev_tau = state.get('tau')
    if prev_tau is not None:
        tau = min(tau, 1.1 * prev_tau)
    state['tau'] = tau
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = scale.copy()
    else:
        # fast to lose trust, slow to regain it
        r = np.where(scale < r, 0.4 * r + 0.6 * scale, 0.7 * r + 0.3 * scale)
    state['rep'] = r
    w = r * scale
    out = S + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate IterativeCenteredClipRep
Instead of clipping around its own state, which is noisy when the node's own labels are noisy, the node estimates a robust centre by iterative centred clipping (Karimireddy et al. 2021). The centre starts at the midpoint between the node's own state and the coordinate-wise median, and each candidate is weighted by a reputation carried across iterations. The node then takes one bounded step from its own state toward that centre, so a poor honest node is pulled toward consensus while any Byzantine influence stays limited.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    Z = np.sqrt(np.concatenate([xs[None], Y], 0))     # (n, m, C)
    n = k + 1
    F = Z.reshape(n, -1)
    r = state.get('rep')
    if r is None or len(r) != k:
        r = np.ones(k)
    prior = np.concatenate([[1.0], r])
    c = 0.5 * F[0] + 0.5 * np.median(F, 0)
    scale = np.ones(n)
    for _ in range(3):
        Dc = F - c[None]
        dn = np.linalg.norm(Dc, axis=1)
        tau = float(np.median(dn[1:])) + 1e-12
        scale = np.minimum(1.0, tau / np.maximum(dn, 1e-12))
        w = prior * scale
        c = c + (w[:, None] * Dc).sum(0) / prior.sum()
    r = 0.6 * r + 0.4 * scale[1:] if 'rep' in state else scale[1:].copy()
    state['rep'] = r
    # bounded step from self toward the robust centre
    step = c - F[0]
    sn = np.linalg.norm(step)
    dself = np.linalg.norm(F[1:] - F[0][None], axis=1)
    lim = float(np.quantile(dself, 0.5)) + 1e-12
    alpha = (0.5 + 0.5 * (t + 1) / T)
    out = F[0] + alpha * step * min(1.0, lim / max(sn, 1e-12))
    out = np.clip(out.reshape(xs.shape), 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate SoftKrumReputationGossip
New idea: a soft Krum-style density score. Each candidate (self and neighbours) is scored by its mean Hellinger distance to its nearest half of the other candidates. Candidates in the dense honest cluster score low. Outliers (gaussian, signflip, labelflip) and loners score high and get exponentially smaller weight. The weights are smoothed over iterations, and the result is applied as a clipped gossip step from the node's own state.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    n = k + 1
    F = np.sqrt(np.concatenate([xs[None], Y], 0)).reshape(n, -1)
    G = F @ F.T
    sq = np.diag(G)
    Dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))
    h = max(1, (n - 1) // 2)
    Ds = np.sort(Dm + np.diag(np.full(n, np.inf)), axis=1)[:, :h]
    score = Ds.mean(1)
    base = float(np.median(score)) + 1e-12
    dens = np.exp(-4.0 * np.maximum(score - score.min(), 0.0) / base)
    cur = dens[1:]
    r = state.get('rep')
    if r is None or len(r) != k:
        r = cur.copy()
    else:
        r = np.minimum(cur + 0.1, 0.5 * r + 0.5 * cur)
    r = np.clip(r, 0.0, 1.0)
    state['rep'] = r
    D = F[1:] - F[0][None]
    nrm = Dm[0, 1:]
    tau = float(np.quantile(nrm, 0.5)) + 1e-12
    sc = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    w = r * sc
    out = F[0] + (w[:, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out.reshape(xs.shape), 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
```

### Candidate RowwiseClipNeighbourReputation
New idea: two-level robustness. Clipping happens per sample (per row), with a radius taken from the across-neighbour median distance for that row, blended with the global median radius. A neighbour that disagrees only on some samples, such as an honest node with noisy labels, still contributes fully on the rest. A neighbour-level reputation, the EMA of its average row acceptance, scales the whole message, so attacks that perturb every row lose influence over time.
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        C = X.shape[-1]
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    xs = san(x_self)
    Y = san(X_nb)
    k = len(Y)
    S = np.sqrt(xs)
    R = np.sqrt(Y)
    D = R - S[None]                                   # (k, m, C)
    dr = np.linalg.norm(D, axis=2)                    # (k, m)
    q = 0.4 if k >= 3 else 0.5
    tau_row = np.quantile(dr, q, axis=0)              # (m,)
    tau_glob = float(np.median(dr))
    tau = 0.5 * tau_row + 0.5 * tau_glob + 1e-12
    sr = np.minimum(1.0, tau[None, :] / np.maximum(dr, 1e-12))   # (k, m)
    acc = sr.mean(1)
    r = state.get('rep')
    if r is None or len(r) != k:
        r = acc.copy()
    else:
        r = np.where(acc < r, 0.4 * r + 0.6 * acc, 0.7 * r + 0.3 * acc)
    state['rep'] = r
    rw = np.clip((r - 0.3) / 0.5, 0.0, 1.0)            # hard-gate low-reputation neighbours
    w = rw[:, None] * sr                               # (k, m)
    out = S + (w[:, :, None] * D).sum(0) / (1.0 + rw.sum())
    out = np.clip(out, 0.0, None) ** 2 + 1e-12
    return out / out.sum(-1, keepdims=True)
```
