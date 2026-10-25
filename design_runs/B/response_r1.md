### Candidate SimplexGeoMedian
Every message is first checked against the protocol: it should be a row-stochastic matrix. Each message is projected onto the simplex, and a neighbour is down-weighted by how far its raw message was from that projection (the "violation"). The new state is then the per-sample weighted geometric median (Weiszfeld) of the self state and the projected messages. A geometric median has a 50% breakdown point per sample and does not average every coordinate the way a mean does.

```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)              # (k,) distance from the simplex
    w0 = np.concatenate([[1.0], np.exp(-viol / 0.05)])  # protocol-violation down-weighting
    Z = np.concatenate([xs[None], P], 0)              # (n, m, C)
    y = np.median(Z, 0)
    for _ in range(10):
        d = np.sqrt(((Z - y[None]) ** 2).sum(-1)) + 1e-6   # (n, m)
        w = w0[:, None] / d
        y = (w[:, :, None] * Z).sum(0) / w.sum(0)[:, None]
    return y
```

### Candidate ReputationGossip
Each neighbour gets a trust score from evidence about its prediction behaviour. The evidence is how often its argmax agrees with a robust per-sample centre (the coordinate-wise median including self) and with the node's own state, how far it sits from that centre compared with typical peers, and whether it violates the simplex. Trust is kept in memory with an asymmetric update: it drops at once on bad evidence and rises slowly on good evidence, so a neighbour that switches to attacking mid-run is cut off quickly. The new state is a trust-weighted mean of the projected messages, anchored on self.

```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.abs(R - P).sum(-1).mean(1)
    Z = np.concatenate([xs[None], P], 0)
    center = np.median(Z, 0)
    lab_c = center.argmax(-1)
    lab_s = xs.argmax(-1)
    lab_n = P.argmax(-1)
    agree = 0.5 * ((lab_n == lab_c[None]).mean(1) + (lab_n == lab_s[None]).mean(1))
    lo, hi = 2.5 / C, 5.0 / C                       # chance-relative agreement thresholds
    w_agree = np.clip((agree - lo) / (hi - lo), 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)   # mean row L1 to centre, (n,)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_valid = np.exp(-viol / 0.05)
    w_new = w_agree * w_dist * w_valid
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    num = xs + (trust[:, None, None] * P).sum(0)
    return num / (1.0 + trust.sum())
```

### Candidate AnchoredCenteredClipping
This is centred clipping (Karimireddy et al.), but the clipping is centred on a robust anchor, the coordinate-wise median of self plus the projected messages, rather than on the node's own state. A noisy honest node therefore does not reject good neighbours. The radius adapts to the median deviation, and it is not allowed to grow faster than 1.5x per iteration, because honest disagreement should shrink as gossip proceeds. Raw, unprojected messages are clipped, so the size of an outlier is used as evidence while its influence stays capped at tau per message.

```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    xs = proj(clean(x_self))
    Z = np.concatenate([xs[None], R], 0)
    n = Z.shape[0]
    v = np.median(np.concatenate([xs[None], proj(R)], 0), 0)
    prev_tau = state.get('tau')
    tau_used = None
    for _ in range(3):
        D = Z - v[None]
        nrm = np.linalg.norm(D.reshape(n, -1), axis=1)
        tau = float(np.median(nrm)) + 1e-12
        if prev_tau is not None:
            tau = min(tau, 1.5 * prev_tau)
        if tau_used is None:
            tau_used = tau
        sc = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
        v = v + (sc[:, None, None] * D).mean(0)
    state['tau'] = tau_used
    return proj(v)
```

### Candidate ConsistentCore
This is a Krum- or medoid-style selection by mutual consistency. The node computes pairwise mean-row-L1 distances between the self state and the projected messages, adding a penalty for simplex violations. It then finds the medoid, the message with the smallest summed distance to its ceil(n/2)-1 nearest others, with a slight tie-break in favour of self. The new state averages the messages close to that medoid: within twice the medoid's q-th nearest-neighbour distance, capped by an absolute bound on the simplex, which also protects nodes of degree 1.

```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def proj(X):
        X = np.clip(X, 0.0, None)
        s = X.sum(-1, keepdims=True)
        return np.where(s > 1e-12, X / np.maximum(s, 1e-12), 1.0 / C)

    R = clean(X_nb)
    k = R.shape[0]
    P = proj(R)
    xs = proj(clean(x_self))
    viol = np.concatenate([[0.0], np.abs(R - P).sum(-1).mean(1)])
    Z = np.concatenate([xs[None], P], 0)
    n = k + 1
    Dm = np.abs(Z[:, None] - Z[None]).sum(-1).mean(-1)   # (n, n) mean row L1 in [0, 2]
    Dm = Dm + 10.0 * (viol[:, None] + viol[None, :])
    np.fill_diagonal(Dm, 0.0)
    q = max(1, (n + 1) // 2 - 1)
    score = np.sort(Dm, 1)[:, 1:q + 1].sum(1)
    score[0] *= 0.9
    med = int(np.argmin(score))
    d = Dm[med]
    radius = min(1.4, max(2.0 * float(np.sort(d)[q]), 0.05))
    members = d <= radius
    members[med] = True
    return Z[members].mean(0)
```
