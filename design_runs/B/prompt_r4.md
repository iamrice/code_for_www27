You are designing a Byzantine-robust aggregation rule for *fully decentralized* federated learning.

## Setting
- 20 nodes on a sparse peer-to-peer graph, no server. Each node trains a local model on private non-IID data.
- Nodes share *knowledge*, not parameters: their softmax predictions on a shared unlabeled public set,
  a matrix of shape (m, C) = (1000, 10); each row is a probability vector.
- Knowledge consensus runs T = 10 synchronous gossip iterations. At iteration t every honest node i receives
  one (m, C) message from each graph neighbour and replaces its state by the output of YOUR function.
  The final state (clipped to >= 0 and row-normalised) becomes the distillation target of node i.
- Some neighbours are Byzantine: they may send arbitrary arrays (not necessarily probabilities), may collude,
  know the honest messages, and may change behaviour over iterations. A node does not know which neighbours,
  or how many, are Byzantine; overall about 20% of nodes are, but locally it can be more.
- Some honest nodes have very noisy labels (their knowledge is poor but not malicious).
- Node degrees are between ~2 and ~10.

## Interface (pure numpy; `np` and `math` are pre-imported; no other imports)
```python
def aggregate(x_self, X_nb, t, T, state):
    # x_self: np.ndarray (m, C) — this node's current state (initially its own local knowledge)
    # X_nb:   np.ndarray (k, m, C) — messages received this iteration (order is fixed per node across t)
    # t, T:   current iteration (0-based) and total iterations
    # state:  dict, persists across the T iterations of this node (empty at t = 0); use it for memory
    # return: np.ndarray (m, C), finite values
```
Constraints: deterministic; < 50 ms per call for k <= 10; must not crash for k = 1.

## Objective
Honest nodes should end with accurate (argmax matches the true label) and mutually consistent targets.
Score = 0.5 * mean accuracy + 0.5 * worst-case accuracy over the evaluation scenarios.
The rule will later be deployed against attacks, datasets and topologies beyond the evaluation scenarios
below, so prefer principled mechanisms over tricks that only exploit the specific attacks listed.

## Evaluation scenarios you get feedback on
FashionMNIST knowledge snapshots from FL rounds {1, 4, 10}, Erdős–Rényi graph (p=0.3), 20% Byzantine,
attacks: none (Byzantines behave honestly), gaussian (N(0,1) noise), signflip (negated own knowledge),
labelflip (own knowledge with class order reversed), alie (coordinate-wise mean - 1.0*std of honest messages).
Reference points: average local accuracy of honest nodes before consensus ≈ 0.684; accuracy of the
honest-only average (an oracle that knows who is Byzantine) ≈ 0.860.


## Existing rules (hand-designed, hyper-parameters already tuned on these scenarios)
### Mean: score=0.2005 (mean=0.393, worst=0.008; none=0.818, gaussian=0.249, signflip=0.019, labelflip=0.017, alie=0.862)
```python
def aggregate(x_self, X_nb, t, T, state):
    return (x_self + X_nb.sum(0)) / (len(X_nb) + 1)
```
### CW-Median: score=0.7831 (mean=0.831, worst=0.736; none=0.840, gaussian=0.848, signflip=0.826, labelflip=0.805, alie=0.834)
```python
def aggregate(x_self, X_nb, t, T, state):
    return np.median(np.concatenate([x_self[None], X_nb]), 0)
```
### ClippedGossip: score=0.8193 (mean=0.847, worst=0.792; none=0.847, gaussian=0.853, signflip=0.833, labelflip=0.848, alie=0.853)
```python
def aggregate(x_self, X_nb, t, T, state):
    # self-centred clipping (He, Karimireddy & Jaggi 2022)
    k = len(X_nb)
    D = X_nb - x_self
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    return x_self + (scale[:, None, None] * D).sum(0) / (k + 1)
```
### BALANCE: score=0.8215 (mean=0.854, worst=0.789; none=0.827, gaussian=0.861, signflip=0.861, labelflip=0.862, alie=0.862)
```python
def aggregate(x_self, X_nb, t, T, state):
    # distance-threshold acceptance (Fang et al., CCS 2024)
    nrm = np.linalg.norm((X_nb - x_self).reshape(len(X_nb), -1), axis=1)
    thr = 1.5 * np.exp(-1.0 * t / T) * np.linalg.norm(x_self)
    acc = X_nb[nrm <= thr]
    if len(acc) == 0:
        return x_self
    return 0.5 * x_self + 0.5 * acc.mean(0)
```

## Best of your 12 previous candidates (feedback from the evaluation scenarios)

### B2.4_B2_4_SemanticAlignment (round 2): score=0.8342 (mean=0.855, worst=0.813; none=0.848, gaussian=0.857, signflip=0.857, labelflip=0.858, alie=0.856)
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

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
    ref = 0.5 * (xs + center)
    M = np.einsum('kmc,md->kcd', P, ref)               # (k, C, C) soft confusion
    rowsum = M.sum(2)
    diag = np.einsum('kcc->kc', M)
    rmax = M.max(2)
    ratio = diag / np.maximum(rmax, 1e-12)
    mass = rowsum / np.maximum(rowsum.sum(1, keepdims=True), 1e-12)
    align = (mass * ratio).sum(1)
    w_align = np.clip((align - 0.4) / 0.4, 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_new = w_align * w_dist * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum > 1e-6:
        nb = (trust[:, None, None] * P).sum(0) / tsum
        s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
        w_self = 0.5 + 0.5 * s_agree
    else:
        w_self = 1.0
    return (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)

```

### B2.1_B2_1_TrustCore (round 2): score=0.8333 (mean=0.855, worst=0.812; none=0.846, gaussian=0.857, signflip=0.857, labelflip=0.857, alie=0.857)
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

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
    n = k + 1
    S = np.sqrt(Z)
    H = np.sqrt(np.maximum(0.5 * ((S[:, None] - S[None]) ** 2).sum(-1), 0.0)).mean(-1)  # (n,n) in [0,1]
    vv = np.concatenate([[0.0], viol])
    Dm = H + 5.0 * (vv[:, None] + vv[None, :])
    np.fill_diagonal(Dm, 0.0)
    q = max(1, (n + 1) // 2 - 1)
    sc = np.sort(Dm, 1)[:, 1:q + 1].mean(1)
    sc[0] *= 0.9
    med = int(np.argmin(sc))
    d = Dm[med]
    r = max(float(np.sort(d)[min(q, n - 1)]), 0.02)
    w_core = 1.0 / (1.0 + (d / (2.0 * r)) ** 4)
    w_core[med] = 1.0
    center = (w_core[:, None, None] * Z).sum(0) / w_core.sum()
    lab = Z.argmax(-1)
    lab_c = center.argmax(-1)
    lab_s = lab[0]
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab_s[None]).mean(1))
    lo, hi = 2.5 / C, 5.0 / C
    w_agree = np.clip((agree - lo) / (hi - lo), 0.0, 1.0)
    w_new = (w_core * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    w_self = max(0.5, float(w_core[0]))
    out = (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + trust.sum())
    return out

```

### B2.2_B2_2_EvidenceLedger (round 2): score=0.8330 (mean=0.854, worst=0.812; none=0.846, gaussian=0.860, signflip=0.860, labelflip=0.851, alie=0.854)
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

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
    n = k + 1
    lab = Z.argmax(-1)
    d = np.empty(n)
    a = np.empty(n)
    for j in range(n):
        c = np.median(np.delete(Z, j, 0), 0)
        d[j] = np.abs(Z[j] - c).sum(-1).mean()
        a[j] = (lab[j] == c.argmax(-1)).mean()
    a_self = (lab == lab[0][None]).mean(1)
    a = 0.5 * (a + a_self)
    ref = max(float(np.median(d)), 0.02)
    g_dist = np.clip(2.0 - d / ref, -1.0, 1.0)
    g_agree = np.clip((a - 3.0 / C) / (2.0 / C), -1.0, 1.0)
    g = np.minimum(g_dist, g_agree)[1:] - viol / 0.05
    g = np.clip(g, -3.0, 1.0)
    L = state.get('L')
    if L is None or len(L) != k:
        L = np.zeros(k)
    L = np.clip(L + np.where(g < 0, 3.0 * g, g), -10.0, 4.0)
    state['L'] = L
    trust = 1.0 / (1.0 + np.exp(-3.0 * (L - 0.5)))
    return (xs + (trust[:, None, None] * P).sum(0)) / (1.0 + trust.sum())

```

### B3.3_R3_3_SemanticAlignmentMomentum (round 3): score=0.8236 (mean=0.851, worst=0.796; none=0.838, gaussian=0.858, signflip=0.857, labelflip=0.844, alie=0.858)
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

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
    ref = 0.5 * (xs + center)
    M = np.einsum('kmc,md->kcd', P, ref)
    rowsum = M.sum(2)
    diag = np.einsum('kcc->kc', M)
    rmax = M.max(2)
    ratio = diag / np.maximum(rmax, 1e-12)
    mass = rowsum / np.maximum(rowsum.sum(1, keepdims=True), 1e-12)
    align = (mass * ratio).sum(1)
    w_align = np.clip((align - 0.4) / 0.4, 0.0, 1.0)
    dist = np.abs(Z - center[None]).sum(-1).mean(1)
    d_ref = float(np.median(dist)) + 1e-3
    w_dist = 1.0 / (1.0 + (dist[1:] / (2.0 * d_ref)) ** 4)
    w_new = w_align * w_dist * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.7 * prev + 0.3 * w_new)
    state['trust'] = trust
    tsum = float(trust.sum())
    if tsum > 1e-6:
        nb = (trust[:, None, None] * P).sum(0) / tsum
        s_agree = float((xs.argmax(-1) == nb.argmax(-1)).mean())
        w_self = 0.5 + 0.5 * s_agree
    else:
        w_self = 1.0
    y = (w_self * xs + (trust[:, None, None] * P).sum(0)) / (w_self + tsum)
    xp = state.get('x_prev')
    if xp is not None and xp.shape == xs.shape and tsum > 1e-6:
        beta = 0.3 if t < T - 1 else 0.15
        y = y + beta * (xs - xp)
    state['x_prev'] = xs.copy()
    return proj(y)

```

### B3.2_R3_2_ClassConditionalReliability (round 3): score=0.8219 (mean=0.854, worst=0.790; none=0.851, gaussian=0.859, signflip=0.859, labelflip=0.842, alie=0.859)
```python
def aggregate(x_self, X_nb, t, T, state):
    C = x_self.shape[-1]

    def clean(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e6, 1e6)

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
    n = k + 1
    S = np.sqrt(Z)
    BC = np.einsum('imc,jmc->ijm', S, S)
    H = np.sqrt(np.clip(1.0 - BC, 0.0, 1.0)).mean(-1)
    vv = np.concatenate([[0.0], viol])
    Dm = H + 5.0 * (vv[:, None] + vv[None, :])
    np.fill_diagonal(Dm, 0.0)
    q = max(1, (n + 1) // 2 - 1)
    sc = np.sort(Dm, 1)[:, 1:q + 1].mean(1)
    sc[0] *= 0.9
    med = int(np.argmin(sc))
    d = Dm[med]
    r = max(float(np.sort(d)[min(q, n - 1)]), 0.02)
    w_core = 1.0 / (1.0 + (d / (2.0 * r)) ** 4)
    w_core[med] = 1.0
    Wz = w_core[:, None, None] * Z
    tot = Wz.sum(0)
    wsum = float(w_core.sum())
    center = tot / wsum
    den = wsum - w_core
    loo = np.where(den[:, None, None] > 1e-6,
                   (tot[None] - Wz) / np.maximum(den, 1e-6)[:, None, None],
                   center[None])
    lab = Z.argmax(-1)
    lab_l = loo.argmax(-1)
    lab_c = center.argmax(-1)
    hitm = (lab == lab_l)
    L = np.eye(C)[lab_l]                                   # (n, m, C)
    cnt = L.sum(1)                                         # (n, C)
    hit = (L * hitm[..., None]).sum(1)                     # (n, C)
    pa = hitm.mean(1)
    rc = (hit + 4.0 * pa[:, None]) / (cnt + 4.0)
    rc = np.clip(rc, 1e-3, 1.0 - 1e-3)
    lw = np.clip(np.log(rc * (C - 1) / (1.0 - rc)), 0.0, 6.0)   # (n, C)
    u = (center @ lw.T).T                                  # (n, m)
    members = w_core > 0.5
    ref = max(float(np.median(u.mean(1)[members])), 1e-6)
    u = np.clip(u / ref, 0.0, 2.0)
    agree = 0.5 * ((lab == lab_c[None]).mean(1) + (lab == lab[0][None]).mean(1))
    w_agree = np.clip((agree - 2.5 / C) / (2.5 / C), 0.0, 1.0)
    w_new = (w_core * w_agree)[1:] * np.exp(-viol / 0.05)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = w_new
    else:
        trust = np.where(w_new < prev, w_new, 0.6 * prev + 0.4 * w_new)
    state['trust'] = trust
    g = np.concatenate([[max(0.5, float(w_core[0]))], trust])
    W = g[:, None] * u
    W[0] = np.maximum(W[0], 0.25)
    return (W[:, :, None] * Z).sum(0) / W.sum(0)[:, None]

```

## Results of your most recent round

- B3.1_R3_1_ReliabilityCore: score=0.8103 (mean=0.848, worst=0.773; none=0.844, gaussian=0.853, signflip=0.853, labelflip=0.836, alie=0.853)

- B3.2_R3_2_ClassConditionalReliability: score=0.8219 (mean=0.854, worst=0.790; none=0.851, gaussian=0.859, signflip=0.859, labelflip=0.842, alie=0.859)

- B3.3_R3_3_SemanticAlignmentMomentum: score=0.8236 (mean=0.851, worst=0.796; none=0.838, gaussian=0.858, signflip=0.857, labelflip=0.844, alie=0.858)

- B3.4_R3_4_LogLinearPooling: score=0.8032 (mean=0.844, worst=0.762; none=0.846, gaussian=0.848, signflip=0.848, labelflip=0.831, alie=0.848)

Analyse what worked and what failed, then improve: refine the strongest ideas, combine complementary mechanisms, and include at least one genuinely new idea.


## What to return
Propose 4 *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
