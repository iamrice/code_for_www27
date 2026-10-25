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

## Best of your 4 previous candidates (feedback from the evaluation scenarios)

### B1.2_ReputationGossip (round 1): score=0.8208 (mean=0.852, worst=0.789; none=0.847, gaussian=0.858, signflip=0.858, labelflip=0.841, alie=0.857)
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

### B1.4_ConsistentCore (round 1): score=0.8187 (mean=0.852, worst=0.786; none=0.847, gaussian=0.857, signflip=0.857, labelflip=0.840, alie=0.857)
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

### B1.1_SimplexGeoMedian (round 1): score=0.8109 (mean=0.847, worst=0.775; none=0.842, gaussian=0.856, signflip=0.856, labelflip=0.823, alie=0.856)
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

### B1.3_AnchoredCenteredClipping (round 1): score=0.7937 (mean=0.844, worst=0.743; none=0.846, gaussian=0.860, signflip=0.845, labelflip=0.808, alie=0.860)
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

## Results of your most recent round

- B1.1_SimplexGeoMedian: score=0.8109 (mean=0.847, worst=0.775; none=0.842, gaussian=0.856, signflip=0.856, labelflip=0.823, alie=0.856)

- B1.2_ReputationGossip: score=0.8208 (mean=0.852, worst=0.789; none=0.847, gaussian=0.858, signflip=0.858, labelflip=0.841, alie=0.857)

- B1.3_AnchoredCenteredClipping: score=0.7937 (mean=0.844, worst=0.743; none=0.846, gaussian=0.860, signflip=0.845, labelflip=0.808, alie=0.860)

- B1.4_ConsistentCore: score=0.8187 (mean=0.852, worst=0.786; none=0.847, gaussian=0.857, signflip=0.857, labelflip=0.840, alie=0.857)

Analyse what worked and what failed, then improve: refine the strongest ideas, combine complementary mechanisms, and include at least one genuinely new idea.


## What to return
Propose 4 *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
