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

## Best of your 8 previous candidates (feedback from the evaluation scenarios)

### A1.4_ReputationClippedGossip (round 1): score=0.8364 (mean=0.856, worst=0.816; none=0.848, gaussian=0.859, signflip=0.858, labelflip=0.860, alie=0.857)
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
    D = Y - xs[None]
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    tau = float(np.quantile(nrm, 0.25)) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = scale.copy()
    else:
        r = 0.6 * r + 0.4 * scale
    state['rep'] = r
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)

```

### A2.1_HellingerContractiveRepClip (round 2): score=0.8270 (mean=0.848, worst=0.806; none=0.841, gaussian=0.852, signflip=0.849, labelflip=0.852, alie=0.847)
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

### A2.4_RowwiseClipNeighbourReputation (round 2): score=0.8188 (mean=0.847, worst=0.790; none=0.850, gaussian=0.853, signflip=0.850, labelflip=0.839, alie=0.845)
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

### A1.2_AgreementTrust (round 1): score=0.8159 (mean=0.849, worst=0.782; none=0.821, gaussian=0.856, signflip=0.856, labelflip=0.856, alie=0.857)
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
    C = xs.shape[-1]
    Z = np.concatenate([xs[None], Y], 0)
    med = np.median(Z, 0)
    ref = (0.5 * xs + 0.5 * med).argmax(-1)
    agree = (Y.argmax(-1) == ref[None]).mean(1)
    prev = state.get('trust')
    if prev is None or len(prev) != k:
        trust = agree
    else:
        trust = np.minimum(agree, 0.5 * prev + 0.5 * agree)
    state['trust'] = trust
    amax = float(trust.max())
    lo = max(2.0 / C, 0.5 * amax)
    hi = max(lo + 0.05, 0.85 * amax)
    w = np.clip((trust - lo) / (hi - lo), 0.0, 1.0)
    out = (xs + (w[:, None, None] * Y).sum(0)) / (1.0 + w.sum())
    return out

```

### A1.1_HellingerGeoMedian (round 1): score=0.8140 (mean=0.847, worst=0.781; none=0.843, gaussian=0.854, signflip=0.850, labelflip=0.836, alie=0.853)
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
    P = np.sqrt(np.concatenate([xs[None], Y], 0))
    n = P.shape[0]
    F = P.reshape(n, -1)
    z = np.median(F, 0)
    for _ in range(30):
        d = np.linalg.norm(F - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * F).sum(0) / w.sum()
        if np.linalg.norm(z_new - z) < 1e-7:
            z = z_new
            break
        z = z_new
    ps = F[0]
    D = z - ps
    nd = np.linalg.norm(D)
    dn = np.linalg.norm(F[1:] - ps, axis=1)
    tau = float(np.median(dn)) + 1e-12
    step = D * min(1.0, tau / max(nd, 1e-12))
    out = (ps + step).reshape(xs.shape) ** 2
    out = out + 1e-12
    return out / out.sum(-1, keepdims=True)

```

## Results of your most recent round

- A2.1_HellingerContractiveRepClip: score=0.8270 (mean=0.848, worst=0.806; none=0.841, gaussian=0.852, signflip=0.849, labelflip=0.852, alie=0.847)

- A2.2_IterativeCenteredClipRep: score=0.7915 (mean=0.839, worst=0.744; none=0.846, gaussian=0.849, signflip=0.852, labelflip=0.795, alie=0.852)

- A2.3_SoftKrumReputationGossip: score=0.7873 (mean=0.828, worst=0.747; none=0.820, gaussian=0.835, signflip=0.839, labelflip=0.802, alie=0.842)

- A2.4_RowwiseClipNeighbourReputation: score=0.8188 (mean=0.847, worst=0.790; none=0.850, gaussian=0.853, signflip=0.850, labelflip=0.839, alie=0.845)

Analyse what worked and what failed, then improve: refine the strongest ideas, combine complementary mechanisms, and include at least one genuinely new idea.


## What to return
Propose 4 *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
