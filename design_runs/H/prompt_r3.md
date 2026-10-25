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

### H2.2_AnnealedMedianBand (round 2): score=0.8195 (mean=0.843, worst=0.796; none=0.853, gaussian=0.861, signflip=0.831, labelflip=0.810, alie=0.860)
```python
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

```

### H2.4_GeometricMedianFromClip (round 2): score=0.8079 (mean=0.840, worst=0.775; none=0.846, gaussian=0.849, signflip=0.823, labelflip=0.835, alie=0.849)
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    xs = x_self.reshape(-1)
    D = X_nb.reshape(k, -1) - xs
    nrm = np.linalg.norm(D, axis=1)
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    Q = np.concatenate([xs[None], xs + scale[:, None] * D])
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    z = Q.mean(0)
    for _ in range(20):
        d = np.linalg.norm(Q - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * Q).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)

```

### H1.3_GeometricMedianCentre (round 1): score=0.8066 (mean=0.844, worst=0.769; none=0.845, gaussian=0.856, signflip=0.823, labelflip=0.834, alie=0.863)
```python
def aggregate(x_self, X_nb, t, T, state):
    P = np.concatenate([x_self[None], X_nb]).reshape(len(X_nb) + 1, -1)
    z = np.median(P, 0)
    for _ in range(30):
        d = np.linalg.norm(P - z, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        z_new = (w[:, None] * P).sum(0) / w.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    return z.reshape(x_self.shape)

```

### H1.1_MedoidFilteredMean (round 1): score=0.8040 (mean=0.843, worst=0.765; none=0.837, gaussian=0.848, signflip=0.851, labelflip=0.850, alie=0.828)
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    S = np.concatenate([x_self[None], X_nb])
    med = np.median(S, 0)
    d = np.linalg.norm((S - med).reshape(k + 1, -1), axis=1)
    keep = max(1, int(np.ceil(0.6 * (k + 1))))
    idx = np.argsort(d, kind='stable')[:keep]
    return S[idx].mean(0)

```

### H2.3_SimplexTVReputation (round 2): score=0.8017 (mean=0.847, worst=0.757; none=0.847, gaussian=0.856, signflip=0.855, labelflip=0.820, alie=0.857)
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    C = x_self.shape[-1]
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None).reshape(k + 1, -1, C)
    s = P.sum(-1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-12), 1.0 / C)
    ref = np.median(P, 0)
    tv = 0.5 * np.abs(P - ref).sum(-1).mean(-1)
    rep = state.get('rep')
    if rep is None or len(rep) != k:
        rep = np.ones(k)
    rep = np.asarray(rep, dtype=float)
    u = tv[1:] / (np.median(tv[1:]) + 1e-12)
    good = np.exp(-np.maximum(u - 1.5, 0.0) ** 2)
    rep = 0.7 * rep + 0.3 * good
    state['rep'] = rep
    tau = np.median(tv) + 1e-12
    w = rep * np.minimum(1.0, tau / np.maximum(tv[1:], 1e-12))
    Xn = P[1:]
    num = P[0] + (w[:, None, None] * Xn).sum(0)
    den = 1.0 + w.sum()
    out = num / den
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(x_self.shape)

```

## Results of your most recent round

- H2.1_HuberIRLSAnchored: score=0.7658 (mean=0.813, worst=0.718; none=0.844, gaussian=0.856, signflip=0.751, labelflip=0.753, alie=0.863)

- H2.2_AnnealedMedianBand: score=0.8195 (mean=0.843, worst=0.796; none=0.853, gaussian=0.861, signflip=0.831, labelflip=0.810, alie=0.860)

- H2.3_SimplexTVReputation: score=0.8017 (mean=0.847, worst=0.757; none=0.847, gaussian=0.856, signflip=0.855, labelflip=0.820, alie=0.857)

- H2.4_GeometricMedianFromClip: score=0.8079 (mean=0.840, worst=0.775; none=0.846, gaussian=0.849, signflip=0.823, labelflip=0.835, alie=0.849)

Analyse what worked and what failed, then improve: refine the strongest ideas, combine complementary mechanisms, and include at least one genuinely new idea.


## What to return
Propose 4 *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
