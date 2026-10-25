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

## Best of your 20 previous candidates (feedback from the evaluation scenarios)

### H4.3_PersistentSuspicionBand (round 4): score=0.8442 (mean=0.857, worst=0.831; none=0.857, gaussian=0.856, signflip=0.859, labelflip=0.859, alie=0.855)
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    r = d / (np.median(d) + 1e-12)
    s_prev = state.get('s')
    if s_prev is None or len(s_prev) != k:
        s_prev = np.zeros(k)
    s = 0.6 * np.asarray(s_prev, dtype=float) + 0.4 * r[1:]
    state['s'] = s
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    trust = band * np.exp(-np.maximum(s - 1.0, 0.0))
    w = np.concatenate([[1.0], trust])
    z = (w[:, None] * P).sum(0) / w.sum()
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)

```

### H5.1_H5_1_PersistentBandClippedMix (round 5): score=0.8409 (mean=0.854, worst=0.828; none=0.855, gaussian=0.853, signflip=0.856, labelflip=0.857, alie=0.848)
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    r = d / (np.median(d) + 1e-12)
    s_prev = state.get('s')
    if s_prev is None or len(s_prev) != k:
        s_prev = np.zeros(k)
    s = 0.6 * np.asarray(s_prev, dtype=float) + 0.4 * r[1:]
    state['s'] = s
    thr = 1.0 + math.exp(-2.0 * t / T)
    band = np.clip((thr - r[1:]) / (0.5 * thr + 1e-12), 0.0, 1.0)
    trust = band * np.exp(-np.maximum(s - 1.0, 0.0))
    x0 = P[0]
    D = P[1:] - x0
    nrm = np.linalg.norm(D, axis=1)
    tau = np.median(nrm) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    z = x0 + (trust * scale)[:, None].__mul__(D).sum(0) / (1.0 + trust.sum())
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)

```

### H4.1_HellingerBandMixture (round 4): score=0.8335 (mean=0.849, worst=0.818; none=0.851, gaussian=0.852, signflip=0.845, labelflip=0.850, alie=0.848)
```python
def aggregate(x_self, X_nb, t, T, state):
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / P.shape[-1])
    S = np.sqrt(P)
    ref = np.median(S, 0)
    d = np.sqrt(((S - ref) ** 2).sum(-1)).mean(-1)
    scale = np.median(d) + 1e-12
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr - d) / (0.5 * thr + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    out = (w[:, None, None] * P).sum(0) / w.sum()
    return out.reshape(x_self.shape)

```

### H3.1_AnnealedSoftBandGeoMedian (round 3): score=0.8292 (mean=0.846, worst=0.812; none=0.840, gaussian=0.851, signflip=0.849, labelflip=0.847, alie=0.842)
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).reshape(k + 1, -1).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.median(P, 0)
    d = np.linalg.norm(P - ref, axis=1)
    scale = np.median(d) + 1e-12
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr - d) / (0.5 * thr + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    z = (w[:, None] * P).sum(0) / w.sum()
    for _ in range(20):
        dz = np.linalg.norm(P - z, axis=1)
        ww = w / np.maximum(dz, 1e-8)
        z_new = (ww[:, None] * P).sum(0) / ww.sum()
        done = np.linalg.norm(z_new - z) <= 1e-9 * (1.0 + np.linalg.norm(z))
        z = z_new
        if done:
            break
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(x_self.shape)

```

### H5.3_H5_3_PerSampleRowTrust (round 5): score=0.8208 (mean=0.846, worst=0.796; none=0.841, gaussian=0.857, signflip=0.835, labelflip=0.846, alie=0.849)
```python
def aggregate(x_self, X_nb, t, T, state):
    k = len(X_nb)
    if k == 0:
        return x_self
    P = np.concatenate([x_self[None], X_nb]).astype(float)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, None)
    rs = P.sum(-1, keepdims=True)
    P = np.where(rs > 1e-12, P / np.maximum(rs, 1e-12), 1.0 / P.shape[-1])
    ref = np.median(P, 0)
    d = np.abs(P - ref).sum(-1)
    scale = np.median(d, 0) + 1e-9
    thr = scale * (1.0 + math.exp(-2.0 * t / T))
    w = np.clip((thr[None, :] - d) / (0.5 * thr[None, :] + 1e-12), 0.0, 1.0)
    w[0] = 1.0
    out = (w[..., None] * P).sum(0) / w.sum(0)[:, None]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, 0.0, None)
    rs = out.sum(-1, keepdims=True)
    out = np.where(rs > 1e-12, out / np.maximum(rs, 1e-12), 1.0 / out.shape[-1])
    return out.reshape(x_self.shape)

```

## Results of your most recent round

- H5.1_H5_1_PersistentBandClippedMix: score=0.8409 (mean=0.854, worst=0.828; none=0.855, gaussian=0.853, signflip=0.856, labelflip=0.857, alie=0.848)

- H5.2_H5_2_WelschIRLSAnnealed: score=0.7928 (mean=0.836, worst=0.750; none=0.852, gaussian=0.860, signflip=0.815, labelflip=0.789, alie=0.863)

- H5.3_H5_3_PerSampleRowTrust: score=0.8208 (mean=0.846, worst=0.796; none=0.841, gaussian=0.857, signflip=0.835, labelflip=0.846, alie=0.849)

- H5.4_H5_4_LargestGapCutWithBanMemory: score=0.8067 (mean=0.851, worst=0.763; none=0.855, gaussian=0.861, signflip=0.861, labelflip=0.860, alie=0.817)

Analyse what worked and what failed, then improve: refine the strongest ideas, combine complementary mechanisms, and include at least one genuinely new idea.


## What to return
Propose 4 *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
