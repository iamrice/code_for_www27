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

### C1.4_RowMADSuspicion (round 1): score=0.8264 (mean=0.848, worst=0.805; none=0.842, gaussian=0.859, signflip=0.848, labelflip=0.840, alie=0.850)
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    med = np.median(np.concatenate([x[None], X], 0), 0)
    r = np.abs(X - med[None]).sum(2)                 # (k, m)
    r0 = np.abs(x - med).sum(1)                      # (m,)
    sc = np.median(np.concatenate([r0[None], r], 0), 0) + 0.05
    rowacc = (r <= 2.0 * sc[None]).astype(float)     # (k, m)
    frac = 1.0 - rowacc.mean(1)                      # rejection rate (k,)
    susp = state.get('susp')
    if susp is None or susp.shape != frac.shape:
        susp = frac.copy()
    else:
        susp = np.maximum(frac, 0.8 * susp + 0.2 * frac)
    state['susp'] = susp
    trust = np.clip((0.6 - susp) / 0.4, 0.0, 1.0)
    W = trust[:, None] * rowacc                      # (k, m)
    out = (x + (W[..., None] * X).sum(0)) / (1.0 + W.sum(0))[..., None]
    return san(out)

```

### C1.2_ArgmaxReputation (round 1): score=0.8162 (mean=0.848, worst=0.784; none=0.822, gaussian=0.861, signflip=0.859, labelflip=0.843, alie=0.857)
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    med = np.median(np.concatenate([x[None], X], 0), 0)
    lab = med.argmax(1)
    agree = (X.argmax(2) == lab[None]).mean(1)       # (k,)
    a_self = float((x.argmax(1) == lab).mean())
    rep = state.get('rep')
    if rep is None or rep.shape != agree.shape:
        rep = agree.copy()
    else:
        rep = 0.6 * rep + 0.4 * agree
    state['rep'] = rep
    eff = np.minimum(rep, agree)
    ref = max(a_self, float(agree.max())) + 1e-12
    w = np.clip((eff - 0.5 * ref) / (0.4 * ref), 0.0, 1.0)
    out = (x + (w[:, None, None] * X).sum(0)) / (1.0 + w.sum())
    return san(out)

```

### C1.3_HellingerGM (round 1): score=0.8028 (mean=0.832, worst=0.773; none=0.838, gaussian=0.839, signflip=0.822, labelflip=0.821, alie=0.841)
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    S = np.sqrt(np.concatenate([x[None], X], 0))     # (n, m, C)
    wts = np.ones(k + 1)
    wts[0] = 1.0 + 0.25 * k
    y = np.median(S, 0)
    for _ in range(10):
        d = np.linalg.norm(S - y[None], axis=2)      # (n, m)
        inv = wts[:, None] / np.maximum(d, 1e-6)
        y = (inv[..., None] * S).sum(0) / inv.sum(0)[..., None]
    return san(y ** 2)

```

### C1.1_SanRowClip (round 1): score=0.7754 (mean=0.833, worst=0.718; none=0.843, gaussian=0.854, signflip=0.856, labelflip=0.752, alie=0.857)
```python
def aggregate(x_self, X_nb, t, T, state):
    def san(A):
        A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, 0.0, 1e6)
        s = A.sum(-1, keepdims=True)
        C = A.shape[-1]
        return np.where(s > 1e-12, A / np.maximum(s, 1e-12), 1.0 / C)

    x = san(x_self)
    X = san(X_nb)
    k = X.shape[0]
    D = X - x[None]                                  # (k, m, C)
    r = np.linalg.norm(D, axis=2)                    # (k, m) per-row distances
    tau = np.quantile(r, 0.4, axis=0) + 1e-8         # (m,) per-row radius
    scale = np.minimum(1.0, tau[None] / np.maximum(r, 1e-12))
    out = x + (scale[..., None] * D).sum(0) / (k + 1)
    return san(out)

```

## Results of your most recent round

- C1.1_SanRowClip: score=0.7754 (mean=0.833, worst=0.718; none=0.843, gaussian=0.854, signflip=0.856, labelflip=0.752, alie=0.857)

- C1.2_ArgmaxReputation: score=0.8162 (mean=0.848, worst=0.784; none=0.822, gaussian=0.861, signflip=0.859, labelflip=0.843, alie=0.857)

- C1.3_HellingerGM: score=0.8028 (mean=0.832, worst=0.773; none=0.838, gaussian=0.839, signflip=0.822, labelflip=0.821, alie=0.841)

- C1.4_RowMADSuspicion: score=0.8264 (mean=0.848, worst=0.805; none=0.842, gaussian=0.859, signflip=0.848, labelflip=0.840, alie=0.850)

Analyse what worked and what failed, then improve: refine the strongest ideas, combine complementary mechanisms, and include at least one genuinely new idea.


## What to return
Propose 4 *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
