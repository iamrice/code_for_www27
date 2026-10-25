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

### A3.3_A3_3_TrustScaledRadiusClip (round 3): score=0.8375 (mean=0.857, worst=0.818; none=0.850, gaussian=0.859, signflip=0.858, labelflip=0.861, alie=0.858)
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
    base = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    r = state.get('rep')
    if r is None or len(r) != k:
        r = base.copy()
    else:
        r = np.where(base < r, 0.4 * r + 0.6 * base, 0.7 * r + 0.3 * base)
    state['rep'] = r
    tau_j = tau * (1.0 + 2.0 * r ** 2)
    scale = np.minimum(1.0, tau_j / np.maximum(nrm, 1e-12))
    w = r * scale
    out = xs + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)

```

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

### A3.1_A3_1_SelfAwareCenterRepClip (round 3): score=0.8358 (mean=0.856, worst=0.815; none=0.847, gaussian=0.859, signflip=0.858, labelflip=0.860, alie=0.857)
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
    c = xs
    if k >= 3:
        Z = np.concatenate([xs[None], Y], 0)
        med = np.median(Z, 0)
        ds = float(np.linalg.norm(xs - med))
        dn = np.linalg.norm((Y - med[None]).reshape(k, -1), axis=1)
        ref = float(np.median(dn)) + 1e-12
        alpha = float(np.clip(0.5 * (ds / ref - 1.0), 0.0, 0.5))
        prev = state.get('alpha')
        if prev is not None:
            alpha = 0.5 * prev + 0.5 * alpha
        state['alpha'] = alpha
        c = xs + alpha * (med - xs)
    D = Y - c[None]
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
    out = c + (w[:, None, None] * D).sum(0) / (1.0 + r.sum())
    out = np.clip(out, 0.0, None) + 1e-12
    return out / out.sum(-1, keepdims=True)

```

### A3.2_A3_2_DualEvidenceRepClip (round 3): score=0.8307 (mean=0.853, worst=0.808; none=0.846, gaussian=0.855, signflip=0.855, labelflip=0.855, alie=0.855)
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
    Z = np.concatenate([xs[None], Y], 0)
    med = np.median(Z, 0)
    ref = (0.5 * xs + 0.5 * med).argmax(-1)
    agree = (Y.argmax(-1) == ref[None]).mean(1)
    amax = float(agree.max())
    lo = 0.3
    hi = max(lo + 0.05, 0.8 * amax)
    ag = np.clip((agree - lo) / (hi - lo), 0.0, 1.0)
    ev = np.sqrt(scale) * ag
    r = state.get('rep')
    if r is None or len(r) != k:
        r = ev.copy()
    else:
        r = np.where(ev < r, 0.4 * r + 0.6 * ev, 0.7 * r + 0.3 * ev)
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

## Results of your most recent round

- A3.1_A3_1_SelfAwareCenterRepClip: score=0.8358 (mean=0.856, worst=0.815; none=0.847, gaussian=0.859, signflip=0.858, labelflip=0.860, alie=0.857)

- A3.2_A3_2_DualEvidenceRepClip: score=0.8307 (mean=0.853, worst=0.808; none=0.846, gaussian=0.855, signflip=0.855, labelflip=0.855, alie=0.855)

- A3.3_A3_3_TrustScaledRadiusClip: score=0.8375 (mean=0.857, worst=0.818; none=0.850, gaussian=0.859, signflip=0.858, labelflip=0.861, alie=0.858)

- A3.4_A3_4_DawidSkeneReliabilityGossip: score=0.7856 (mean=0.841, worst=0.731; none=0.837, gaussian=0.827, signflip=0.858, labelflip=0.824, alie=0.858)

Analyse what worked and what failed, then improve: refine the strongest ideas, combine complementary mechanisms, and include at least one genuinely new idea.


## What to return
Propose 4 *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
