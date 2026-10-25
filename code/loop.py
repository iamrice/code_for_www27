"""Generate-evaluate-refine loop for consensus-rule design (driver side).

python loop.py prompt --run A --round 1          # writes design_runs/A/prompt_r1.md
python loop.py ingest --run A --round 1 --resp design_runs/A/response_r1.md
python loop.py select --run A
The LLM sees only the prompt; it has no tool access to the code base or held-out scenarios.
"""
import argparse
import json
import os
import re
import numpy as np

import consensus as C
import evaluate as E

ROOT = E.ROOT
B = 4          # candidates per round
TOPK = 5       # programs shown back to the LLM

TASK = """You are designing a Byzantine-robust aggregation rule for *fully decentralized* federated learning.

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
Reference points: average local accuracy of honest nodes before consensus ≈ {local}; accuracy of the
honest-only average (an oracle that knows who is Byzantine) ≈ {oracle}.
"""

FORMAT = """
## What to return
Propose {B} *diverse* candidate rules. For each: a short heading `### Candidate <name>`, 1-3 sentences on the idea,
then ONE ```python code block defining `aggregate`. Return nothing else. Do not use any tools.
"""


def run_dir(run):
    d = f"{ROOT}/design_runs/{run}"
    os.makedirs(d, exist_ok=True)
    return d


def hist_path(run):
    return f"{run_dir(run)}/history.json"


def load_hist(run):
    p = hist_path(run)
    return json.load(open(p)) if os.path.exists(p) else []


def fmt_scores(s):
    pa = ", ".join(f"{k}={v:.3f}" for k, v in s["per_attack"].items())
    return f"score={s['score']:.4f} (mean={s['mean_acc']:.3f}, worst={s['worst_acc']:.3f}; {pa})" + \
        (f"; FAILED scenarios={s['n_fail']}" if s["n_fail"] else "")


BASELINE_SRC = {
    "Mean": """def aggregate(x_self, X_nb, t, T, state):
    return (x_self + X_nb.sum(0)) / (len(X_nb) + 1)
""",
    "CW-Median": """def aggregate(x_self, X_nb, t, T, state):
    return np.median(np.concatenate([x_self[None], X_nb]), 0)
""",
    "ClippedGossip": """def aggregate(x_self, X_nb, t, T, state):
    # self-centred clipping (He, Karimireddy & Jaggi 2022)
    k = len(X_nb)
    D = X_nb - x_self
    nrm = np.linalg.norm(D.reshape(k, -1), axis=1)
    tau = 0.5 * np.quantile(nrm, 0.25) + 1e-12
    scale = np.minimum(1.0, tau / np.maximum(nrm, 1e-12))
    return x_self + (scale[:, None, None] * D).sum(0) / (k + 1)
""",
    "BALANCE": """def aggregate(x_self, X_nb, t, T, state):
    # distance-threshold acceptance (Fang et al., CCS 2024)
    nrm = np.linalg.norm((X_nb - x_self).reshape(len(X_nb), -1), axis=1)
    thr = 1.5 * np.exp(-1.0 * t / T) * np.linalg.norm(x_self)
    acc = X_nb[nrm <= thr]
    if len(acc) == 0:
        return x_self
    return 0.5 * x_self + 0.5 * acc.mean(0)
""",
}


def baseline_block():
    res = json.load(open(f"{ROOT}/results/baselines_design.json"))
    out = ["## Existing rules (hand-designed, hyper-parameters already tuned on these scenarios)"]
    for name, src in BASELINE_SRC.items():
        out.append(f"### {name}: {fmt_scores(res[name]['summary'])}\n```python\n{src}```")
    return "\n".join(out)


def refs():
    res = json.load(open(f"{ROOT}/results/baselines_design.json"))["Mean"]["rows"]
    return np.mean([r["local_acc"] for r in res]), np.mean([r["oracle_acc"] for r in res])


def make_prompt(run, rnd):
    loc, orc = refs()
    parts = [TASK.replace("{local}", f"{loc:.3f}").replace("{oracle}", f"{orc:.3f}"), baseline_block()]
    hist = load_hist(run)
    if hist:
        ranked = sorted(hist, key=lambda h: -h["design"]["score"])[:TOPK]
        parts.append(f"## Best of your {len(hist)} previous candidates (feedback from the evaluation scenarios)")
        for h in ranked:
            parts.append(f"### {h['name']} (round {h['round']}): {fmt_scores(h['design'])}\n"
                         f"```python\n{h['src']}\n```")
        last = [h for h in hist if h["round"] == rnd - 1]
        parts.append("## Results of your most recent round")
        for h in last:
            err = f" ERROR: {h['error']}" if h.get("error") else ""
            parts.append(f"- {h['name']}: {fmt_scores(h['design'])}{err}")
        parts.append("Analyse what worked and what failed, then improve: refine the strongest ideas, "
                     "combine complementary mechanisms, and include at least one genuinely new idea.")
    parts.append(FORMAT.replace("{B}", str(B)))
    txt = "\n\n".join(parts)
    open(f"{run_dir(run)}/prompt_r{rnd}.md", "w").write(txt)
    return txt


def parse(resp):
    blocks = re.findall(r"###\s*Candidate\s*([^\n]*)\n(.*?)```python\n(.*?)```", resp, re.S)
    return [(re.sub(r"[^A-Za-z0-9_-]+", "_", n.strip())[:40] or f"c{i}", d.strip(), c) for i, (n, d, c) in enumerate(blocks)]


def ingest(run, rnd, resp_path):
    resp = open(resp_path).read()
    hist = load_hist(run)
    cands = parse(resp)
    print(f"parsed {len(cands)} candidates")
    for idx, (name, desc, src) in enumerate(cands):
        name = f"{run}{rnd}.{idx + 1}_{name}"
        p = f"{run_dir(run)}/{name}.py"
        open(p, "w").write(src)
        d = E.evaluate(src=src, suite="design", procs=6)
        v = E.evaluate(src=src, suite="design_val", procs=6)
        errs = [r["error"] for r in d["rows"] if not r["ok"]]
        hist.append(dict(name=name, round=rnd, desc=desc, src=src, file=p, design=d["summary"],
                         val=v["summary"], error=errs[0] if errs else None))
        print(name, fmt_scores(d["summary"]), "| val", round(v["summary"]["score"], 4))
    json.dump(hist, open(hist_path(run), "w"), indent=1)


def select(run):
    hist = load_hist(run)
    best = max(hist, key=lambda h: h["val"]["score"])
    print("selected", best["name"], "val", best["val"]["score"], "design", best["design"]["score"])
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run")
    ap.add_argument("--round", type=int)
    ap.add_argument("--resp")
    a = ap.parse_args()
    if a.cmd == "prompt":
        print(make_prompt(a.run, a.round))
    elif a.cmd == "ingest":
        ingest(a.run, a.round, a.resp)
    elif a.cmd == "select":
        select(a.run)
