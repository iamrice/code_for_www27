"""End-to-end CIFAR-10 DFL table (final honest test accuracy, mean over available seeds)."""
import json, glob, os, collections
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results/e2e"
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/paper"
acc = collections.defaultdict(list); last = collections.defaultdict(list)
for f in glob.glob(f"{R}/*.json"):
    n = os.path.basename(f)[:-5]
    parts = n.split("_")
    seed = 1 if parts[-1] == "s1" else 0
    if seed: parts = parts[:-1]
    m, a = "_".join(parts[:-1]), parts[-1]
    d = json.load(open(f))
    acc[(m, a)].append(d["final"]["acc_honest"])
    last[(m, a)].append(np.mean([r["acc_honest"] for r in d["rounds"][-3:]]))
ATK = ["none", "signflip", "alie", "ipm", "dissensus", "targeted", "adaptive", "drift"]
SEEN = {"none", "signflip", "alie"}
METH = [("Local", "Local training only"), ("DFedAvg", "DFedAvg (param.\\ gossip)"), ("Mean", "Gossip averaging"), ("CW-Median", "CW-Median"),
        ("BRIDGE-T", "BRIDGE-T"), ("GeoMed", "GeoMed"), ("ClippedGossip", "ClippedGossip"), ("BALANCE", "BALANCE"),
        ("Rand200", "Random search (200)"), ("LLM-H", "LLM run H"), ("LLM-A", "LLM run A"), ("ARCO-G", "ARCO-G"), ("ARCO-B", "ARCO-B"), ("ARCO-H", "ARCO-H")]
def cell(m, a):
    if m == "Local":
        v = acc.get(("Local", "none"), [])
    else:
        v = acc.get((m, a), [])
    if not v: return None, ""
    s = f"{100*np.mean(v):.1f}" + ("" if len(v) == 1 else r"$^{\ast}$")
    return np.mean(v), s
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{End-to-end decentralized training on CIFAR-10 (20 nodes, 20 rounds, 20\% Byzantine, 4 noisy honest nodes): honest-node test accuracy (\%) after the last round. $^\ast$: mean of two seeds (29 cells; median seed-to-seed difference 0.5 points, maximum 4.1), otherwise one seed. $^\dagger$: attack not seen during design. Local-only training does not depend on the attack. DFedAvg exchanges parameters ($5.4\times$ the communication) and cannot express knowledge-space attacks (--).}",
     r"\label{tab:e2e}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l" + "c" * len(ATK) + "c}", r"\toprule",
     "Method & " + " & ".join(a + ("" if a in SEEN else r"$^\dagger$") for a in ATK) + r" & mean (attacked) \\", r"\midrule"]
best = {}
for a in ATK:
    vals = [cell(m, a)[0] for m, _ in METH if m not in ("Local", "DFedAvg")]
    vals = [v for v in vals if v is not None]
    best[a] = max(vals) if vals else None
for m, disp in METH:
    row, vs = [], []
    for a in ATK:
        v, s = cell(m, a)
        if v is None:
            row.append("--")
        else:
            row.append(rf"\textbf{{{s}}}" if best[a] is not None and abs(v - best[a]) < 1e-4 and m not in ("Local", "DFedAvg") else s)
            if a != "none": vs.append(v)
    complete = len(vs) == len(ATK) - 1
    row.append(f"{100*np.mean(vs):.1f}" if complete else "--")
    L.append(disp + " & " + " & ".join(row) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{P}/tab_e2e.tex", "w").write("\n".join(L))
for m, _ in METH:
    print(f"{m:14s}", " ".join(f"{cell(m,a)[1]:>7s}" for a in ATK))
