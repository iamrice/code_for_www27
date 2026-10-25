"""Aggregate held-out results into the numbers used in the paper."""
import glob, json, os
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results"
SEEN = {"none", "gaussian", "signflip", "labelflip", "alie"}

def load():
    out = {}
    for f in glob.glob(f"{R}/heldout/*.json"):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out

def rows(res, suite):
    return res[suite]["rows"]

def stats(rs):
    a = np.array([r["acc"] for r in rs]); o = np.array([r["oracle_acc"] for r in rs])
    rel = a / o
    return dict(acc=a.mean(), rel=rel.mean(), worst_rel=rel.min(), p10_rel=np.quantile(rel, 0.1), n=len(a))

if __name__ == "__main__":
    H = load()
    order = sorted(H, key=lambda k: -np.mean([stats(rows(H[k], s))["rel"] for s in H[k]]))
    print(f"{'method':18s}" + "".join(f"{s[8:]:>22s}" for s in ["heldout_attack", "heldout_data", "heldout_topo", "heldout_frac"]))
    for k in order:
        line = f"{k:18s}"
        for s in ["heldout_attack", "heldout_data", "heldout_topo", "heldout_frac"]:
            if s in H[k]:
                st = stats(rows(H[k], s)); line += f"   {st['rel']:.3f}/{st['worst_rel']:.3f}/{st['p10_rel']:.3f}"
        print(line)
    print("\nheldout_data per attack (rel. to oracle), unseen attacks marked *")
    atks = ["none", "gaussian", "signflip", "labelflip", "alie", "ipm", "mimic", "zerosum", "dissensus", "targeted", "adaptive"]
    print(f"{'':18s}" + "".join(f"{a[:7]+('*' if a not in SEEN else ''):>9s}" for a in atks))
    for k in order:
        rs = rows(H[k], "heldout_data")
        print(f"{k:18s}" + "".join(f"{np.mean([r['acc']/r['oracle_acc'] for r in rs if r['scenario'][2]==a]):9.3f}" for a in atks))
