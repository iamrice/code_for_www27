"""Held-out evaluation of every frozen rule on Web-text knowledge (20 Newsgroups); added for the WWW version."""
import json
import numpy as np
import evaluate as E
from consensus import BASELINES
from heldout_drift import PROGS

if __name__ == "__main__":
    out = {}
    for name in list(BASELINES) + list(PROGS):
        r = E.evaluate(name=name, suite="heldout_web", procs=4) if name in BASELINES else \
            E.evaluate(src=open(PROGS[name]).read(), suite="heldout_web", procs=4)
        for row in r["rows"]:
            row["rel"] = row["acc"] / row["oracle_acc"] if row.get("oracle_acc") else 0.0
        out[name] = r
        rel = np.array([x["rel"] for x in r["rows"]])
        atk = [x["scenario"][2] for x in r["rows"]]
        g = lambda k: np.mean([v for v, a in zip(rel, atk) if a == k])
        print(f"{name:14s} mean={rel.mean():.3f} min={rel.min():.3f} none={g('none'):.3f} drift={g('drift'):.3f} fail={r['summary']['n_fail']}", flush=True)
    json.dump(out, open(f"{E.ROOT}/results/heldout_web.json", "w"))
