"""Evaluate frozen rules on all pre-registered held-out suites.

python heldout.py --baselines                      # all handcrafted rules
python heldout.py --prog NAME=path/to/prog.py ...  # frozen designed rules
"""
import argparse
import json
import os
import evaluate as E
from consensus import BASELINES

SUITES = ["heldout_attack", "heldout_data", "heldout_topo", "heldout_frac"]
OUT = f"{E.ROOT}/results/heldout"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--prog", nargs="*", default=[])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--suites", nargs="*", default=SUITES)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(b, None) for b in BASELINES] if a.baselines else []
    jobs += [tuple(p.split("=", 1)) for p in a.prog]
    for name, path in jobs:
        res = {}
        for s in a.suites:
            if path is None:
                r = E.evaluate(name=name, suite=s, procs=a.procs)
            else:
                r = E.evaluate(src=open(path).read(), suite=s, procs=a.procs)
            res[s] = r
            print(name, s, json.dumps(r["summary"]), flush=True)
        json.dump(res, open(f"{OUT}/{name}.json", "w"))
