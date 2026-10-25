"""Score every designed / sampled candidate on the heldout_lite suite."""
import glob, json, os, sys
import evaluate as E
out = f"{E.ROOT}/results/lite_all.json"
res = json.load(open(out)) if os.path.exists(out) else {}
files = sorted(glob.glob(f"{E.ROOT}/design_runs/[ABCH]/*.py")) + sorted(glob.glob(f"{E.ROOT}/design_runs/R0/*.py"))
from consensus import BASELINES
for b in BASELINES:
    if b not in res:
        res[b] = E.evaluate(name=b, suite="heldout_lite", procs=int(sys.argv[1]))["summary"]
for f in files:
    k = os.path.relpath(f, f"{E.ROOT}/design_runs")
    if k in res:
        continue
    res[k] = E.evaluate(src=open(f).read(), suite="heldout_lite", procs=int(sys.argv[1]))["summary"]
    print(k, round(res[k]["mean_acc"], 4), flush=True)
    json.dump(res, open(out, "w"))
json.dump(res, open(out, "w"))
