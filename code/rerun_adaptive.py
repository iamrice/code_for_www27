"""Recompute only the 'adaptive' rows of held-out result files after the attack fix."""
import json, sys, multiprocessing as mp
import numpy as np
import evaluate as E
from consensus import BASELINES
D = f"{E.ROOT}/design_runs"
PROGS = {"LLM-A": f"{D}/A/A1.4_ReputationClippedGossip.py", "LLM-H": f"{D}/H/H5.1_H5_1_PersistentBandClippedMix.py",
         "Rand200": f"{D}/R0/rand0_197.py", "Rand24": f"{D}/R0/rand0_031.py"}
for name in sys.argv[2:]:
    path = f"{E.ROOT}/results/heldout/{name}.json"
    H = json.load(open(path))
    src = None if name in BASELINES else open(PROGS[name]).read()
    for s, v in H.items():
        idx = [i for i, r in enumerate(v["rows"]) if r["scenario"][2] == "adaptive"]
        with mp.get_context("fork").Pool(int(sys.argv[1])) as pool:
            new = pool.map(E._worker, [(src, name, tuple(v["rows"][i]["scenario"]), 300) for i in idx])
        for i, r in zip(idx, new):
            v["rows"][i] = r
        accs = np.array([r["acc"] for r in v["rows"]])
        v["summary"].update(mean_acc=float(accs.mean()), worst_acc=float(accs.min()),
                            score=float(0.5 * accs.mean() + 0.5 * accs.min()),
                            n_fail=int(sum(not r["ok"] for r in v["rows"])))
        print(name, s, "adaptive", np.mean([r["acc"] for r in new]).round(4), "fails", sum(not r["ok"] for r in new), flush=True)
    json.dump(H, open(path, "w"))
