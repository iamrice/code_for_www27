import json, itertools, sys
import consensus as C, evaluate as E
from arco import make_arco
grid = list(itertools.product(["mm", "grad"], ["huber", "log"], ["node", "row"], [0.0, 0.05, 0.2, 0.5], [0.5, 1.0, 2.0]))
res = []
for solver, loss, group, kappa, delta in grid:
    C.BASELINES["_t"] = make_arco(kappa=kappa, delta=delta, loss=loss, solver=solver, group=group)
    d = E.evaluate(name="_t", suite="design", procs=6)["summary"]
    v = E.evaluate(name="_t", suite="design_val", procs=6)["summary"]
    res.append(dict(solver=solver, loss=loss, group=group, kappa=kappa, delta=delta, design=d["score"], val=v["score"], per=d["per_attack"]))
    print(solver, loss, group, kappa, delta, round(d["score"], 4), round(v["score"], 4), flush=True)
    json.dump(res, open("../results/tune_arco.json", "w"), indent=1)
