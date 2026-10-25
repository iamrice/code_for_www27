import json, multiprocessing as mp, numpy as np
import evaluate as E
from programs import load_program
D = f"{E.ROOT}/design_runs"
ABL = {"Abl-noSan": "ablation/A14_noSan.py", "Abl-noRep": "ablation/A14_noRep.py", "Abl-noSan-noRep": "ablation/A14_noSan_noRep.py",
       "Abl-M-linearW": "ablation/M_linearW.py", "Abl-M-normK1": "ablation/M_normK1.py", "Abl-M-tauHalf": "ablation/M_tauHalf.py",
       "Abl-M-noProj": "ablation/M_noProj.py", "Abl-CG+San": "ablation/CG_plusSan.py",
       "LLM-A-bydesign": "A/A4.4_A4_4_RelativeHedgeTrustClip.py", "LLM-H-bydesign": "H/H4.3_PersistentSuspicionBand.py"}
def job(a):
    n, ds, r, T, s = a
    E.T_CONS = T
    o = E.run_scenario(load_program(f"{D}/{ABL[n]}"), (ds, r, "drift", "er", 0.2, "random", s))
    return dict(method=n, ds=ds, round=r, T=T, seed=s, rel=o["acc"] / o["oracle_acc"])
if __name__ == "__main__":
    jobs = [(n, ds, r, T, s) for n in ABL for ds in ("cifar10", "fmnist_a01") for r in (1, 4, 10) for T in (10, 40) for s in (1, 2)]
    with mp.get_context("fork").Pool(5) as p:
        out = p.map(job, jobs)
    json.dump(out, open(f"{E.ROOT}/results/abl_drift.json", "w"))
    for n in ABL: print(n, round(np.mean([o["rel"] for o in out if o["method"] == n and o["T"] == 40]), 3))
