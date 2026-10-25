import os
import json, numpy as np
from scipy.stats import spearmanr
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results"
DR = R + "/../design_runs"
P = R + "/../paper"
drift = json.load(open(f"{R}/heldout_drift.json")) + json.load(open(f"{R}/abl_drift.json"))
def held(n):
    H = json.load(open(f"{R}/heldout/{n}.json"))
    g = np.array([r["acc"] / r["oracle_acc"] for s in H.values() for r in s["rows"] if r["scenario"][2] != "none"])
    nn = np.mean([r["acc"] / r["oracle_acc"] for s in H.values() for r in s["rows"] if r["scenario"][2] == "none"])
    d = np.mean([o["rel"] for o in drift if o["method"] == n and o["T"] == 40])
    return g.mean(), np.quantile(g, 0.05), g.min(), nn, d
L = json.load(open(f"{R}/lite_all.json"))
# ---- design-run table
rows = []
for run, model in [("A", "Opus"), ("B", "Opus"), ("C", "Opus"), ("H", "Haiku")]:
    h = json.load(open(f"{DR}/{run}/history.json"))
    sel = max(h, key=lambda x: x["val"]["score"]); byd = max(h, key=lambda x: x["design"]["score"])
    lite = [L[x["file"].split("design_runs/")[1]]["mean_acc"] for x in h]
    rho = spearmanr([x["design"]["score"] for x in h], lite)[0]
    nfail = sum(1 for x in h if x["design"]["n_fail"] > 0)
    rows.append((run, model, sel["name"].split("_")[0], sel["round"], sel["design"]["score"], sel["val"]["score"], *held(f"LLM-{run}"), rho, nfail, byd["round"]))
R0 = json.load(open(f"{DR}/R0/history.json"))
rho_r = spearmanr([x["design"]["score"] for x in R0], [L[x["file"].split("design_runs/")[1]]["mean_acc"] for x in R0])[0]
T = [r"\begin{table}[t]", r"\centering", r"\caption{LLM-guided design runs (6 rounds $\times$ 4 programs each). Sel.: program chosen by the pre-registered rule (best validation score) and the round it was proposed in. Held-out columns as in Table~\ref{tab:heldout} (gross = all non-drift held-out scenarios). $\rho$: Spearman correlation between design score and held-out-lite accuracy over the run's 24 programs (random search: %.2f over 200).}" % rho_r,
     r"\label{tab:llmruns}", r"\setlength{\tabcolsep}{1.6pt}", r"\scriptsize", r"\resizebox{\columnwidth}{!}{\begin{tabular}{llcccccccc}", r"\toprule",
     r"Run & LLM & sel.\ (round) & design & val. & gross mean & gross min & benign & drift & $\rho$ \\", r"\midrule"]
for r in rows:
    T.append(f"{r[0]} & {r[1]} & {r[2]} ({r[3]}) & {r[4]:.3f} & {r[5]:.3f} & {r[6]:.3f} & {r[8]:.3f} & {r[9]:.3f} & {r[10]:.3f} & {r[11]:.2f}" + r" \\")
T += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(f"{P}/tab_llmruns.tex", "w").write("\n".join(T))
for r in rows: print(r)
# per-round held-out-lite mean
for run in "ABCH":
    h = json.load(open(f"{DR}/{run}/history.json"))
    print(run, [round(np.mean([L[x["file"].split("design_runs/")[1]]["mean_acc"] for x in h if x["round"] == k]), 3) for k in range(1, 7)],
          "design", [round(np.mean([x["design"]["score"] for x in h if x["round"] == k]), 3) for k in range(1, 7)])
# ---- ablation table
AB = [("LLM-A", "A1.4, full LLM-found rule"), ("Abl-noSan", "$-$ input sanitisation"), ("Abl-noRep", "$-$ reputation memory"),
      ("Abl-noSan-noRep", "$-$ both (= core)"), ("Abl-M-normK1", "core, step $1/(k{+}1)$"), ("Abl-M-noProj", "core, no output projection"),
      ("Abl-M-tauHalf", "core, scale $c=0.5$"), ("Abl-M-linearW", "core, Huber weights ($p=1$)"), ("Abl-CG+San", "ClippedGossip + sanit."),
      ("ClippedGossip", "ClippedGossip (tuned)"), ("LLM-A-bydesign", "A4.4 (best \\emph{design} score)")]
T = [r"\begin{table}[t]", r"\centering", r"\caption{Ablation of the best LLM-found rule. The two mechanisms the LLM described as its key ideas (sanitisation, reputation) do not matter; the loss exponent $p$ and the scale $c$ do, and they move the rule along the frontier rather than improving it.}",
     r"\label{tab:ablation}", r"\setlength{\tabcolsep}{3pt}", r"\scriptsize", r"\resizebox{\columnwidth}{!}{\begin{tabular}{lccccc}", r"\toprule",
     r"Variant & gross mean & 5th pct. & min & benign & drift \\", r"\midrule"]
for n, d in AB:
    v = held(n)
    T.append(f"{d} & " + " & ".join(f"{x:.3f}" for x in v) + r" \\")
T += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(f"{P}/tab_ablation.tex", "w").write("\n".join(T))
for n, d in AB: print(f"{n:16s}", " ".join(f"{x:.3f}" for x in held(n)))
