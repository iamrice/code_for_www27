"""Generate LaTeX tables from result files."""
import os
import json, numpy as np
from arco_configs import CONFIGS
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results"
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/paper"
F = json.load(open(f"{R}/frontier_design.json"))
drift = json.load(open(f"{R}/heldout_drift.json"))
rs = {h["name"]: h for h in json.load(open(f"{R}/../design_runs/R0/history.json"))}
def design_val(n):
    if n in CONFIGS:
        c = CONFIGS[n]; return F[f"ARCO {c['loss']} {c['center']} q{c['q']} d{c['delta']} k{c['kappa']}"]["val"]
    if n == "Rand24first": return rs["rand0_012"]["val"]["score"]
    if n == "Rand200": return rs["rand0_197"]["val"]["score"]
    return F[n]["val"]
GROUPS = [("Hand-designed (tuned on design suite)", ["Mean", "CW-Median", "BRIDGE-T", "Krum", "GeoMed", "Huber-IRLS", "ClippedGossip", "BALANCE"]),
          ("Automated search", ["Rand24first", "Rand200", "LLM-A", "LLM-B", "LLM-C", "LLM-H"]),
          ("ARCO (ours, frozen)", ["ARCO-G", "ARCO-B", "ARCO-H"])]
DISPLAY = {"Rand24first": "Random search (24)", "Rand200": "Random search (200)", "LLM-A": "LLM run A", "LLM-B": "LLM run B",
           "LLM-C": "LLM run C", "LLM-H": "LLM run H (small)", "Huber-IRLS": "Huber (IRLS)"}
def rel(rows): return np.array([r["acc"] / r["oracle_acc"] for r in rows])
def main_table():
    cols = []
    data = {}
    for _, names in GROUPS:
        for n in names:
            H = json.load(open(f"{R}/heldout/{n}.json"))
            allg = rel([r for s in H.values() for r in s["rows"] if r["scenario"][2] != "none"])
            none = rel([r for s in H.values() for r in s["rows"] if r["scenario"][2] == "none"])
            d = {T: np.mean([o["rel"] for o in drift if o["method"] == n and o["T"] == T]) for T in (10, 40)}
            data[n] = [design_val(n), rel(H["heldout_attack"]["rows"]).mean(), rel(H["heldout_data"]["rows"]).mean(),
                       rel(H["heldout_topo"]["rows"]).mean(), rel(H["heldout_frac"]["rows"]).mean(),
                       np.quantile(allg, 0.05), allg.min(), none.mean(), d[10], d[40]]
    M = np.array(list(data.values()))
    best = M.max(0)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Held-out evaluation of frozen rules (relative accuracy = honest consensus accuracy / oracle accuracy; 432 scenarios, 396 of them with an active attack, plus 24 drift scenarios). Design val.\ is the only number any rule was tuned or selected on. Bold: best in column; no rule is best in all.}",
         r"\label{tab:heldout}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l c cccc cc c cc}", r"\toprule",
         r" & design & \multicolumn{4}{c}{mean over held-out suite} & \multicolumn{2}{c}{gross-attack tail} & benign & \multicolumn{2}{c}{drift} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){10-11}",
         r"Rule & val. & attacks & data & topology & Byz.\ frac. & 5th pct. & min & no attack & $T{=}10$ & $T{=}40$ \\", r"\midrule"]
    for g, names in GROUPS:
        L.append(rf"\multicolumn{{11}}{{l}}{{\emph{{{g}}}}} \\")
        for n in names:
            v = data[n]
            cells = [(rf"\textbf{{{x:.3f}}}" if abs(x - b) < 5e-4 else f"{x:.3f}") for x, b in zip(v, best)]
            L.append(f"{DISPLAY.get(n, n)} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if g != GROUPS[-1][0] else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table*}"]
    open(f"{P}/tab_heldout.tex", "w").write("\n".join(L))
    return data
def attack_table():
    atks = ["none", "gaussian", "signflip", "labelflip", "alie", "ipm", "mimic", "zerosum", "dissensus", "targeted", "adaptive"]
    names = ["Mean", "BRIDGE-T", "GeoMed", "ClippedGossip", "BALANCE", "LLM-A", "ARCO-G", "ARCO-B"]
    L = [r"\begin{table}[t]", r"\centering", r"\caption{Per-attack relative accuracy on unseen datasets (CIFAR-10, FMNIST-0.1; suite \texttt{heldout\_data}). $^\dagger$: attack never seen during design. Adaptive = black-box optimisation against the frozen rule.}",
         r"\label{tab:attacks}", r"\setlength{\tabcolsep}{1.5pt}", r"\scriptsize",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{l" + "c" * len(names) + "}", r"\toprule", "Attack & " + " & ".join(DISPLAY.get(n, n).replace("ClippedGossip", "ClipG.").replace("BRIDGE-T", "BRIDGE") for n in names) + r" \\", r"\midrule"]
    seen = {"none", "gaussian", "signflip", "labelflip", "alie"}
    for a in atks:
        vals = []
        for n in names:
            rows = json.load(open(f"{R}/heldout/{n}.json"))["heldout_data"]["rows"]
            vals.append(np.mean([r["acc"] / r["oracle_acc"] for r in rows if r["scenario"][2] == a]))
        b = max(vals)
        L.append(a + ("" if a in seen else r"$^\dagger$") + " & " + " & ".join(rf"\textbf{{{v:.2f}}}" if abs(v - b) < 5e-3 else f"{v:.2f}" for v in vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(f"{P}/tab_attacks.tex", "w").write("\n".join(L))
if __name__ == "__main__":
    d = main_table(); attack_table()
    for k, v in d.items(): print(f"{k:14s}", " ".join(f"{x:.3f}" for x in v))
