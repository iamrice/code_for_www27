"""Tables for the WWW version: held-out table with the Web-text suite, and a compact two-dataset end-to-end table."""
import json, glob, os, collections
import numpy as np
import tables as TB

R, P = TB.R, TB.P.replace("/paper", "/paper_www")
WEB = json.load(open(f"{R}/heldout_web.json"))


def heldout_table():
    data = TB.main_table()  # also rewrites ../paper/tab_heldout.tex (unchanged content)
    for n in data:
        rows = WEB[n]["rows"]
        att = np.array([r["rel"] for r in rows if r["scenario"][2] not in ("none", "drift")])
        none = np.mean([r["rel"] for r in rows if r["scenario"][2] == "none"])
        dr = np.mean([r["rel"] for r in rows if r["scenario"][2] == "drift"])
        data[n] = list(data[n]) + [att.mean(), att.min(), none, dr]
    M = np.array(list(data.values()))
    best = M.max(0)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Held-out evaluation of frozen rules (relative accuracy = honest consensus accuracy / oracle accuracy). Image suites: 432 scenarios (396 attacked) plus 24 drift scenarios. Web text: 20 Newsgroups knowledge, 72 scenarios, all 12 attacks. Design val.\ is the only number any rule was tuned or selected on. Bold: best in column. No rule is best in all columns.}",
         r"\label{tab:heldout}", r"\setlength{\tabcolsep}{3pt}", r"\small",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{l c cccc cc c cc cccc}", r"\toprule",
         r" & design & \multicolumn{4}{c}{image: mean over held-out suite} & \multicolumn{2}{c}{image: gross tail} & image & \multicolumn{2}{c}{image: drift} & \multicolumn{4}{c}{Web text (20 Newsgroups)} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){10-11}\cmidrule(lr){12-15}",
         r"Rule & val. & attacks & data & topology & Byz.\ frac. & 5th pct. & min & benign & $T{=}10$ & $T{=}40$ & attacked & min & benign & drift \\", r"\midrule"]
    for g, names in TB.GROUPS:
        L.append(rf"\multicolumn{{15}}{{l}}{{\emph{{{g}}}}} \\")
        for n in names:
            cells = [(rf"\textbf{{{x:.3f}}}" if abs(x - b) < 5e-4 else f"{x:.3f}") for x, b in zip(data[n], best)]
            L.append(f"{TB.DISPLAY.get(n, n)} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if g != TB.GROUPS[-1][0] else r"\bottomrule")
    L += [r"\end{tabular}}", r"\end{table*}"]
    open(f"{P}/tab_heldout_www.tex", "w").write("\n".join(L))
    return data


ATK = ["none", "signflip", "alie", "ipm", "dissensus", "targeted", "adaptive", "drift"]
METH = [("Local", "Local only"), ("DFedAvg", "DFedAvg (param.)"), ("Mean", "Gossip averaging"), ("CW-Median", "CW-Median"),
        ("BRIDGE-T", "BRIDGE-T"), ("GeoMed", "GeoMed"), ("ClippedGossip", "ClippedGossip"), ("BALANCE", "BALANCE"),
        ("Rand200", "Random search (200)"), ("LLM-H", "LLM run H"), ("LLM-A", "LLM run A"),
        ("ARCO-G", "ARCO-G"), ("ARCO-B", "ARCO-B"), ("ARCO-H", "ARCO-H")]


def load_e2e():
    """acc[(dataset, method, attack)] -> list over seeds of final honest accuracy (%)."""
    acc = collections.defaultdict(list)
    for f in glob.glob(f"{R}/e2e/*.json"):
        n = os.path.basename(f)[:-5]
        parts = n.split("_")
        ds = "cifar10"
        if parts[0] == "news20":
            ds, parts = "news20", parts[1:]
        if parts[-1] in ("s0", "s1", "s2"):
            parts = parts[:-1]
        m, a = "_".join(parts[:-1]), parts[-1]
        acc[(ds, m, a)].append(100 * json.load(open(f))["final"]["acc_honest"])
    return acc


def cell(acc, ds, m, a):
    v = acc.get((ds, "Local", "none")) if m == "Local" else acc.get((ds, m, a))
    return v or None


def e2e_tables():
    acc = load_e2e()
    # compact main-text table: per dataset benign / mean over attacked columns / worst attacked / drift
    cols = [(ds, k) for ds in ("cifar10", "news20") for k in ("benign", "mean", "worst", "drift")]
    vals = {}
    for m, _ in METH:
        row = []
        for ds, k in cols:
            per = {a: cell(acc, ds, m, a) for a in ATK}
            if k == "benign":
                row.append(np.mean(per["none"]) if per["none"] else None)
            elif k == "drift":
                row.append(np.mean(per["drift"]) if per["drift"] else None)
            else:
                att = [np.mean(per[a]) for a in ATK[1:-1] if per[a]]
                ok = len(att) == len(ATK) - 2
                row.append((np.mean(att) if k == "mean" else np.min(att)) if ok else None)
        vals[m] = row
    best = [max((v[c] for mm, v in vals.items() if v[c] is not None and mm not in ("Local", "DFedAvg")), default=0) for c in range(len(cols))]
    fmt = lambda x, b: "--" if x is None else (rf"\textbf{{{x:.1f}}}" if abs(x - b) < 0.05 else f"{x:.1f}")
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{End-to-end DFL (20 nodes, 20 rounds, 20\% Byzantine): honest test accuracy (\%). \emph{mean}/\emph{worst}: over the six non-drift attacks (sign flip, ALIE, IPM, dissensus, targeted, adaptive). CIFAR-10: one or two seeds per cell; 20 Newsgroups: three seeds. DFedAvg exchanges parameters and cannot express several knowledge-space attacks (--). Per-attack results are in the extended version.}",
         r"\label{tab:e2e}", r"\setlength{\tabcolsep}{2.2pt}", r"\footnotesize",
         r"\begin{tabular}{l cccc cccc}", r"\toprule",
         r" & \multicolumn{4}{c}{CIFAR-10} & \multicolumn{4}{c}{20 Newsgroups (Web text)} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Method & benign & mean & worst & drift & benign & mean & worst & drift \\", r"\midrule"]
    for m, disp in METH:
        L.append(disp + " & " + " & ".join(fmt(x, b) for x, b in zip(vals[m], best)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{P}/tab_e2e_www.tex", "w").write("\n".join(L))
    # full per-attack tables (extended version)
    for ds, title in (("cifar10", "CIFAR-10"), ("news20", "20 Newsgroups")):
        T = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{End-to-end DFL on {title}, per attack: honest test accuracy (\%), mean $\pm$ std over seeds (number of seeds in parentheses where it differs from the table default).}}",
             rf"\label{{tab:e2e_full_{ds}}}", r"\setlength{\tabcolsep}{3pt}", r"\small",
             r"\begin{tabular}{l" + "c" * len(ATK) + "}", r"\toprule", "Method & " + " & ".join(ATK) + r" \\", r"\midrule"]
        for m, disp in METH:
            cs = []
            for a in ATK:
                v = cell(acc, ds, m, a)
                if not v:
                    cs.append("--")
                elif len(v) == 1:
                    cs.append(f"{v[0]:.1f}")
                else:
                    cs.append(f"{np.mean(v):.1f}$\\pm${np.std(v):.1f} ({len(v)})")
            T.append(disp + " & " + " & ".join(cs) + r" \\")
        T += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        open(f"{P}/tab_e2e_full_{ds}.tex", "w").write("\n".join(T))
    return vals


if __name__ == "__main__":
    heldout_table()
    v = e2e_tables()
    for m, _ in METH:
        print(f"{m:14s}", " ".join("  --" if x is None else f"{x:5.1f}" for x in v[m]))
