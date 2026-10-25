"""Generate the extended appendix (paper_www/supp_gen/*.tex and figures) from the raw result files."""
import json, glob, os, re, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tables as TB

ROOT = TB.R.rsplit("/results", 1)[0]
R, OUT = TB.R, f"{ROOT}/paper_www/supp_gen"
os.makedirs(f"{OUT}/fig", exist_ok=True)
NAMES = [n for _, ns in TB.GROUPS for n in ns]
DISP = lambda n: TB.DISPLAY.get(n, n)
ALLATK = ["none", "gaussian", "signflip", "labelflip", "alie", "ipm", "mimic", "zerosum", "dissensus", "targeted", "adaptive"]
SEEN = {"none", "gaussian", "signflip", "labelflip", "alie"}
H = {n: json.load(open(f"{R}/heldout/{n}.json")) for n in NAMES}
WEB = json.load(open(f"{R}/heldout_web.json"))
rel = lambda r: r["acc"] / r["oracle_acc"]


def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def table(path, caption, label, header, rows, colspec, wide=True, size=r"\scriptsize", bold_cols=None):
    """rows: list of (name, [floats or str]); bold_cols: indices where the max is bolded."""
    env = "table*" if wide else "table"
    nums = [[v for v in r[1]] for r in rows if r is not None and not isinstance(r, str)]
    best = {}
    for c in (bold_cols or []):
        col = [x[c] for x in nums if isinstance(x[c], float)]
        best[c] = max(col) if col else None
    L = [rf"\begin{{{env}}}[tp]" if wide else rf"\begin{{{env}}}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         size, r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    for r in rows:
        if isinstance(r, str):
            L.append(r)
            continue
        cells = []
        for c, v in enumerate(r[1]):
            if isinstance(v, float):
                s = f"{v:.3f}"
                if c in best and best[c] is not None and abs(v - best[c]) < 5e-4:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            else:
                cells.append(str(v))
        L.append(r[0] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    open(f"{OUT}/{path}", "w").write("\n".join(L) + "\n")


def grouped_rows(fn, ncols):
    rows = []
    for g, ns in TB.GROUPS:
        rows.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{g}}}}} \\")
        rows += [(DISP(n), fn(n)) for n in ns]
    return rows


# ---------------------------------------------------------------- per-attack tables per held-out suite
def per_attack(suite, atks, cap, label, src=None):
    def fn(n):
        rows = (src or H)[n][suite]["rows"] if src is None else src[n]["rows"]
        out = []
        for a in atks:
            v = [rel(r) for r in rows if r["scenario"][2] == a]
            out.append(float(np.mean(v)) if v else "--")
        allv = [rel(r) for r in rows if r["scenario"][2] != "none"]
        return out + [float(np.mean(allv)), float(np.min(allv))]
    hdr = "Rule & " + " & ".join(a + ("" if a in SEEN else r"$^\dagger$") for a in atks) + r" & mean$_{\rm att}$ & min$_{\rm att}$"
    table(f"pa_{label}.tex", cap, f"tab:pa_{label}", hdr, grouped_rows(fn, len(atks) + 3), "l" + "c" * (len(atks) + 2),
          bold_cols=list(range(len(atks) + 2)))


per_attack("heldout_attack", ["ipm", "mimic", "zerosum", "dissensus", "targeted", "adaptive"],
           r"Suite \texttt{heldout\_attack} (FMNIST knowledge, unseen attacks, 36 scenarios): relative accuracy per attack, mean over FL rounds $\{1,4,10\}$ and two Byzantine placements.", "attack")
per_attack("heldout_data", ALLATK,
           r"Suite \texttt{heldout\_data} (CIFAR-10 and FMNIST-$\alpha{=}0.1$ knowledge, 132 scenarios): relative accuracy per attack. $^\dagger$: attack never seen by any designer.", "data")
per_attack("heldout_topo", ALLATK,
           r"Suite \texttt{heldout\_topo} (CIFAR-10 knowledge at round 4 on ring, Watts--Strogatz, random geometric and sparse ER graphs, 88 scenarios): relative accuracy per attack, mean over the four topologies.", "topo")
per_attack("heldout_frac", ALLATK,
           r"Suite \texttt{heldout\_frac} (CIFAR-10 knowledge at round 4, Byzantine fraction 10--40\%, random and localized placement, 176 scenarios): relative accuracy per attack.", "frac")
per_attack(None, ALLATK + ["drift"],
           r"Web-text suite (20 Newsgroups knowledge, 20 classes, 72 scenarios): relative accuracy per attack, mean over FL rounds $\{1,4,10\}$ and two placements.", "web", src=WEB)


# ---------------------------------------------------------------- topology, fraction, round breakdowns
TOPOS = ["ring", "ws", "geo", "er_sparse"]


def topo_fn(n):
    rows = H[n]["heldout_topo"]["rows"]
    out = []
    for t in TOPOS:
        v = [rel(r) for r in rows if r["scenario"][3] == t and r["scenario"][2] != "none"]
        out += [float(np.mean(v)), float(np.min(v))]
    return out


table("topo.tex", r"Per-topology relative accuracy (suite \texttt{heldout\_topo}, attacked scenarios): mean and minimum over the 10 active attacks and two placements.",
      "tab:topo", "Rule & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t.replace('_', ' ')}}}" for t in TOPOS) + r" \\ & " + " & ".join(["mean & min"] * 4),
      grouped_rows(topo_fn, 9), "l" + "cc" * 4, bold_cols=list(range(8)))


def frac_fn(n):
    rows = H[n]["heldout_frac"]["rows"]
    out = []
    for mode in ("random", "localized"):
        for f in (0.1, 0.2, 0.3, 0.4):
            v = [rel(r) for r in rows if abs(r["scenario"][4] - f) < 1e-9 and r["scenario"][5] == mode and r["scenario"][2] != "none"]
            out.append(float(np.mean(v)))
    return out


table("frac.tex", r"Relative accuracy vs.\ Byzantine fraction and placement (suite \texttt{heldout\_frac}, mean over the 10 active attacks and two seeds).",
      "tab:fractab", r"Rule & \multicolumn{4}{c}{random placement} & \multicolumn{4}{c}{localized placement} \\ & 10\% & 20\% & 30\% & 40\% & 10\% & 20\% & 30\% & 40\%",
      grouped_rows(frac_fn, 9), "l" + "c" * 8, bold_cols=list(range(8)))


def round_fn(n):
    out = []
    for ds in ("cifar10", "fmnist_a01"):
        for r_ in (1, 4, 10):
            v = [rel(r) for r in H[n]["heldout_data"]["rows"] if r["scenario"][0] == ds and r["scenario"][1] == r_ and r["scenario"][2] != "none"]
            out.append(float(np.mean(v)))
    for r_ in (1, 4, 10):
        v = [rel(r) for r in WEB[n]["rows"] if r["scenario"][1] == r_ and r["scenario"][2] not in ("none", "drift")]
        out.append(float(np.mean(v)))
    return out


table("rounds.tex", r"Relative accuracy by FL round of the knowledge snapshot (attacked scenarios). Early knowledge (round 1) is weak and heterogeneous; late knowledge (round 10) is strong and similar.",
      "tab:rounds", r"Rule & \multicolumn{3}{c}{CIFAR-10} & \multicolumn{3}{c}{FMNIST-0.1} & \multicolumn{3}{c}{20 Newsgroups} \\ & r1 & r4 & r10 & r1 & r4 & r10 & r1 & r4 & r10",
      grouped_rows(round_fn, 10), "l" + "c" * 9, bold_cols=list(range(9)))

# ---------------------------------------------------------------- drift vs T, kappa sweep, bound tightness
DT = json.load(open(f"{R}/drift_T_frozen.json"))
Ts = sorted({o["T"] for o in DT})
meths = list(dict.fromkeys(o["method"] for o in DT))
rows = []
for m in meths:
    out = []
    for atk in ("drift", "none"):
        for T in Ts:
            v = [o["rel"] for o in DT if o["method"] == m and o["T"] == T and o["attack"] == atk]
            out.append(float(np.mean(v)) if v else "--")
    rows.append((DISP(m), out))
atks_dt = sorted({o["attack"] for o in DT})
table("driftT.tex", r"Relative accuracy vs.\ number of consensus iterations $T$ under the drift attack and with stubborn (\emph{none}) Byzantine nodes (held-out data; mean over datasets, rounds and seeds).",
      "tab:driftT", r"Rule & \multicolumn{" + str(len(Ts)) + r"}{c}{drift} & \multicolumn{" + str(len(Ts)) + r"}{c}{none (stubborn)} \\ & " + " & ".join([f"$T{{=}}{T}$" for T in Ts] * 2),
      rows, "l" + "c" * (2 * len(Ts)), wide=True)

KS = json.load(open(f"{R}/kappa_sweep.json"))
kap = sorted({o["kappa"] for o in KS})
rows = []
for cfg in dict.fromkeys(o["cfg"] for o in KS):
    for atk in dict.fromkeys(o["attack"] for o in KS):
        v = [float(np.mean([o["rel"] for o in KS if o["cfg"] == cfg and o["attack"] == atk and o["kappa"] == k])) for k in kap]
        rows.append((f"{cfg}, {atk}", v))
    v = [float(np.mean([o["disagree"] for o in KS if o["cfg"] == cfg and o["kappa"] == k])) for k in kap]
    rows.append((f"{cfg}, disagreement", [f"{x:.4f}" for x in v]))
    rows.append(r"\midrule")
rows = rows[:-1]
table("kappa.tex", r"Anchor sweep: relative accuracy per attack and honest disagreement (mean squared distance between honest targets) vs.\ $\kappa$ ($T=40$, held-out data). Only $\kappa$ is changed in the frozen configurations.",
      "tab:kappatab", "Configuration, attack & " + " & ".join(f"$\\kappa{{=}}{k:g}$" for k in kap), rows, "l" + "c" * len(kap), wide=True)

VB = json.load(open(f"{R}/verify_bound.json"))
rows = [(f"{o['attack']}", [f"{o['kappa']:g}", f"{o['delta']:g}", f"{o['dev']:.3f}", f"{o['bound']:.3f}", f"{100 * o['dev'] / o['bound']:.1f}\\%"]) for o in VB]
table("bound.tex", r"Tightness of Theorem~1: measured deviation $\|X^m-X^\star\|$ of the converged attacked solution from the attack-free one vs.\ the bound $\delta\|b\|_2/\kappa$ (projected gradient, absolute-scale Huber coupling, 3000 iterations, CIFAR-10 knowledge on 200 public items).",
      "tab:bound", r"Attack & $\kappa$ & $\delta$ & deviation & bound & ratio", rows, "lccccc", wide=False)

# ---------------------------------------------------------------- ARCO design grid
FD = json.load(open(f"{R}/frontier_design.json"))
rows = []
for k, v in FD.items():
    rows.append((tex_escape(k.replace("ARCO ", "")) if k.startswith("ARCO") else DISP(k), [v["val"], v["val_mean"], v["drift10"], v["drift40"], v["none40"]]))
table("grid.tex", r"Design-side map used for Fig.~1(a) and for selecting the frozen ARCO configurations: all 48 ARCO grid points (loss, centre, quantile $q$, scale $c$, anchor $\kappa$) and the 12 other rules. val = validation score; drift and none on FMNIST design data.",
      "tab:grid", r"Rule & val.\ score & val.\ mean & drift $T{=}10$ & drift $T{=}40$ & none $T{=}40$", rows, "lccccc", wide=True, bold_cols=[0, 1, 2, 3, 4])

# ---------------------------------------------------------------- LLM runs: every program
LITE = json.load(open(f"{R}/lite_all.json"))
for run in "ABCH":
    hist = json.load(open(f"{ROOT}/design_runs/{run}/history.json"))
    rows = []
    for h in hist:
        key = f"{run}/{h['name']}.py"
        lite = LITE.get(key, {}).get("mean_acc")
        rows.append((tex_escape(h["name"][:42]), [str(h["round"]), h["design"]["score"] if h.get("design") else "--",
                                                  h["val"]["score"] if h.get("val") else "--", float(lite) if lite is not None else "--"]))
    table(f"run_{run}.tex", rf"LLM design run {run}: every program, with the round it was proposed in, its design score (the feedback the agent saw), validation score (the selection criterion) and held-out-lite accuracy (never shown).",
          f"tab:run_{run}", "Program & round & design & val. & held-out-lite", rows, "lcccc", wide=False, bold_cols=[1, 2, 3])

# random search figure: design vs held-out-lite, with LLM programs overlaid
RS = json.load(open(f"{ROOT}/design_runs/R0/history.json"))
fig, ax = plt.subplots(figsize=(5.2, 3.4))
xs = [h["design"]["score"] for h in RS if h.get("design")]
ys = [LITE.get(f"R0/{h['name']}.py", {}).get("mean_acc", np.nan) for h in RS if h.get("design")]
ax.scatter(xs, ys, s=10, c="#9aa0a6", label="random search (200)")
for run, c in zip("ABCH", ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]):
    hist = json.load(open(f"{ROOT}/design_runs/{run}/history.json"))
    ax.scatter([h["design"]["score"] for h in hist if h.get("design")],
               [LITE.get(f"{run}/{h['name']}.py", {}).get("mean_acc", np.nan) for h in hist if h.get("design")], s=18, c=c, label=f"LLM run {run}")
ax.set_xlabel("design score (feedback)")
ax.set_ylabel("held-out-lite accuracy (never shown)")
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig/design_vs_lite.pdf")

# ---------------------------------------------------------------- end-to-end training curves
def curve(ds, m, a, seed_tag):
    pre = "news20_" if ds == "news20" else ""
    f = f"{R}/e2e/{pre}{m}_{a}{seed_tag}.json"
    if not os.path.exists(f):
        return None
    return [r["acc_honest"] * 100 for r in json.load(open(f))["rounds"]]


for ds, tag, title in (("cifar10", "", "CIFAR-10"), ("news20", "_s0", "20 Newsgroups")):
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, a in zip(axs, ["none", "signflip", "targeted", "drift"]):
        for m, c in [("Local", "#000000"), ("Mean", "#9aa0a6"), ("CW-Median", "#ff7f0e"), ("ClippedGossip", "#1f77b4"),
                     ("BALANCE", "#8c564b"), ("LLM-A", "#2ca02c"), ("ARCO-G", "#d62728"), ("ARCO-B", "#9467bd")]:
            y = curve(ds, m, "none" if m == "Local" else a, tag)
            if y:
                ax.plot(range(1, len(y) + 1), y, color=c, lw=1.4, label=m)
        ax.set_title(a, fontsize=9)
        ax.set_xlabel("FL round")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("honest test acc. (%)")
    axs[-1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle(title + " (seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig/curves_{ds}.pdf")

# ---------------------------------------------------------------- verbatim listings (ASCII-sanitized)
def ascii(s):
    rep = {"—": "--", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "≤": "<=", "≥": ">=",
           "×": "x", "→": "->", "←": "<-", "…": "...", "±": "+/-", "κ": "kappa", "δ": "delta",
           "∑": "sum", "∈": "in", "‖": "||", "√": "sqrt", "·": "*", "≈": "~", "α": "alpha"}
    for k, v in rep.items():
        s = s.replace(k, v)
    s = s.replace(ROOT, "<repo>")
    return "".join(ch if ord(ch) < 128 else "?" for ch in s)


def listing(name, path, caption):
    open(f"{OUT}/{name}", "w").write(ascii(open(path).read()))
    return rf"\lstinputlisting[caption={{{caption}}},label={{lst:{name.split(chr(46))[0]}}}]{{supp_gen/{name}}}"


L = []
L.append(listing("prereg.txt", f"{ROOT}/PREREGISTRATION.md", "Pre-registration file (verbatim), written before any automated design run."))
L.append(listing("prompt_A1.txt", f"{ROOT}/design_runs/A/prompt_r1.md", "Run A, round 1 prompt (verbatim). Later rounds append the five best previous programs with per-attack feedback."))
L.append(listing("prompt_A2_tail.txt", f"{ROOT}/design_runs/A/prompt_r2.md", "Run A, round 2 prompt (verbatim), showing the feedback format."))
L.append(listing("resp_A1.txt", f"{ROOT}/design_runs/A/response_r1.md", "Run A, round 1 response (verbatim), containing the selected program A1.4."))
for run, f in [("A", "A/A1.4_ReputationClippedGossip.py"), ("B", "B/B2.3_B2_3_TwoLevelRobustMean.py"), ("C", "C/C1.3_HellingerGM.py"), ("H", "H/H5.1_H5_1_PersistentBandClippedMix.py")]:
    L.append(listing(f"sel_{run}.py", f"{ROOT}/design_runs/{f}", f"Selected program of run {run} ({tex_escape(os.path.basename(f))})."))
L.append(listing("arco.py", f"{ROOT}/code/arco.py", "ARCO implementation used in all experiments."))
L.append(listing("arco_configs.py", f"{ROOT}/code/arco_configs.py", "The three frozen ARCO configurations."))
open(f"{OUT}/listings.tex", "w").write("\n\n".join(L) + "\n")
print("ok")
