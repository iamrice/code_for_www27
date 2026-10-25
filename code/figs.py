import os
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ""
FIG = f"{ROOT}/paper/fig"
C = dict(A="#2a78d6", B="#1baf7a", C="#6250d6", H="#eb6834", R="#8a8984", base="#52514e")
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6})
LABEL = dict(A="LLM run A", B="LLM run B", C="LLM run C", H="LLM run H (small)", R="Random search")

def hist(r):
    return json.load(open(f"{ROOT}/design_runs/{r}/history.json"))

def fig_trajectory():
    base_val = json.load(open(f"{ROOT}/results/baselines_design_val.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for r in "ABCH":
        h = hist(r)
        n = np.arange(1, len(h) + 1)
        d = np.maximum.accumulate([x["design"]["score"] for x in h])
        # validation score of the candidate that the selection rule would pick so far
        best_val = np.maximum.accumulate([x["val"]["score"] for x in h])
        axes[0].plot(n, d, color=C[r], label=LABEL[r])
        axes[1].plot(n, best_val, color=C[r], label=LABEL[r])
    R = hist("R0")
    for ax, key in zip(axes, ["design", "val"]):
        runs = []
        for c in range(8):
            ch = R[c * 25:c * 25 + 24]
            runs.append(np.maximum.accumulate([x[key]["score"] for x in ch]))
        runs = np.array(runs)
        n = np.arange(1, 25)
        ax.fill_between(n, runs.min(0), runs.max(0), color=C["R"], alpha=0.18, lw=0)
        ax.plot(n, np.median(runs, 0), color=C["R"], ls="--", label="Random search (8 runs, median/range)")
        ax.set_xlabel("programs evaluated")
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    cg = base_val["ClippedGossip"]["summary"]["score"]
    axes[1].axhline(cg, color=C["base"], lw=0.9, ls=":")
    axes[1].text(24, cg + 0.002, "tuned ClippedGossip", ha="right", va="bottom", fontsize=7, color=C["base"])
    bd = json.load(open(f"{ROOT}/results/baselines_design.json"))["BALANCE"]["summary"]["score"]
    axes[0].axhline(bd, color=C["base"], lw=0.9, ls=":")
    axes[0].text(24, bd + 0.002, "best tuned baseline (BALANCE)", ha="right", va="bottom", fontsize=7, color=C["base"])
    axes[0].set_ylabel("best design score so far"); axes[1].set_ylabel("best validation score so far")
    axes[0].set_ylim(0.76, 0.85); axes[1].set_ylim(0.74, 0.82)
    axes[0].set_title("(a) feedback suite (what the designer optimizes)", fontsize=8)
    axes[1].set_title("(b) validation suite (fresh Byzantine placements)", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{FIG}/trajectory.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/trajectory.png", dpi=160, bbox_inches="tight")


BASE_NAMES = ["Mean", "CW-Median", "BRIDGE-T", "Krum", "GeoMed", "ClippedGossip", "BALANCE", "Huber-IRLS"]
LLM_NAMES = ["LLM-A", "LLM-B", "LLM-C", "LLM-H"]
ARCO_NAMES = ["ARCO-G", "ARCO-B", "ARCO-H"]

def heldout_points():
    import glob, os
    drift = json.load(open(f"{ROOT}/results/heldout_drift.json"))
    pts = {}
    for f in glob.glob(f"{ROOT}/results/heldout/*.json"):
        n = os.path.basename(f)[:-5]
        if n.startswith("Abl") or "bydesign" in n or n == "Rand24":
            continue
        H = json.load(open(f))
        rel = [r["acc"] / r["oracle_acc"] for s in H.values() for r in s["rows"]]
        dr = [o["rel"] for o in drift if o["method"] == n and o["T"] == 40]
        if dr:
            none = [r["acc"] / r["oracle_acc"] for s in H.values() for r in s["rows"] if r["scenario"][2] == "none"]
            gross = [r["acc"] / r["oracle_acc"] for s in H.values() for r in s["rows"] if r["scenario"][2] != "none"]
            pts[n] = (float(np.quantile(gross, 0.05)), float(np.mean(dr)), float(np.mean(none)))
    return pts

def pareto(P):
    keys = [k for k in P if not any(P[o][0] >= P[k][0] and P[o][1] >= P[k][1] and (P[o][0] > P[k][0] or P[o][1] > P[k][1]) for o in P)]
    return sorted(keys, key=lambda k: P[k][0])

def fig_frontier():
    F = json.load(open(f"{ROOT}/results/frontier_design.json"))
    design = {k: (v["val"], v["drift40"]) for k, v in F.items()}
    for n in ARCO_NAMES:  # frozen configs = specific grid points
        from arco_configs import CONFIGS
        c = CONFIGS[n]
        key = f"ARCO {c['loss']} {c['center']} q{c['q']} d{c['delta']} k{c['kappa']}"
        design[n] = design[key]
    HP = heldout_points()
    held = {k: v[:2] for k, v in HP.items()}
    held2 = {k: (v[2], v[1]) for k, v in HP.items()}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, P, title, xl in [(axes[0], design, "(a) design data: tuning view", "gross robustness (val. score)"),
                             (axes[1], held, "(b) held-out: attack tail", "gross robustness (5th pct.)"),
                             (axes[2], held2, "(c) held-out: benign cost", "benign accuracy (no attack)")]:
        grid = {k: v for k, v in P.items() if k.startswith("ARCO ")}
        if grid:
            ax.scatter([v[0] for v in grid.values()], [v[1] for v in grid.values()], s=9, color="#c3c2b7", lw=0, label="ARCO grid", zorder=1)
        named = {k: v for k, v in P.items() if not k.startswith("ARCO ")}
        pf = pareto(P)
        ax.plot([P[k][0] for k in pf], [P[k][1] for k in pf], color="#8a8984", lw=0.9, ls="--", zorder=2, label="Pareto front")
        for k, (x, y) in named.items():
            if k in ARCO_NAMES:
                col, mk, sz = C["A"], "o", 30
            elif k in LLM_NAMES:
                col, mk, sz = C["H"], "D", 22
            elif k.startswith("Rand"):
                col, mk, sz = C["C"], "s", 20
            else:
                col, mk, sz = C["base"], "^", 22
            ax.scatter([x], [y], s=sz, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3)
            if k in ("Mean", "Krum", "Huber-IRLS", "CW-Median", "BRIDGE-T", "GeoMed", "Rand200", "LLM-B", "LLM-C", "LLM-H", "Rand24first"):
                continue
            ax.annotate(k, (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6, color="#3d3c39")
        ax.set_title(title, fontsize=8); ax.set_xlabel(xl)
    axes[0].set_ylabel("drift robustness (rel. acc., T=40)")
    axes[0].set_xlim(0.55, 0.84); axes[0].set_ylim(0.72, 0.99)
    axes[1].set_xlim(0.35, 0.85); axes[1].set_ylim(0.70, 0.96)
    axes[2].set_xlim(0.88, 0.985); axes[2].set_ylim(0.70, 0.96)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="#c3c2b7", marker="o", ls="", ms=4, label="ARCO grid"),
          Line2D([], [], color=C["A"], marker="o", ls="", ms=5, label="ARCO (frozen)"),
          Line2D([], [], color=C["base"], marker="^", ls="", ms=5, label="hand-designed (tuned)"),
          Line2D([], [], color=C["H"], marker="D", ls="", ms=4.5, label="LLM-designed"),
          Line2D([], [], color=C["C"], marker="s", ls="", ms=4.5, label="random search"),
          Line2D([], [], color="#8a8984", ls="--", label="Pareto front")]
    fig.legend(handles=hs, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{FIG}/frontier.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/frontier.png", dpi=160, bbox_inches="tight")
    print({k: tuple(round(x, 3) for x in v) for k, v in held.items()})


MAIN = [("ARCO-G", C["A"], "-"), ("ARCO-B", C["A"], "--"), ("LLM-A", C["H"], "-"), ("BALANCE", "#1baf7a", "-"),
        ("ClippedGossip", C["C"], "-"), ("GeoMed", "#8a8984", "-"), ("BRIDGE-T", "#8a8984", ":")]

def fig_frac():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, mode in zip(axes, ("random", "localized")):
        for n, col, ls in MAIN:
            rows = json.load(open(f"{ROOT}/results/heldout/{n}.json"))["heldout_frac"]["rows"]
            fr = (0.1, 0.2, 0.3, 0.4)
            mean = [np.mean([r["acc"] / r["oracle_acc"] for r in rows if r["scenario"][4] == f and r["scenario"][5] == mode]) for f in fr]
            worst = [np.quantile([r["acc"] / r["oracle_acc"] for r in rows if r["scenario"][4] == f and r["scenario"][5] == mode], 0.1) for f in fr]
            ax.plot(fr, mean, color=col, ls=ls, marker="o", ms=3.5, label=n)
        ax.set_title(f"{mode} Byzantine placement", fontsize=8)
        ax.set_xlabel("fraction of Byzantine nodes"); ax.set_xticks([0.1, 0.2, 0.3, 0.4])
    axes[0].set_ylabel("rel. accuracy (CIFAR-10)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.07), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{FIG}/frac.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/frac.png", dpi=160, bbox_inches="tight")

def fig_driftT():
    out = json.load(open(f"{ROOT}/results/drift_T_frozen.json"))
    Ts = sorted({o["T"] for o in out})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, atk, title in zip(axes, ("drift", "none"), ("(a) inlier drift attack", "(b) no attack")):
        for n, col, ls in MAIN + [("ARCO-H", C["A"], ":"), ("Mean", "#c3c2b7", "-")]:
            y = [np.mean([o["rel"] for o in out if o["method"] == n and o["T"] == T and o["attack"] == atk]) for T in Ts]
            if not np.all(np.isfinite(y)):
                continue
            ax.plot(Ts, y, color=col, ls=ls, marker="o", ms=3, label=n)
        ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts])
        ax.set_xlabel("consensus iterations T"); ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("rel. accuracy (held-out data)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=9, bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{FIG}/driftT.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/driftT.png", dpi=160, bbox_inches="tight")


def fig_kappa():
    out = json.load(open(f"{ROOT}/results/kappa_sweep.json"))
    K = sorted({o["kappa"] for o in out})
    xs = [k if k > 0 else 0.01 for k in K]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    cols = {"none": "#8a8984", "drift": C["A"], "labelflip": C["H"], "targeted": "#1baf7a"}
    for ax, cfg in zip(axes[:2], ("ARCO-B", "ARCO-G")):
        for a in ("none", "drift", "labelflip", "targeted"):
            y = [np.mean([o["rel"] for o in out if o["cfg"] == cfg and o["kappa"] == k and o["attack"] == a]) for k in K]
            ax.plot(xs, y, color=cols[a], marker="o", ms=3, label=a if a != "none" else "no attack")
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["0"] + [f"{k:g}" for k in K[1:]], fontsize=6)
        ax.set_title(f"({'a' if cfg == 'ARCO-B' else 'b'}) {cfg}: rel. accuracy", fontsize=8); ax.set_xlabel(r"anchor strength $\kappa$")
    for cfg, ls in (("ARCO-B", "--"), ("ARCO-G", "-")):
        y = [np.mean([o["disagree"] for o in out if o["cfg"] == cfg and o["kappa"] == k and o["attack"] == "none"]) for k in K]
        axes[2].plot(xs, y, color=C["A"], ls=ls, marker="o", ms=3, label=cfg)
    axes[2].set_xscale("log"); axes[2].set_xticks(xs); axes[2].set_xticklabels(["0"] + [f"{k:g}" for k in K[1:]], fontsize=6)
    axes[2].set_title("(c) honest disagreement (no attack)", fontsize=8); axes[2].set_xlabel(r"anchor strength $\kappa$"); axes[2].legend(fontsize=6.5)
    axes[2].set_ylabel("mean TV to honest centroid")
    axes[0].set_ylabel("rel. accuracy (T=40)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.36, 1.08), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{FIG}/kappa.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/kappa.png", dpi=160, bbox_inches="tight")

if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f"fig_{f}"]()
