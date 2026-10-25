"""ARCO configurations frozen from the design-side frontier (results/frontier_design.json)."""
from arco import make_arco
CONFIGS = {
    "ARCO-G": dict(loss="log", center="self", q=0.5, delta=0.5, kappa=0.1),    # max design-val score
    "ARCO-B": dict(loss="log", center="self", q=0.25, delta=2.0, kappa=0.1),   # max mean(val, drift40)
    "ARCO-H": dict(loss="huber", center="self", q=0.25, delta=1.0, kappa=0.1), # convex member (certified)
}
def get(name):
    return make_arco(**CONFIGS[name])
