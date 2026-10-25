"""End-to-end decentralized FL via knowledge consensus (or parameter gossip baseline).

python dfl.py --dataset cifar10 --method LLM:path/to/program.py --attack ipm --seed 0
"""
import argparse
import json
import os
import time
import numpy as np
import torch

from fl import ROOT, DEV, load_dataset, partition, BatchedCNN, BatchedMLP, NodeData, local_train, distill, evaluate
from consensus import make_graph, place_byzantine, run_consensus, finalize, BASELINES, ATTACKS
from programs import load_program


def get_agg(method):
    if method in BASELINES:
        return BASELINES[method]
    if method.startswith("PROG:"):
        return load_program(method[5:])
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fmnist")
    ap.add_argument("--method", default="Mean")  # baseline name | PROG:file | Local | DFedAvg
    ap.add_argument("--attack", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--byz_frac", type=float, default=0.2)
    ap.add_argument("--byz_mode", default="random")
    ap.add_argument("--n_noisy", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--topo", default="er")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--m_pub", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--distill_steps", type=int, default=50)
    ap.add_argument("--per_node", type=int, default=None)
    ap.add_argument("--dump", default=None, help="save knowledge snapshots at these rounds, e.g. 1,4,10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    xtr, ytr, xte, yte = load_dataset(a.dataset)
    C = int(ytr.max()) + 1
    pub, parts = partition(ytr, a.n, a.m_pub, a.alpha, rng, a.per_node)
    adj = make_graph(a.topo, a.n, a.seed)
    byz = place_byzantine(adj, a.byz_frac, a.byz_mode, a.seed) if a.attack != "clean" else set()
    honest = [i for i in range(a.n) if i not in byz]
    noisy = set(np.random.default_rng(a.seed + 7).choice(honest, a.n_noisy, replace=False).tolist())
    data = NodeData(xtr, ytr, parts, noisy, a.noise, C, rng)
    if xtr.dim() == 2:
        net = BatchedMLP(a.n, xtr.shape[1], C, seed=a.seed)
    else:
        net = BatchedCNN(a.n, xtr.shape[1], xtr.shape[2], C, seed=a.seed)
    opt = torch.optim.Adam(net.params.values(), 1e-3)
    gen = torch.Generator().manual_seed(a.seed)
    Xpub, ypub = xtr[pub], ytr[pub].numpy()
    te_idx = np.random.default_rng(1).choice(len(yte), 2000, replace=False)
    victim = sorted(noisy)[0]
    attack = "none" if a.attack == "clean" else a.attack
    hmask = [0.0 if i in byz else 1.0 for i in range(a.n)]
    log = dict(args=vars(a), byz=sorted(byz), noisy=sorted(noisy), rounds=[])
    snaps = {}
    dumps = [int(r) for r in a.dump.split(",")] if a.dump else []
    t_start = time.time()
    comm_floats = 0
    local_train(net, opt, data, 2 * a.local_steps, 32, gen, [1.0] * a.n)  # warm-up
    for r in range(1, a.rounds + 1):
        local_train(net, opt, data, a.local_steps, 32, gen, [1.0] * a.n)
        info = {}
        if a.method == "Local":
            pass
        elif a.method == "DFedAvg":
            V = net.flat().cpu().numpy().astype(np.float64)
            res = run_consensus(V[:, None, :], adj, byz, attack if attack in
                                ("none", "gaussian", "signflip", "alie", "ipm", "zerosum", "dissensus", "mimic")
                                else "none", BASELINES["Mean"], T=1, seed=a.seed * 1000 + r, victim=victim)
            Vn = V.copy()
            for i in honest:
                Vn[i] = res["raw"][i].ravel()
            net.load_flat(torch.tensor(Vn, dtype=torch.float32, device=DEV))
            comm_floats += V.shape[1]
        else:
            P = net.predict(Xpub).cpu().numpy().astype(np.float64)
            if r in dumps:
                snaps[r] = P.astype(np.float32)
            agg = get_agg(a.method)
            res = run_consensus(P, adj, byz, attack, agg, T=a.T, seed=a.seed * 1000 + r, victim=victim)
            tg = np.stack([res["targets"][i] if i in res["targets"] else P[i] for i in range(a.n)])
            info["kd_acc"] = float(np.mean([(tg[i].argmax(-1) == ypub).mean() for i in honest]))
            info["local_kd_acc"] = float(np.mean([(P[i].argmax(-1) == ypub).mean() for i in honest]))
            info["cons_time"] = res["time"]
            comm_floats += a.T * P.shape[1] * P.shape[2]
            distill(net, opt, data, Xpub, torch.tensor(tg, dtype=torch.float32), a.distill_steps, 32, gen, hmask)
        acc = evaluate(net, xte[te_idx], yte[te_idx])
        info.update(round=r, acc_honest=float(np.mean(acc[honest])),
                    acc_clean=float(np.mean([acc[i] for i in honest if i not in noisy])),
                    acc_noisy=float(np.mean([acc[i] for i in noisy])), time=time.time() - t_start)
        log["rounds"].append(info)
        print(json.dumps(info), flush=True)
    acc = evaluate(net, xte, yte)
    log["final"] = dict(acc_honest=float(np.mean(acc[honest])), acc_min=float(np.min(acc[honest])),
                        acc_noisy=float(np.mean([acc[i] for i in noisy])),
                        per_node=acc.tolist(), comm_floats_per_link=comm_floats,
                        n_params=int(net.flat().shape[1]), time=time.time() - t_start)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=1)
    if snaps:
        np.savez_compressed(a.out.replace(".json", "_snaps.npz"), labels=ypub, adj=np.array(adj, dtype=object),
                            **{f"r{k}": v for k, v in snaps.items()})
    print("FINAL", json.dumps({k: v for k, v in log["final"].items() if k != "per_node"}))


if __name__ == "__main__":
    main()
