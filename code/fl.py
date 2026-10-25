"""Batched decentralized knowledge-distillation FL on a single device.

All N node models are trained simultaneously as one grouped network (each node's
weights live in a slice of a stacked parameter tensor), which makes 20-node
simulations feasible on a laptop GPU.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ""


def load_dataset(name):
    if name == "fmnist":
        tr = torchvision.datasets.FashionMNIST(f"{ROOT}/data", train=True, download=True)
        te = torchvision.datasets.FashionMNIST(f"{ROOT}/data", train=False, download=True)
        xtr = tr.data.float().div(255).unsqueeze(1)
        xte = te.data.float().div(255).unsqueeze(1)
        mean, std = 0.286, 0.353
        xtr, xte = (xtr - mean) / std, (xte - mean) / std
        ytr, yte = tr.targets.clone(), te.targets.clone()
    elif name == "cifar10":
        tr = torchvision.datasets.CIFAR10(f"{ROOT}/data", train=True, download=True)
        te = torchvision.datasets.CIFAR10(f"{ROOT}/data", train=False, download=True)
        m = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
        s = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
        xtr = (torch.tensor(tr.data).permute(0, 3, 1, 2).float().div(255) - m) / s
        xte = (torch.tensor(te.data).permute(0, 3, 1, 2).float().div(255) - m) / s
        ytr, yte = torch.tensor(tr.targets), torch.tensor(te.targets)
    elif name == "news20":  # 20 Newsgroups (Usenet posts), TF-IDF bag of words
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        kw = dict(data_home=f"{ROOT}/data/news20", remove=("headers", "footers", "quotes"))
        tr, te = fetch_20newsgroups(subset="train", **kw), fetch_20newsgroups(subset="test", **kw)
        vec = TfidfVectorizer(max_features=3000, sublinear_tf=True, stop_words="english")
        xtr = torch.tensor(vec.fit_transform(tr.data).toarray(), dtype=torch.float32)
        xte = torch.tensor(vec.transform(te.data).toarray(), dtype=torch.float32)
        ytr, yte = torch.tensor(tr.target), torch.tensor(te.target)
    else:
        raise ValueError(name)
    return xtr, ytr, xte, yte


def partition(ytr, n_nodes, m_pub, alpha, rng, per_node=None):
    """Hold out m_pub public (unlabeled) samples, split the rest Dirichlet(alpha) non-IID."""
    idx = rng.permutation(len(ytr))
    pub, rest = idx[:m_pub], idx[m_pub:]
    if per_node is not None:
        rest = rest[: per_node * n_nodes]
    y = ytr[rest].numpy()
    C = int(ytr.max()) + 1
    parts = [[] for _ in range(n_nodes)]
    for c in range(C):
        ids = rest[y == c]
        p = rng.dirichlet(alpha * np.ones(n_nodes))
        cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(ids, cuts)):
            parts[k].extend(chunk.tolist())
    return pub, [np.array(p) for p in parts]


class BatchedCNN:
    """N independent CNNs (conv32-conv64-fc128-fcC) evaluated as one grouped net."""

    def __init__(self, n, in_ch, hw, C, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n, self.in_ch, self.C = n, in_ch, C
        feat = 64 * ((hw // 4) ** 2)

        def init(*shape, fan_in):
            return (torch.randn(*shape, generator=g) * (2.0 / fan_in) ** 0.5).to(DEV).requires_grad_()

        self.params = {
            "c1": init(n * 32, in_ch, 3, 3, fan_in=in_ch * 9),
            "b1": torch.zeros(n * 32, device=DEV, requires_grad=True),
            "c2": init(n * 64, 32, 3, 3, fan_in=32 * 9),
            "b2": torch.zeros(n * 64, device=DEV, requires_grad=True),
            "f1": init(n, feat, 128, fan_in=feat),
            "g1": torch.zeros(n, 1, 128, device=DEV, requires_grad=True),
            "f2": init(n, 128, C, fan_in=128),
            "g2": torch.zeros(n, 1, C, device=DEV, requires_grad=True),
        }

    def forward(self, x):
        """x: (n, B, ch, H, W) -> logits (n, B, C). Node k uses its own weights."""
        n, B = x.shape[:2]
        p = self.params
        h = x.permute(1, 0, 2, 3, 4).reshape(B, n * self.in_ch, *x.shape[3:])
        h = F.max_pool2d(F.relu(F.conv2d(h, p["c1"], p["b1"], padding=1, groups=n)), 2)
        h = F.max_pool2d(F.relu(F.conv2d(h, p["c2"], p["b2"], padding=1, groups=n)), 2)
        h = h.reshape(B, n, -1).permute(1, 0, 2)  # (n, B, feat)
        h = F.relu(torch.bmm(h, p["f1"]) + p["g1"])
        return torch.bmm(h, p["f2"]) + p["g2"]

    def flat(self):
        """Per-node flattened parameter vectors (n, P) — used by parameter-gossip baselines."""
        n = self.n
        out = []
        for k in ["c1", "b1", "c2", "b2"]:
            out.append(self.params[k].detach().reshape(n, -1))
        for k in ["f1", "g1", "f2", "g2"]:
            out.append(self.params[k].detach().reshape(n, -1))
        return torch.cat(out, 1)

    def load_flat(self, V):
        n, o = self.n, 0
        with torch.no_grad():
            for k in ["c1", "b1", "c2", "b2", "f1", "g1", "f2", "g2"]:
                t = self.params[k]
                sz = t.numel() // n
                t.copy_(V[:, o:o + sz].reshape(t.shape))
                o += sz

    @torch.no_grad()
    def predict(self, X, bs=125):
        """Same input X (M, ch, H, W) for every node -> probs (n, M, C)."""
        outs = []
        for s in range(0, len(X), bs):
            xb = X[s:s + bs].to(DEV)
            outs.append(F.softmax(self.forward(xb.unsqueeze(0).expand(self.n, *xb.shape)), -1))
        return torch.cat(outs, 1)


class BatchedMLP(BatchedCNN):
    """N independent MLPs (D-256-C) for bag-of-words inputs; same interface as BatchedCNN."""

    def __init__(self, n, D, C, seed=0, hidden=256):
        g = torch.Generator().manual_seed(seed)
        self.n, self.C = n, C

        def init(*shape, fan_in):
            return (torch.randn(*shape, generator=g) * (2.0 / fan_in) ** 0.5).to(DEV).requires_grad_()

        self.params = {
            "f1": init(n, D, hidden, fan_in=D),
            "g1": torch.zeros(n, 1, hidden, device=DEV, requires_grad=True),
            "f2": init(n, hidden, C, fan_in=hidden),
            "g2": torch.zeros(n, 1, C, device=DEV, requires_grad=True),
        }

    def forward(self, x):
        """x: (n, B, D) -> logits (n, B, C)."""
        p = self.params
        h = F.relu(torch.bmm(x, p["f1"]) + p["g1"])
        return torch.bmm(h, p["f2"]) + p["g2"]

    def _keys(self):
        return ["f1", "g1", "f2", "g2"]

    def flat(self):
        return torch.cat([self.params[k].detach().reshape(self.n, -1) for k in self._keys()], 1)

    def load_flat(self, V):
        o = 0
        with torch.no_grad():
            for k in self._keys():
                t = self.params[k]
                sz = t.numel() // self.n
                t.copy_(V[:, o:o + sz].reshape(t.shape))
                o += sz


class NodeData:
    """Per-node private data kept on device; supports per-node label noise."""

    def __init__(self, xtr, ytr, parts, noisy_nodes, noise_rate, C, rng):
        self.x = [xtr[p].to(DEV) for p in parts]
        ys = []
        for k, p in enumerate(parts):
            y = ytr[p].clone()
            if k in noisy_nodes:
                flip = rng.random(len(y)) < noise_rate
                y[flip] = torch.tensor(rng.integers(0, C, flip.sum()))
            ys.append(y.to(DEV))
        self.y = ys
        self.n = len(parts)

    def batch(self, B, gen):
        xs, ys = [], []
        for k in range(self.n):
            i = torch.randint(0, len(self.y[k]), (B,), generator=gen).to(DEV)
            xs.append(self.x[k][i])
            ys.append(self.y[k][i])
        return torch.stack(xs), torch.stack(ys)


def local_train(net, opt, data, steps, B, gen, active):
    act = torch.tensor(active, device=DEV, dtype=torch.float32)
    for _ in range(steps):
        x, y = data.batch(B, gen)
        logits = net.forward(x)
        loss = (F.cross_entropy(logits.reshape(-1, net.C), y.reshape(-1), reduction="none")
                .view(net.n, B).mean(1) * act).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def distill(net, opt, data, Xpub, targets, steps, B, gen, active, T=2.0, lam=1.0):
    """CE on private data + KL to consensus soft targets on the public set."""
    act = torch.tensor(active, device=DEV, dtype=torch.float32)
    Xpub = Xpub.to(DEV)
    tgt = targets.to(DEV).clamp_min(1e-6)
    tgt = tgt ** (1.0 / T)
    tgt = tgt / tgt.sum(-1, keepdim=True)
    M = len(Xpub)
    for _ in range(steps):
        x, y = data.batch(B, gen)
        i = torch.randint(0, M, (B,), generator=gen).to(DEV)
        xp = Xpub[i]
        xin = torch.cat([x, xp.unsqueeze(0).expand(net.n, *xp.shape)], 1)
        logits = net.forward(xin)
        lp, lq = logits[:, :B], logits[:, B:]
        ce = F.cross_entropy(lp.reshape(-1, net.C), y.reshape(-1), reduction="none").view(net.n, B).mean(1)
        kl = (tgt[:, i] * (tgt[:, i].log() - F.log_softmax(lq / T, -1))).sum(-1).mean(1) * T * T
        loss = ((ce + lam * kl) * act).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(net, xte, yte):
    P = net.predict(xte)
    return (P.argmax(-1).cpu() == yte.unsqueeze(0)).float().mean(1).numpy()
