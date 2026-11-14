"""One training run, used by every step.

split() turns a data file into train and test sets by chain (chains 0-3 train,
4-5 test), never by shuffling snapshots. fit() trains a fresh CNN with Adam and
binary cross-entropy and returns test accuracy and the test outputs.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn

import data
from model import CNN

# A 1,265-weight net trains in ~5.5 s on the Mac GPU (MPS) vs ~56 s on one CPU core (L = 32).
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Throughput measured 11/13 (L = 32 Ising, trainings per second): 1 GPU process
# 0.18; 2-4 GPU processes 0.21-0.22 (GPU saturated); 12 CPU processes 0.10;
# 3 GPU + 12 CPU processes 0.28. So run_parallel uses both, with ~35% of jobs on CPU.
N_GPU, N_CPU, CPU_SHARE = 3, 12, 0.35


def _set_device(dev):
    global DEVICE
    DEVICE = dev if dev == "cpu" or torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(1)


def run_parallel(fn, jobs, n_gpu=N_GPU, n_cpu=N_CPU, cpu_share=CPU_SHARE):
    """Run fn(job) for every job on GPU and CPU workers at once; results in job order.

    Which device a job gets depends only on its index (evenly spread), not on
    which worker is free first, so a rerun gives the same numbers. fn must be a
    module-level function, and the calling script needs `if __name__ == "__main__"`.
    """
    on_cpu = [int((i + 1) * cpu_share) > int(i * cpu_share) for i in range(len(jobs))]
    with ProcessPoolExecutor(n_gpu, initializer=_set_device, initargs=("mps",)) as gpu, \
         ProcessPoolExecutor(n_cpu, initializer=_set_device, initargs=("cpu",)) as cpu:
        futures = [(cpu if c else gpu).submit(fn, j) for j, c in zip(jobs, on_cpu)]
        return [f.result() for f in futures]


def split(d):
    """Arrays for training and testing, flattened over (T, chain, snapshot).

    Returns dict with x (snapshots), T (temperature of each), and the
    per-snapshot measurements absm, e (and nv for XY), for "train" and "test".
    """
    nT = len(d["Ts"])
    out = {}
    for name, chains in (("train", data.TRAIN), ("test", data.TEST)):
        s = d["s"][:, chains]
        out[name] = dict(x=s.reshape(-1, *s.shape[-2:]),
                         T=np.repeat(d["Ts"], len(chains) * s.shape[2]))
        for k in ("absm", "e", "nv"):
            if k in d:
                out[name][k] = d[k][:, chains].reshape(-1)
        assert out[name]["T"].size == nT * len(chains) * s.shape[2]
    return out


def ising_input(s):
    """Spins as a 1-channel float image."""
    return torch.as_tensor(s, dtype=torch.float32)[:, None]


def xy_input(theta):
    """Angles as a 2-channel image (cos theta, sin theta): no fake edge at 2 pi."""
    th = torch.as_tensor(theta, dtype=torch.float32)
    return torch.stack([torch.cos(th), torch.sin(th)], 1)


def rotate(x, gen):
    """Augmentation for XY: global rotation theta -> theta + alpha, alpha random per snapshot."""
    a = (torch.rand(x.shape[0], generator=gen) * 2 * np.pi).to(x.device)[:, None, None]
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]], 1)


def flip(x, gen):
    """Augmentation for Ising: global spin flip s -> -s, chosen per snapshot."""
    sign = torch.randint(0, 2, (x.shape[0], 1, 1, 1), generator=gen) * 2 - 1
    return x * sign.to(x.device)


def fit(x_tr, y_tr, x_te, y_te, seed, layers=((8, 3), (16, 3)), augment=None,
        epochs=15, batch=256, lr=3e-3):
    """Train from scratch. x are float tensors (N, C, L, L); y are 0/1 arrays.

    Returns dict: acc (test accuracy), p (test outputs, sigmoid), net.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = CNN(x_tr.shape[1], layers).to(DEVICE)
    x_tr, x_te = x_tr.to(DEVICE), x_te.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32, device=DEVICE)
    for _ in range(epochs):
        order = torch.randperm(len(x_tr), generator=gen).to(DEVICE)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = x_tr[idx] if augment is None else augment(x_tr[idx], gen)
            opt.zero_grad()
            loss_fn(net(xb), y_tr[idx]).backward()
            opt.step()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(net(x_te[i:i + 2048])) for i in range(0, len(x_te), 2048)])
        p = p.cpu().numpy()
    return dict(acc=((p > 0.5) == np.asarray(y_te, bool)).mean(), p=p, net=net)


def logistic(f_tr, y_tr, f_te, y_te):
    """Baseline: logistic regression on a few hand-made numbers per snapshot
    (e.g. |m| alone). f are arrays (N,) or (N, k). Returns test accuracy.

    Written in torch instead of sklearn: importing sklearn before torch makes
    torch segfault on this Mac (two OpenMP runtimes).
    """
    f_tr = torch.as_tensor(np.asarray(f_tr, float).reshape(len(f_tr), -1))
    f_te = torch.as_tensor(np.asarray(f_te, float).reshape(len(f_te), -1))
    mu, sd = f_tr.mean(0), f_tr.std(0)
    f_tr, f_te = (f_tr - mu) / sd, (f_te - mu) / sd
    lin = nn.Linear(f_tr.shape[1], 1).double()
    y = torch.as_tensor(np.asarray(y_tr, float))
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(f_tr).squeeze(1), y)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = lin(f_te).squeeze(1).numpy() > 0
    return (pred == np.asarray(y_te, bool)).mean()


def middle_peak(cuts, acc):
    """Middle peak of a confusion (W) curve: the interior local maximum with
    the largest prominence. Returns (cut at the peak, parabola estimate
    through it and its neighbors, prominence), all as label boundaries.

    Label = (T < T') with T' on the grid puts the boundary halfway between
    T' - h and T', so both temperatures are reported as T' - h/2.
    """
    h = cuts[1] - cuts[0]
    peaks, props = find_peaks(acc, prominence=0)
    i = peaks[np.argmax(props["prominences"])]
    y0, y1, y2 = acc[i - 1:i + 2]
    sub = cuts[i] + h * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return cuts[i] - h / 2, sub - h / 2, props["prominences"].max()


def binned_r2(f, p, n_bins=40):
    """How much of the output's spread one number f explains: R^2 of p
    against its mean in narrow f bins. 1 means output is a function of f alone."""
    edges = np.unique(np.quantile(f, np.linspace(0, 1, n_bins + 1)))  # f may be discrete
    n_bins = len(edges) - 1
    b = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_bins - 1)
    means = np.bincount(b, p, n_bins) / np.bincount(b, minlength=n_bins)
    return 1 - ((p - means[b]) ** 2).sum() / ((p - p.mean()) ** 2).sum()


def crossing(Ts, p_mean):
    """Temperature where the T-averaged output first falls through 0.5 (linear interpolation)."""
    i = np.flatnonzero((p_mean[:-1] >= 0.5) & (p_mean[1:] < 0.5))[0]
    return Ts[i] + (p_mean[i] - 0.5) / (p_mean[i] - p_mean[i + 1]) * (Ts[i + 1] - Ts[i])
