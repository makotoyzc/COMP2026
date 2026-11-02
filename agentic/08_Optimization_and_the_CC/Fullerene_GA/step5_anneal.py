"""Step 5: simulated annealing (MD, anneal.py) on C20 and C60, all runs in parallel.

C20: does annealing find the bowl (the lowest C20 isomer) more cheaply than random
     multistart (Step 4)? Cost = energy+force calls, MD steps included.
C60: how close does annealing get to E_ref, and what does it make instead?

Usage:  python3 step5_anneal.py   (needs step4_results.npz)
        -> step5_results.npz, step5_anneal.png, step5_anneal.gif, viewer_step5.html
"""
import parallel  # first: one BLAS thread per process

import numpy as np
import matplotlib.pyplot as plt

import carbon_tb as tb
import structures as st
import anneal as an
import viz

T0 = 4000.0                      # K, well above where small carbon clusters melt
RUNS = {20: 36, 60: 18}          # independent anneals per schedule
LENGTHS = {20: (2000, 6000, 20000, 60000), 60: (10000, 30000, 100000, 300000)}


def one_anneal(task):
    N, steps, seed, record = task
    rng = np.random.default_rng(seed)
    radius = 1.25 * N ** (1 / 3)
    R0, _, info0 = st.relax(st.random_cluster(N, rng, radius), fmax=1e-2, maxiter=4000)
    R, Eb, calls, info = an.anneal(R0, T0, steps, rng, R_wall=1.5 * radius,
                                   record_every=record)
    nr, _ = st.rings(R, sizes=(5, 6, 7))
    return dict(N=N, steps=steps, Eb=Eb, calls=calls + info0["nfev"], cls=st.classify(R),
                rings=nr, R=R, path=info["path"], Tkin=info["Tkin"])


if __name__ == "__main__":
    s4 = np.load("step4_results.npz", allow_pickle=True)
    E_ref = float(s4["E_ref"])
    E_bowl = float(s4["E20"][1])
    k20 = list(s4["sizes"]).index(20)
    Eb20, nfev20 = s4["Eb"][k20], s4["nfev"][k20]
    p_ms = np.mean(Eb20 < E_bowl + 1e-3)
    cost_ms = nfev20.mean() / p_ms if p_ms > 0 else np.inf
    print(f"multistart, C20: bowl found by {p_ms:.1%} of starts, {nfev20.mean():.0f} calls each"
          f" -> {cost_ms:.0f} calls per success")

    tasks = [(N, L, 10**6 * N + 1000 * k + r, 200 if (N == 60 and L == 100000 and r == 0) else 0)
             for N in (20, 60) for k, L in enumerate(LENGTHS[N]) for r in range(RUNS[N])]
    tasks.sort(key=lambda t: -t[1])            # longest first, so the pool stays busy
    out = parallel.run(one_anneal, tasks, "annealing", chunksize=1)

    summary = {}
    for N in (20, 60):
        target = E_bowl if N == 20 else E_ref
        print(f"\n===== C{N}, T0 = {T0:.0f} K, {RUNS[N]} runs per schedule =====")
        print(f"{'MD steps':>9s} {'calls':>8s} {'best':>9s} {'median':>9s} {'success':>8s}"
              f" {'calls/success':>14s}   classes")
        for L in LENGTHS[N]:
            res = [o for o in out if o["N"] == N and o["steps"] == L]
            Eb = np.array([o["Eb"] for o in res])
            ok = Eb < target + 1e-3
            calls = np.mean([o["calls"] for o in res])
            cls = {c: [o["cls"] for o in res].count(c) for c in set(o["cls"] for o in res)}
            per = calls / ok.mean() if ok.any() else np.inf
            print(f"{L:9d} {calls:8.0f} {Eb.min():+9.4f} {np.median(Eb):+9.4f} {ok.mean():8.1%}"
                  f" {per:14.0f}   {cls}")
            summary[(N, L)] = dict(Eb=Eb, calls=calls, ok=ok.mean(),
                                   best=res[int(Eb.argmin())])
        if N == 60:
            b = min((o for o in out if o["N"] == 60), key=lambda o: o["Eb"])
            print(f"best C60 anneal ({b['steps']} steps): {b['Eb'] - E_ref:+.4f} eV/atom above E_ref;"
                  f" {st.describe(b['R'])}")

    np.savez("step5_results.npz", T0=T0,
             N=np.array([o["N"] for o in out]), steps=np.array([o["steps"] for o in out]),
             Eb=np.array([o["Eb"] for o in out]), calls=np.array([o["calls"] for o in out]),
             cls=np.array([o["cls"] for o in out]),
             R=np.array([o["R"] for o in out], dtype=object), allow_pickle=True)

    # Figure: energy distribution per schedule
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, N, target, lab in ((axs[0], 20, E_bowl, "C$_{20}$ bowl"), (axs[1], 60, E_ref, "C$_{60}$")):
        for k, L in enumerate(LENGTHS[N]):
            Eb = summary[(N, L)]["Eb"]
            ax.plot(np.full(len(Eb), summary[(N, L)]["calls"]), Eb, "o", ms=4, alpha=0.5,
                    color=f"C{k}", label=f"{L} MD steps")
        if N == 20:
            ax.plot(nfev20[:RUNS[20]], Eb20[:RUNS[20]], "k.", ms=3, alpha=0.5, label="multistart (one start each)")
        ax.axhline(target, color="C3", ls="--", lw=0.8)
        ax.text(ax.get_xlim()[0] if N == 60 else 30, target, lab, color="C3", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("Energy+force calls per run")
        ax.set_ylabel("Final binding energy (eV/atom)")
        ax.set_title(f"C$_{{{N}}}$: annealing from {T0:.0f} K")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("step5_anneal.png", dpi=150)
    plt.close(fig)

    show = next(o for o in out if o["path"])
    Es = [tb.binding(tb.energy(p), 60) for p in show["path"]] + [show["Eb"]]
    path = show["path"] + [show["R"]]
    viz.movie(path, Es, "step5_anneal.gif", every=max(1, len(path) // 80))
    runs = [("C60 anneal 100k steps: the MD path", path, Es)]
    for N in (20, 60):
        for L in LENGTHS[N]:
            b = summary[(N, L)]["best"]
            runs.append((f"C{N} best of {RUNS[N]}, {L} MD steps", [b["R"]], [b["Eb"]]))
    viz.viewer(runs, "viewer_step5.html", "snapshot")
    print("saved step5_anneal.png, step5_anneal.gif, viewer_step5.html")
