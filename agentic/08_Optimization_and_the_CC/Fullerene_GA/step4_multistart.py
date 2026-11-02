"""Step 4: random multistart. Many random starts, each relaxed downhill; keep the lowest.

Part A  the three C20 isomers of Deaven & Ho (ring, bowl, cage), relaxed directly,
        compared with their energies (PRL 75, 288, p. 290).
Part B  1000 random starts for each N = 10..24.
Part C  720 random starts for C60, the baseline the GA has to beat.

Usage:  python3 step4_multistart.py   -> step4_*.png, step4_results.npz, viewer_step4.html
"""
import parallel  # first: one BLAS thread per process

import numpy as np
import matplotlib.pyplot as plt

import carbon_tb as tb
import structures as st
import viz

SIZES = range(10, 25)
STARTS = 1000
STARTS_C60 = 720
FMAX = 1e-3                      # eV/A; enough for energies to 1e-5 eV/atom
DH_C20 = dict(ring=-8.503, bowl=-8.671, cage=-8.613)   # Deaven & Ho, their energy zero
CLASSES = ["chain", "ring", "flat", "bowl", "cage", "sp3", "fragments"]


def start_radius(N):
    """Random atoms go in a sphere about as big as a compact N-atom cluster."""
    return 1.25 * N ** (1 / 3)


def one_start(task):
    """Relax one random start. task = (N, seed)."""
    N, seed = task
    rng = np.random.default_rng(seed)
    R, E, info = st.relax(st.random_cluster(N, rng, start_radius(N)), fmax=FMAX, maxiter=4000)
    return dict(N=N, Eb=tb.binding(E, N), cls=st.classify(R), nfev=info["nfev"],
                converged=info["converged"], R=R)


def summarize(N, res):
    Eb = np.array([r["Eb"] for r in res])
    best = int(Eb.argmin())
    hit = Eb < Eb[best] + 1e-3
    half = len(res) // 2
    distinct = len(np.unique(np.round(Eb, 3)))
    nfev = np.mean([r["nfev"] for r in res])
    return dict(N=N, best=best, Eb=Eb, hit=hit.mean(), distinct=distinct, nfev=nfev,
                half_gap=Eb[:half].min() - Eb.min(),
                conv=np.mean([r["converged"] for r in res]),
                cls=[r["cls"] for r in res])


if __name__ == "__main__":
    ref = np.load("c60_ref.npy")
    E_ref = tb.binding(tb.energy(ref), 60)

    # ---------------- Part A: C20 isomers ----------------
    print("===== A. C20 isomers vs Deaven & Ho =====")
    rng = np.random.default_rng(20)
    iso = {"ring": st.ring(20) + rng.normal(0, 0.05, (20, 3)),   # kick breaks the symmetry
           "bowl": st.c20_bowl(ref), "cage": st.c20_cage()}
    E20 = {}
    for name, R0 in iso.items():
        R, E, info = st.relax(R0, fmax=1e-6)
        iso[name], E20[name] = R, tb.binding(E, 20)
        print(f"{name}: {st.describe(R)}")
    for name in iso:
        print(f"{name:4s} - bowl: ours {E20[name] - E20['bowl']:+.4f},"
              f"  D&H {DH_C20[name] - DH_C20['bowl']:+.4f} eV/atom;"
              f"  energy-zero offset D&H - E_tot/N = {DH_C20[name] - (E20[name] + tb.E_ATOM):+.4f}")
    i, j, r = st.bonds(iso["ring"])
    print(f"ring bonds alternate {r.min():.3f} / {r.max():.3f} A (Peierls dimerization)")
    print(f"C20 bowl - C60: ours {E20['bowl'] - E_ref:+.4f}, D&H Fig. 1 ~ {-8.671 + 9.39:+.2f} eV/atom")

    # ---------------- Part B: N = 10..24 ----------------
    print(f"\n===== B. random multistart, {STARTS} starts per size =====")
    tasks = [(N, 1000 * N + k) for N in SIZES for k in range(STARTS)]
    out = parallel.run(one_start, tasks, "multistart N = 10..24", chunksize=8)
    summ = {}
    print(f"{'N':>3s} {'best E_b':>9s} {'class':>9s} {'hit':>6s} {'distinct':>8s} "
          f"{'calls':>6s} {'1st half':>9s} {'conv':>5s}   classes of all minima")
    for N in SIZES:
        res = [o for o in out if o["N"] == N]
        s = summ[N] = summarize(N, res)
        counts = {c: s["cls"].count(c) for c in CLASSES if s["cls"].count(c)}
        print(f"{N:3d} {s['Eb'].min():+9.4f} {res[s['best']]['cls']:>9s} {s['hit']:6.1%} "
              f"{s['distinct']:8d} {s['nfev']:6.0f} {s['half_gap']:+9.4f} {s['conv']:5.0%}   {counts}")
        summ[N]["R"] = res[s["best"]]["R"]
    s20 = summ[20]
    for name in iso:
        frac = np.mean(np.abs(s20["Eb"] - E20[name]) < 1e-3)
        print(f"C20 {name}: found by {frac:.1%} of the starts")

    # ---------------- Part C: C60 ----------------
    print(f"\n===== C. random multistart on C60, {STARTS_C60} starts =====")
    out60 = parallel.run(one_start, [(60, 600000 + k) for k in range(STARTS_C60)],
                         "multistart C60", chunksize=2)
    s60 = summarize(60, out60)
    counts = {c: s60["cls"].count(c) for c in CLASSES if s60["cls"].count(c)}
    b60 = out60[s60["best"]]
    print(f"best of {STARTS_C60}: E_b = {s60['Eb'].min():+.4f}, {s60['Eb'].min() - E_ref:+.4f} above E_ref;"
          f" {st.describe(b60['R'])}")
    print(f"median {np.median(s60['Eb']):+.4f}; mean calls per start {s60['nfev']:.0f};"
          f" converged {s60['conv']:.0%}; classes {counts}")
    print(f"starts within 0.01 eV/atom of E_ref: {np.sum(s60['Eb'] < E_ref + 0.01)}")

    np.savez("step4_results.npz",
             sizes=np.array(list(SIZES)), Eb=np.array([summ[N]["Eb"] for N in SIZES]),
             nfev=np.array([[o["nfev"] for o in out if o["N"] == N] for N in SIZES]),
             best_R=np.array([summ[N]["R"] for N in SIZES], dtype=object),
             cls=np.array([summ[N]["cls"] for N in SIZES]),
             Eb60=s60["Eb"], nfev60=np.array([o["nfev"] for o in out60]),
             cls60=np.array(s60["cls"]), best60=b60["R"],
             E20=np.array([E20[k] for k in ("ring", "bowl", "cage")]), E_ref=E_ref,
             allow_pickle=True)

    # ---------------- Figures ----------------
    fig = plt.figure(figsize=(13, 8))
    for k, N in enumerate(SIZES):
        ax = fig.add_subplot(3, 5, k + 1, projection="3d")
        res = [o for o in out if o["N"] == N]
        viz.draw(ax, summ[N]["R"], f"C$_{{{N}}}$ {res[summ[N]['best']]['cls']}\n"
                 f"E_b = {summ[N]['Eb'].min():+.3f}, found by {summ[N]['hit']:.1%}")
    fig.tight_layout()
    fig.savefig("step4_gallery.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    ax.hist(s20["Eb"], bins=80, color="0.6")
    for name, c in zip(iso, ("C2", "C0", "C3")):
        ax.axvline(E20[name], color=c, ls="--", label=f"C$_{{20}}$ {name}")
    ax.set_xlabel("Binding energy of the local minimum (eV/atom)")
    ax.set_ylabel("Random starts")
    ax.set_title(f"C$_{{20}}$: where {STARTS} relaxations end")
    ax.legend()
    ax = axs[1]
    frac = np.array([[summ[N]["cls"].count(c) / STARTS for c in CLASSES] for N in SIZES])
    bottom = np.zeros(len(SIZES))
    for k, c in enumerate(CLASSES):
        ax.bar(list(SIZES), frac[:, k], bottom=bottom, label=c, color=f"C{k}")
        bottom += frac[:, k]
    ax.set_xlabel("N")
    ax.set_ylabel("Fraction of local minima")
    ax.set_title("What random relaxations produce")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax = axs[2]
    ax.hist(s60["Eb"], bins=60, color="0.6")
    ax.axvline(E_ref, color="C3", ls="--", label="C$_{60}$ (E_ref)")
    ax.set_xlabel("Binding energy of the local minimum (eV/atom)")
    ax.set_ylabel("Random starts")
    ax.set_title(f"C$_{{60}}$: {STARTS_C60} random relaxations")
    ax.legend()
    fig.tight_layout()
    fig.savefig("step4_landscape.png", dpi=150)
    plt.close(fig)
    print("saved step4_gallery.png, step4_landscape.png")

    runs = [(f"C{N} best of {STARTS}", [summ[N]["R"]], [summ[N]["Eb"].min()]) for N in SIZES]
    runs += [(f"C20 {name} (built)", [iso[name]], [E20[name]]) for name in iso]
    runs += [(f"C60 best of {STARTS_C60} random", [b60["R"]], [s60["Eb"].min()]),
             ("C60 reference", [ref], [E_ref])]
    viz.viewer(runs, "viewer_step4.html", "structure")
