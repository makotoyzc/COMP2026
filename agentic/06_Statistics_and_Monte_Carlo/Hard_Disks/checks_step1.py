"""Step 1 checks: is the hard-disk sampler correct?

  1. count_overlaps catches a planted overlap, including one across the edge.
  2. Zero overlaps after every sweep, even when crowded (eta = 0.72).
  3. Acceptance falls as the step size delta grows.
  4. Two disks: the sampler is fair. Every allowed placement is equally likely,
     so the separation r has density proportional to r for 1 < r < L/2.

Run:  ~/anaconda3/bin/python3 checks_step1.py   (about 30 s)
Writes figures/step1.png.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from scipy.stats import kstest

from hard_disks import box_side, square_grid, min_image, sweep, count_overlaps

rng = np.random.default_rng(1)
N = 256

# ---- Check 1: test the test ------------------------------------------------
L = 5.0
y = 2.0
touching_across_edge = np.array([[0.2, y], [L - 0.7, y]])  # distance 0.9
apart_across_edge = np.array([[0.2, y], [L - 0.9, y]])     # distance 1.1
assert count_overlaps(touching_across_edge, L) == 1
assert count_overlaps(apart_across_edge, L) == 0
print("check 1  planted overlap found, near miss not flagged      PASS")

# ---- Check 2: no overlaps ever, at a crowded density -----------------------
snapshots = {}
for eta, delta in [(0.30, 0.5), (0.72, 0.05)]:
    L = box_side(N, eta)
    x = square_grid(N, L)
    assert count_overlaps(x, L) == 0
    for s in range(300):
        sweep(x, L, delta, rng)
        assert count_overlaps(x, L) == 0, f"overlap at eta={eta}, sweep {s}"
    snapshots[eta] = (x.copy(), L)
print("check 2  0 overlaps in 300 sweeps at eta = 0.30 and 0.72    PASS")

# ---- Check 3: acceptance vs delta -------------------------------------------
deltas = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
acceptance = {}
for eta in [0.30, 0.60, 0.72]:
    L = box_side(N, eta)
    x = square_grid(N, L)
    rates = []
    for delta in deltas:
        for _ in range(30):  # let it settle at this delta
            sweep(x, L, delta, rng)
        rates.append(sum(sweep(x, L, delta, rng) for _ in range(20)) / (20 * N))
    rates = np.array(rates)
    acceptance[eta] = rates
    assert np.all(np.diff(rates) < 0), f"acceptance not falling at eta={eta}: {rates}"
    # delta where acceptance crosses 40%, interpolated on a log scale
    d40 = np.exp(np.interp(0.4, rates[::-1], np.log(deltas[::-1])))
    print(f"         eta = {eta:.2f}: acceptance {np.round(rates, 2)},  40% at delta = {d40:.3f}")
print("check 3  acceptance falls monotonically with delta          PASS")

# ---- Check 4: two disks, the separation must have density ~ r ------------------
# For 1 < r < L/2 the whole ring of radius r fits inside the periodic cell, so
# the allowed area at separation r is 2 pi r dr. Conditioned on r < L/2,
#     CDF(r) = (r^2 - 1) / ((L/2)^2 - 1).
L2 = 4.0
cdf = lambda r: (r**2 - 1) / ((L2 / 2) ** 2 - 1)
two_disk_r = {}
for delta, thin in [(L2 / 2, 1), (0.3, 20)]:  # big steps, then small steps
    x = np.array([[1.0, 1.0], [3.0, 3.0]])
    r = []
    for s in range(20000 * thin):
        sweep(x, L2, delta, rng)
        if s % thin == 0:
            r.append(np.hypot(*min_image(x[0] - x[1], L2)))
    r = np.array(r)
    r = r[r < L2 / 2]
    p = kstest(r, cdf).pvalue
    two_disk_r[delta] = r
    print(f"         delta = {delta:.1f}: {len(r)} samples, KS p-value = {p:.2f}")
    assert p > 0.01, f"two-disk separation does not follow density ~ r (delta={delta})"
print("check 4  two-disk separation follows density ~ r            PASS")


# ---- Figure -------------------------------------------------------------------
def draw_disks(ax, x, L):
    shifts = [(sx, sy) for sx in (-L, 0, L) for sy in (-L, 0, L)]
    circles = [Circle((cx + sx, cy + sy), 0.5) for cx, cy in x for sx, sy in shifts]
    ax.add_collection(PatchCollection(circles, edgecolor="k", linewidth=0.4))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")


fig, axs = plt.subplots(2, 2, figsize=(10, 9))
for ax, eta in zip(axs[0], [0.30, 0.72]):
    draw_disks(ax, *snapshots[eta])
    ax.set_title(f"eta = {eta:.2f}, after 300 sweeps")

ax = axs[1, 0]
for eta, rates in acceptance.items():
    ax.semilogx(deltas, rates, "o-", label=f"eta = {eta:.2f}")
ax.axhline(0.4, color="gray", linestyle="--")
ax.set_xlabel("step size delta")
ax.set_ylabel("acceptance")
ax.set_title("Acceptance vs step size (N = 256)")
ax.legend()

ax = axs[1, 1]
bins = np.linspace(1, L2 / 2, 31)
for delta, r in two_disk_r.items():
    ax.hist(r, bins=bins, density=True, histtype="step", label=f"delta = {delta:.1f}")
rr = np.linspace(1, L2 / 2, 100)
ax.plot(rr, 2 * rr / ((L2 / 2) ** 2 - 1), "k--", label="exact, density ~ r")
ax.set_xlabel("separation r")
ax.set_ylabel("probability density")
ax.set_title(f"Two disks in a box of side {L2:.0f}")
ax.legend()

fig.tight_layout()
fig.savefig("figures/step1.png", dpi=150)
print("wrote figures/step1.png")
