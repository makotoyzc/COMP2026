"""Step 2 checks: does the measured pressure agree with the exact answer?

  A. g(r) = 0 for r < 1: not one pair closer than a diameter.
  B. g(r) -> 1 far away: distant disks don't know about each other.
  C. Equilibrated: Z from the first and second half of the run agree.
  D. Z matches the virial series at eta <= 0.3, within 3 error bars.

The virial series is the exact low-density expansion of the pressure,

    Z = 1 + 2 eta + b3 eta^2 + b4 eta^3 + ...

Its coefficients are known numerically to many digits. Seven terms are used
here, so the terms left out add less than 0.001 at eta = 0.3.

Run:  ~/anaconda3/bin/python3 checks_step2.py   (about 2.5 min)
Writes results_step2.npz and figures/step2.png.
"""
import numpy as np
import matplotlib.pyplot as plt

from pressure import run

# Z = 1 + sum_k b[k] eta^k, hard-disk virial coefficients B_n / B_2^(n-1) times 2^(n-1)
VIRIAL = [2.0, 3.12802, 4.25785, 5.33689, 6.36486, 7.37818, 8.41626]


def Z_virial(eta, n_terms=len(VIRIAL)):
    return 1 + sum(b * eta ** (k + 1) for k, b in enumerate(VIRIAL[:n_terms]))


def Z_henderson(eta):
    """Henderson's closed-form fit for the hard-disk fluid, a reference curve only."""
    return (1 + eta**2 / 8) / (1 - eta) ** 2


rng = np.random.default_rng(2)
etas = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
results = []
for eta in etas:
    r = run(eta, rng)
    results.append(r)
    print(f"eta = {eta:.2f}  delta = {r['delta']:.3f}  acceptance = {r['acceptance']:.2f}  "
          f"Z = {r['Z']:.4f} +/- {r['dZ']:.4f}  virial = {Z_virial(eta):.4f}")

print()
for r in results:
    centers = 0.5 * (r["edges"][1:] + r["edges"][:-1])

    # A: no pairs inside one diameter
    assert r["counts"][centers < 1].sum() == 0

    # B: g -> 1 over the outermost 1.5 diameters
    far = centers > r["L"] / 2 - 1.5
    assert abs(r["g"][far].mean() - 1) < 0.02, f"g far = {r['g'][far].mean():.3f} at eta = {r['eta']}"

    # C: first half vs second half of the blocks
    a, b = np.split(r["Z_blocks"], 2)
    diff = a.mean() - b.mean()
    err = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    assert abs(diff) < 3 * err, f"not equilibrated at eta = {r['eta']}: halves differ by {diff:.4f} +/- {err:.4f}"

    # D: virial series at low density
    if r["eta"] <= 0.3:
        assert abs(r["Z"] - Z_virial(r["eta"])) < 3 * r["dZ"], f"Z off the virial series at eta = {r['eta']}"
print("check A  no pairs closer than one diameter, at every eta    PASS")
print("check B  g(r) -> 1 at large r, at every eta                 PASS")
print("check C  first and second halves agree, at every eta        PASS")
print("check D  Z matches the virial series for eta <= 0.3         PASS")

np.savez("results_step2.npz", etas=etas, Z=[r["Z"] for r in results], dZ=[r["dZ"] for r in results])

# ---- Figure -------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axs[0]
for r in results:
    if r["eta"] in (0.10, 0.30, 0.60):
        centers = 0.5 * (r["edges"][1:] + r["edges"][:-1])
        ax.plot(centers, r["g"], label=f"eta = {r['eta']:.2f}")
ax.axhline(1, color="gray", linestyle="--")
ax.set_xlim(0, 4)
ax.set_xlabel("separation r (diameters)")
ax.set_ylabel("g(r)")
ax.set_title("Pair distribution")
ax.legend()

eta_fine = np.linspace(0, 0.65, 200)
eta_arr = np.array(etas)
Z = np.array([r["Z"] for r in results])
dZ = np.array([r["dZ"] for r in results])

ax = axs[1]
ax.plot(eta_fine, Z_virial(eta_fine, 2), ":", label="virial, 2 terms")
ax.plot(eta_fine, Z_virial(eta_fine), "-", label="virial, 7 terms")
ax.plot(eta_fine, Z_henderson(eta_fine), "--", label="Henderson")
ax.errorbar(eta_arr, Z, dZ, fmt="ko", label="Monte Carlo, N = 256")
ax.set_xlabel("packing fraction eta")
ax.set_ylabel("Z = PA / NkT")
ax.set_title("Equation of state")
ax.legend()

ax = axs[2]
ax.errorbar(eta_arr, Z - Z_virial(eta_arr), dZ, fmt="ko", label="Monte Carlo - virial")
ax.plot(eta_fine, Z_henderson(eta_fine) - Z_virial(eta_fine), "--", label="Henderson - virial")
ax.axhline(0, color="gray")
ax.set_xlabel("packing fraction eta")
ax.set_ylabel("Z - Z_virial (7 terms)")
ax.set_title("Residual")
ax.legend()

fig.tight_layout()
fig.savefig("figures/step2.png", dpi=150)
print("wrote figures/step2.png")
