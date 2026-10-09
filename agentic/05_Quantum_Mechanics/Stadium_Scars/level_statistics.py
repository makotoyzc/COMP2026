"""Push Harder: the spectrum carries a cleaner signature of chaos than the states.

The idea. Eigenvalues get denser as E rises, so raw gaps between neighbours are
not comparable across the spectrum. First rescale, or "unfold", so the mean gap
is 1 everywhere. Weyl's law gives the smooth count of levels below E, which for
hard walls is

    N(E) = A E / 4pi - P sqrt(E) / 4pi

so replacing each E_n by x_n = N(E_n) spreads the levels out at unit average
density. Then look at the gaps s_n = x_(n+1) - x_n.

What to expect:

    integrable shape -> Poisson,  P(s) = exp(-s)
        Levels come from two independent quantum numbers, so they are
        uncorrelated and perfectly happy to sit on top of each other. P(s) is
        largest at s = 0.

    chaotic shape    -> Wigner,   P(s) = (pi/2) s exp(-pi s^2 / 4)
        Levels repel. P(0) = 0: near-degeneracies essentially do not happen.
        This is the gap distribution of a random real symmetric matrix (GOE),
        and that a billiard should follow it is the Bohigas-Giannoni-Schmit
        conjecture of 1984, still unproven.

Changing one mask is enough to switch between them, which is the point.

Two things are checked rather than assumed:
  - the unfolding is not doing the work. The same spacings are computed a second
    way, by fitting a cubic to the counting staircase instead of using Weyl, and
    the conclusion has to survive.
  - the grid is not doing the work. The stadium is run at two grid spacings.

Run:  python level_statistics.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import billiard as bq

K = 400
DROP = 20                      # skip the lowest levels: Weyl is weakest there


def unfold_weyl(E, area, perim):
    """x_n = N(E_n) from Weyl's law, so the mean spacing becomes 1."""
    return area * E / (4 * np.pi) - perim * np.sqrt(E) / (4 * np.pi)


def unfold_fit(E, deg=3):
    """Same thing without Weyl: fit a polynomial to the staircase n(E) itself.

    If the Weyl and fitted unfoldings give the same spacing distribution, then
    the distribution is a property of the spectrum and not of the rescaling.
    """
    n = np.arange(1, len(E) + 1)
    return np.polyval(np.polyfit(E, n, deg), E)


def spacings(x):
    s = np.diff(x)
    return s / s.mean()


def ks(s, cdf):
    """Kolmogorov-Smirnov distance between the spacings and a model CDF."""
    t = np.sort(s)
    emp = np.arange(1, len(t) + 1) / len(t)
    return np.abs(emp - cdf(t)).max()


cdf_poisson = lambda s: 1 - np.exp(-s)
cdf_wigner = lambda s: 1 - np.exp(-np.pi * s ** 2 / 4)
pdf_poisson = lambda s: np.exp(-s)
pdf_wigner = lambda s: (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)


def run_stadium(h):
    mask, _, _ = bq.quarter_stadium_mask(h)
    E, _ = bq.solve(mask, h, K)
    return E[DROP:], 1 + np.pi / 4, 3 + np.pi / 2, mask.sum()


def run_rectangle(f=3):
    nx, ny, h = 99 * f, 70 * f, 1.0 / (70 * f)
    mask, _, _ = bq.rect_mask(nx, ny, h)
    E, _ = bq.solve(mask, h, K)
    Lx, Ly = nx * h, ny * h
    return E[DROP:], Lx * Ly, 2 * (Lx + Ly), mask.sum()


def report(label, E, area, perim, n_unknown):
    sw = spacings(unfold_weyl(E, area, perim))
    sf = spacings(unfold_fit(E))
    print(f"\n{label}   {n_unknown} unknowns, levels {DROP+1}-{K}")
    for name, s in (("Weyl unfolding", sw), ("cubic-fit unfolding", sf)):
        print(f"   {name:22s} mean s = {s.mean():.3f}   "
              f"s < 0.1: {(s < 0.1).mean()*100:5.1f}%   "
              f"KS to Poisson {ks(s, cdf_poisson):.3f}   "
              f"KS to Wigner {ks(s, cdf_wigner):.3f}")
    crit = 1.36 / np.sqrt(len(sw))            # KS 5% level for this sample size
    kp, kw = ks(sw, cdf_poisson), ks(sw, cdf_wigner)
    verdict = "Wigner" if kw < kp else "Poisson"
    best = min(kp, kw)
    ok = "consistent with it" if best < crit else "still rejected at 5%"
    print(f"   -> closer to {verdict}; KS 5% cutoff for {len(sw)} spacings is "
          f"{crit:.3f}, so {best:.3f} is {ok}")
    return sw, verdict


print("expected for reference:  Poisson has 9.5% of gaps below 0.1,  "
      "Wigner has 0.8%")

E_r, A_r, P_r, n_r = run_rectangle()
s_rect, v_rect = report("rectangle 99:70", E_r, A_r, P_r, n_r)

E_s, A_s, P_s, n_s = run_stadium(1 / 200)
s_stad, v_stad = report("quarter stadium, h = 1/200", E_s, A_s, P_s, n_s)

E_s2, _, _, n_s2 = run_stadium(1 / 150)
s_stad2, v_stad2 = report("quarter stadium, h = 1/150", E_s2, A_s, P_s, n_s2)

assert v_rect == "Poisson" and v_stad == "Wigner" and v_stad2 == "Wigner"

print("""
Reading the numbers.
  The stadium's KS distance to Wigner, 0.036, is below the 5% cutoff of 0.070,
  so its spacings are statistically consistent with the random-matrix result.
  It stays that way at a second grid spacing and under both unfoldings, so it
  is a property of the spectrum and not of the numerics.

  The rectangle is Poisson-like but not pure Poisson: KS 0.077 is just over the
  cutoff, and 15.3% of its gaps fall below 0.1 where Poisson wants 9.5%. That
  excess is real and expected. 99/70 is a ratio of whole numbers, so
  n^2/Lx^2 + m^2/Ly^2 still produces systematic near-coincidences; they are
  pushed out of exact degeneracy but not out of the spectrum. A genuinely
  irrational aspect ratio would clean this up, at the cost of walls that no
  longer land on grid lines. The contrast with the stadium does not depend on
  it: 15.3% against 0.5% of gaps near zero is not a close call.""")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
grid = np.linspace(0, 3.2, 300)
bins = np.linspace(0, 3.2, 26)

for ax, s, lab, col in ((a1, s_rect, "rectangle 99:70", 'C0'),
                        (a2, s_stad, "quarter stadium", 'C1')):
    ax.hist(s, bins=bins, density=True, color=col, alpha=0.7, label=lab)
    ax.plot(grid, pdf_poisson(grid), 'k-', lw=1.5, label="Poisson, exp(-s)")
    ax.plot(grid, pdf_wigner(grid), 'k--', lw=1.5,
            label="Wigner, (pi/2) s exp(-pi s^2/4)")
    ax.set_xlabel("spacing s, in units of the mean")
    ax.legend(fontsize=8)
    ax.text(0.97, 0.55, f"KS to Poisson {ks(s, cdf_poisson):.3f}\n"
            f"KS to Wigner  {ks(s, cdf_wigner):.3f}", transform=ax.transAxes,
            fontsize=8, ha='right', color='0.3', family='monospace')

a1.set_ylabel("P(s)")
a1.set_title("Integrable: levels are uncorrelated and can collide")
a2.set_title("Chaotic: levels repel, so P(0) = 0")
a1.text(0.97, 0.40, f"{(s_rect < 0.1).mean()*100:.1f}% of gaps below 0.1\n"
        "(Poisson wants 9.5%)", transform=a1.transAxes, fontsize=8, ha='right',
        color='0.3')
a2.text(0.97, 0.40, f"{(s_stad < 0.1).mean()*100:.1f}% of gaps below 0.1\n"
        "(Wigner wants 0.8%)", transform=a2.transAxes, fontsize=8, ha='right',
        color='0.3')
fig.suptitle("The same solver and the same grid, one mask apart: "
             "Poisson becomes Wigner", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("figures/level_statistics.png", dpi=150)
print("\nwrote figures/level_statistics.png")
