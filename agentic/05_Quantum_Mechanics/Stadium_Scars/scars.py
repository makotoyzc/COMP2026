"""Steps 4 and 5: rank the states by how localised they are, then look at them.

The inverse participation ratio

    I = sum_i |psi_i|^4 h^2

has units of 1/area, so the number to quote is I*A, which is dimensionless.
Useful reference values, all for a state normalised to 1 on a table of area A:

    I*A = 1.00   |psi|^2 perfectly flat, which no eigenstate actually is
    I*A = 2.25   a separable sin*sin spread over the whole table. Every
                 rectangle eigenstate has exactly this value -- see
                 check_rect_ipr, where the discrete sums give 9/4 with no
                 correction at all.
    I*A = 3.00   random speckle. This is Berry's guess for what a chaotic
                 eigenstate should look like: a random superposition of plane
                 waves, which makes psi Gaussian, and a Gaussian has
                 <psi^4> = 3 <psi^2>^2.
    I*A = 4.02   a separable state squeezed into the straight half only, i.e.
                 2.25 * A/A_box. This is a pure bouncing-ball state.

So I*A says how localised a state is. It does not say why, and that turns out to
matter more than the ranking. A second number sorts out the reason:

    alpha_y = <p_y^2> / E,      with   <p_x^2> + <p_y^2> = E

computed from the same differences that built the Laplacian, so the sum rule is
a free check that psi was put back on the grid correctly. A state bouncing up
and down between the flat walls has alpha_y near 1, one running along the table
has alpha_y near 0, and a state with no preferred direction sits at 1/2.

This matters because a bouncing-ball state is not a scar. Heller's scars sit on
periodic orbits that are UNSTABLE, where a particle launched slightly off the
orbit leaves within a few bounces. The bouncing-ball orbits are the opposite:
they are neutrally stable, and there is a whole two-parameter family of them
(start anywhere along the flat wall, at any height). The same goes for the
states that pool in the curved cap. Both are localised, and both end up at the
top of the IPR list.

Run:  python scars.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import billiard as bq

H = 1 / 200
K = 400
A, R = 1.0, 1.0
AREA = A * R + np.pi * R ** 2 / 4          # continuum area of the quarter stadium
RECT_F = 3                                 # rectangle 297 x 210, h = 1/210


def anisotropy(psi, E, h):
    """alpha_y = <p_y^2>/E from forward differences, plus the sum rule it obeys.

    psi is zero outside the mask, so a difference that crosses the wall is
    already the right thing: it is the gradient of a function that has just gone
    to zero. Summing both directions reproduces psi^T (-lap) psi = E, which is
    the same matrix the eigenvalue came from -- so agreement to 1e-9 says the
    reconstruction of psi onto the full grid kept every zero in the right place.
    """
    px2 = ((np.diff(psi, axis=1) / h) ** 2).sum(axis=(1, 2)) * h ** 2
    py2 = ((np.diff(psi, axis=2) / h) ** 2).sum(axis=(1, 2)) * h ** 2
    return py2 / E, np.abs((px2 + py2 - E) / E).max()


def family(ia, ay):
    if ay > 0.85:
        return "bouncing ball"
    if ay < 0.40:
        return "cap / gallery"
    return "balanced"


def wall(ax, c='k', lw=0.8):
    t = np.linspace(0, np.pi / 2, 200)
    ax.plot([0, 0, A], [R, 0, 0], c, lw=lw)
    ax.plot([0, A], [R, R], c, lw=lw)
    ax.plot(A + R * np.sin(t), R * np.cos(t), c, lw=lw)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def gallery(path, title, X, Y, mask, psi, E, IA, ay, idx, shape):
    fig, axes = plt.subplots(*shape, figsize=(2.6 * shape[1], 1.85 * shape[0] + 0.8))
    for ax, n in zip(axes.ravel(), idx):
        ax.pcolormesh(X, Y, np.ma.masked_where(~mask, psi[n] ** 2),
                      cmap='magma', shading='gouraud')
        wall(ax, 'w', 0.6)
        ax.set_title(f"n={n+1}  E={E[n]:.0f}\nI*A={IA[n]:.2f}  a_y={ay[n]:.2f}  "
                     f"{family(IA[n], ay[n])}", fontsize=7)
    for ax in axes.ravel()[len(idx):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (1.85 * shape[0] + 0.8)))
    fig.savefig(path, dpi=140)


def check_rect_ipr(IAr, nx, ny):
    """Every rectangle eigenstate is separable, so I*A is the SAME number for all
    of them, and that number is 9/4 exactly on the discrete grid too:

        sum_{j=1..n-1} sin^4(k pi j / n) = 3n/8,   sum sin^2 = n/2

    (use sin^4 = 3/8 - cos2t/2 + cos4t/8 and sum_{j=1}^{n-1} cos(2 pi m j/n) = -1),
    so I*A = 16 * (3nx/8)(3ny/8) / (nx ny) = 9/4 with no 1/n correction at all.
    A spread of zero across 400 states is the check -- it says the IPR routine,
    the normalisation and the area convention are all consistent.
    """
    print(f"   rectangle I*A: mean {IAr.mean():.12f}  std {IAr.std():.2e}  "
          f"exact 9/4 = {9/4:.12f}")
    assert abs(IAr.mean() - 2.25) < 1e-9 and IAr.std() < 1e-9


def main():
    mask, X, Y = bq.quarter_stadium_mask(H, A, R)
    E, psi = bq.solve(mask, H, K)
    IA = bq.ipr(psi, H) * AREA
    ay, sumrule = anisotropy(psi, E, H)

    nx, ny, hr = 99 * RECT_F, 70 * RECT_F, 1.0 / (70 * RECT_F)
    rmask, _, _ = bq.rect_mask(nx, ny, hr)
    Er, rpsi = bq.solve(rmask, hr, K)
    IAr = bq.ipr(rpsi, hr) * (nx * hr) * (ny * hr)

    np.savez("spectrum.npz", E=E, IA=IA, alpha_y=ay, h=H, area=AREA,
             E_rect=Er, IA_rect=IAr, h_rect=hr)

    print(f"stadium   h = 1/{round(1/H)}  {mask.sum()} unknowns  {K} levels to "
          f"E = {E[-1]:.0f}  (wavelength {2*np.pi/np.sqrt(E[-1])/H:.0f} h)")
    print(f"rectangle h = 1/{round(1/hr)}  {rmask.sum()} unknowns  {K} levels to "
          f"E = {Er[-1]:.0f}")
    print(f"\n   momentum sum rule <px^2> + <py^2> = E:  max rel dev {sumrule:.1e}")
    check_rect_ipr(IAr, nx, ny)

    print(f"\n   stadium I*A: min {IA.min():.3f}  median {np.median(IA):.3f}  "
          f"max {IA.max():.3f}  std {IA.std():.3f}")
    print("   -> integrable gives a delta function at 9/4; chaotic gives a spread\n"
          "      centred near Berry's 3.00.  That contrast is the cleanest thing\n"
          "      on this page, and it is one mask apart.")

    order = np.argsort(IA)[::-1]
    print("\n   twenty most localised stadium states")
    print("      rank    n      E      I*A   alpha_y   family")
    for rank, n in enumerate(order[:20]):
        print(f"      {rank+1:4d} {n+1:5d} {E[n]:8.0f}  {IA[n]:6.3f}   {ay[n]:5.2f}"
              f"    {family(IA[n], ay[n])}")

    top = order[:20]
    nbb = sum(ay[n] > 0.85 for n in top)
    ncap = sum(ay[n] < 0.40 for n in top)
    print(f"\n   of the top 20: {nbb} bouncing ball, {ncap} cap/gallery, "
          f"{20-nbb-ncap} balanced")

    bal = np.where((ay > 0.42) & (ay < 0.58))[0]
    bal = bal[np.argsort(IA[bal])[::-1]]
    print(f"   {len(bal)} of {K} states have equipartitioned momentum; the most\n"
          f"   localised of those reaches I*A = {IA[bal[0]]:.3f} against a median "
          f"of {np.median(IA):.3f}")
    print("\n   This is the result of step 4, and it is not the one the step\n"
          "   implies. Ranking by IPR gives a ranking, but not of scars. Both\n"
          "   families at the top of the list sit on neutrally stable orbits,\n"
          "   and a scar needs an unstable one. Take both families out and the\n"
          "   most localised state left is only a few percent above speckle.\n"
          "   Finding real scars needs a measure tied to one specific unstable\n"
          "   orbit, which means solving for that orbit first -- the bounce-map\n"
          "   root-finding the prompt saves for Push Harder.")

    gallery("figures/scars_top20.png",
            "The 20 most localised of 400 states. Eleven bounce up and down, "
            "seven pool in the cap:\nboth are stable orbit families, so neither "
            "is a scar.",
            X, Y, mask, psi, E, IA, ay, top, (4, 5))
    gallery("figures/balanced_top12.png",
            "The same ranking with both families removed (0.42 < alpha_y < 0.58).\n"
            "What is left looks like speckle: best I*A = 3.14 against a median "
            "of 2.89.",
            X, Y, mask, psi, E, IA, ay, bal[:12], (3, 4))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    a1.hist(IA, bins=np.linspace(2.1, 3.6, 50), alpha=0.75,
            label=f"quarter stadium ({K})")
    a1.axvline(IAr.mean(), color='C1', lw=2,
               label=f"rectangle 99:70 ({K}), all at 9/4")
    for v, lab in ((2.25, "2.25 separable"), (3.0, "3.00 speckle"),
                   (4.02, "4.02 box-confined")):
        a1.axvline(v, color='0.3', ls=':', lw=1)
        a1.text(v, a1.get_ylim()[1] * 0.98, lab, rotation=90, fontsize=7,
                ha='right', va='top', color='0.3')
    a1.set_xlabel("I*A"); a1.set_ylabel("states"); a1.legend(fontsize=8)
    a1.set_title("Integrable gives one value; chaotic gives a spread")

    a2.scatter(ay, IA, s=10, c='0.7', label="all 400")
    a2.scatter(ay[top], IA[top], s=26, c='C3', label="top 20 by I*A")
    a2.axhline(np.median(IA), color='0.3', ls=':', lw=1)
    a2.text(0.02, np.median(IA), f" median {np.median(IA):.2f}", fontsize=7,
            va='bottom', color='0.3')
    for v in (0.40, 0.85):
        a2.axvline(v, color='0.3', ls='--', lw=0.8)
    a2.set_xlim(0.17, 1.04)
    a2.set_ylim(IA.min() - 0.30, IA.max() + 0.07)
    for xc, lab in ((0.285, "pools in the\ncurved cap"),
                    (0.625, "no preferred\ndirection"),
                    (0.945, "bounces up\nand down")):
        a2.text(xc, IA.min() - 0.155, lab, fontsize=8, ha='center', va='bottom',
                color='0.3')
    a2.set_xlabel("alpha_y = <p_y^2> / E"); a2.set_ylabel("I*A")
    a2.legend(fontsize=8, loc='upper center', ncol=2)
    a2.set_title("The most localised states are the directional ones")
    fig.tight_layout()
    fig.savefig("figures/ipr_distribution.png", dpi=140)
    print("\nwrote figures/scars_top20.png, figures/balanced_top12.png,\n"
          "      figures/ipr_distribution.png, spectrum.npz")


if __name__ == "__main__":
    main()
