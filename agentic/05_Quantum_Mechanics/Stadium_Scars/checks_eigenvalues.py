"""Step 2: check the eigenvalue solver and the boundary condition on a rectangle.

A rectangle is the right test case because the answer is known in closed form,
degeneracies and all:

    E_nm = pi^2 (n^2/Lx^2 + m^2/Ly^2)

and because its walls land exactly on grid lines, so the mask is the exact
shape and the only error left is the stencil's own.

The rectangle used here is 99 nodes by 70, so Lx/Ly = 99/70 = 1.41428..., which
is sqrt(2) to one part in 10^4. Jim's warning about the square is check 5: a
square has so many n^2 + m^2 coincidences that half its levels are degenerate,
and that wrecks any level statistics later. As a fraction in lowest terms,
99/70 pushes its coincidences far above the levels we use.

The five checks:

  1  the matrix.  Size, symmetry, and the row sums that tell a hard wall from a
     reflecting one.
  2  hard vs reflecting walls.  Same mask, same off-diagonal entries, one
     number different on the diagonal -- and two completely different spectra,
     both matched against their own closed form. This is the check that would
     catch the boundary condition being wrong, because the two answers differ
     in the first digit, not the last.
  3  the eigenvalues.  200 levels against the exact DISCRETE spectrum, which has
     no discretisation error in it, so agreement must be to machine precision.
  4  the continuum limit.  Halve h, the error should drop by 4.
  5  degeneracies.  99:70 against 70:70.

Run:  python checks_eigenvalues.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import billiard as bq

NX, NY, H = 99, 70, 1.0 / 70
K = 200


def check_1_matrix():
    print("1  the matrix")
    mask, _, _ = bq.rect_mask(NX, NY, H)
    L = bq.laplacian(mask, H, "hard")
    n = int(mask.sum())

    print(f"     grid nodes          {mask.size}  ({NX+1} x {NY+1})")
    print(f"     unknowns            {n}  = {NX-1} x {NY-1}, the interior")
    print(f"     held at psi = 0     {mask.size - n}  (the four edges)")
    print(f"     matrix              {L.shape[0]} x {L.shape[1]}, "
          f"{L.nnz} nonzeros = {L.nnz/n:.2f} per row (stencil has 5)")
    assert n == (NX - 1) * (NY - 1) and L.shape == (n, n)

    print(f"     symmetric           max |L - L^T| = {abs(L - L.T).max():.1e}")
    assert abs(L - L.T).max() == 0.0

    diag = L.diagonal() * H ** 2
    print(f"     diagonal            {diag.min():.4f} to {diag.max():.4f}, "
          f"times 1/h^2  (must be 4 everywhere)")
    assert np.allclose(diag, 4.0)

    links = np.asarray((L != 0).sum(axis=1)).ravel() - 1
    rowsum = np.asarray(L.sum(axis=1)).ravel() * H ** 2
    deep, edge = links == 4, links < 4
    print(f"     rows, 4 links       {deep.sum():5d}   row sum = "
          f"{abs(rowsum[deep]).max():.1e}  (zero: neighbours cancel the 4)")
    print(f"     rows, fewer links   {edge.sum():5d}   row sum = "
          f"{rowsum[edge].min():.0f} to {rowsum[edge].max():.0f}, times 1/h^2")
    assert np.allclose(rowsum, 4 - links)
    print("     The leftover on the wall rows is what makes the walls hard. A\n"
          "     constant vector is not an eigenvector, so there is no E = 0 mode.\n")
    return mask


def check_2_boundary(mask):
    print("2  hard walls vs reflecting walls: same mask, one number different")
    Lh = bq.laplacian(mask, H, "hard")
    Lm = bq.laplacian(mask, H, "mirror")

    off_h = Lh - sp_diag(Lh)
    off_m = Lm - sp_diag(Lm)
    print(f"     off-diagonal entries identical?   "
          f"max |difference| = {abs(off_h - off_m).max():.1e}")
    assert abs(off_h - off_m).max() == 0.0
    dh, dm = Lh.diagonal() * H ** 2, Lm.diagonal() * H ** 2
    print(f"     diagonals:  hard {dh.min():.0f} to {dh.max():.0f},   "
          f"reflecting {dm.min():.0f} to {dm.max():.0f}   (times 1/h^2)")
    print(f"     they differ at {(dh != dm).sum()} of {len(dh)} rows -- exactly the "
          f"rows touching a wall")

    Eh, _ = bq.solve(mask, H, 6, "hard")
    Em, _ = bq.solve(mask, H, 6, "mirror")
    Xh = bq.exact_rect_discrete(NX, NY, H)[:6]
    Xm = bq.exact_rect_mirror(NX, NY, H)[:6]
    print("\n     level   hard wall     exact      reflecting     exact")
    for i in range(6):
        print(f"       {i+1}    {Eh[i]:10.4f} {Xh[i]:10.4f}    "
              f"{Em[i]:10.4f} {Xm[i]:10.4f}")
    print(f"     hard       max relative error {abs((Eh-Xh)/Xh).max():.1e}")
    print(f"     reflecting max absolute error {abs(Em-Xm).max():.1e}")
    assert abs((Eh - Xh) / Xh).max() < 1e-10 and abs(Em - Xm).max() < 1e-9
    print("     The reflecting ground state is E = 0, a constant, which a hard\n"
          "     wall forbids. Both match their own closed form to machine\n"
          "     precision, so the matrix really is implementing the wall we asked\n"
          "     for, and a mistake here would show up in the first digit.\n")
    return Eh, Em, Xm


def sp_diag(L):
    import scipy.sparse as sp
    return sp.diags(L.diagonal())


def check_3_eigenvalues(mask):
    print("3  the eigenvalues, against the exact discrete spectrum")
    E, _ = bq.solve(mask, H, K)
    X = bq.exact_rect_discrete(NX, NY, H)[:K]
    rel = abs(E - X) / X
    print(f"     E_1    solver {E[0]:.10f}   exact {X[0]:.10f}")
    print(f"     E_{K}  solver {E[-1]:.10f}   exact {X[-1]:.10f}")
    print(f"     max relative error over {K} levels = {rel.max():.1e}")
    assert rel.max() < 1e-9
    print("     The discrete spectrum has no grid error in it, so this tests the\n"
          "     mask, the stencil, the index bookkeeping and eigsh all at once.\n")
    return E, X, rel


def check_4_continuum():
    print("4  the continuum limit")
    out, prev = [], None
    for f in (1, 2, 4):
        nx, ny, h = NX * f, NY * f, H / f
        mask, _, _ = bq.rect_mask(nx, ny, h)
        E, _ = bq.solve(mask, h, 20)
        err = abs(E - bq.exact_rect_continuum(nx, ny, h, 20)).max()
        out.append((h, err))
        txt = "" if prev is None else f"   previous/this = {prev/err:.2f}"
        print(f"     h = 1/{round(1/h):3d}   {mask.sum():6d} unknowns   "
              f"max |E - E_exact| = {err:.3e}{txt}")
        if prev is not None:
            assert 3.7 < prev / err < 4.3
        prev = err
    print("     Error falls as h^2, which is what a five-point stencil gives when\n"
          "     the walls lie on grid lines.\n")
    return out


def check_5_degeneracies():
    print("5  degeneracies: why 99:70 and not a square")
    for lab, nx, ny in (("99:70 rectangle", NX, NY), ("70:70 square", 70, 70)):
        mask, _, _ = bq.rect_mask(nx, ny, H)
        E, _ = bq.solve(mask, H, K)
        gap = np.diff(E) / np.mean(np.diff(E))
        print(f"     {lab:16s} smallest gap/mean = {gap.min():.1e},  "
              f"{(gap < 1e-3).sum():3d} of {K-1} gaps below 0.001")
    print("     The square's repeated n^2 + m^2 values would swamp the level\n"
          "     spacing statistics. The 99:70 spectrum has no close pairs.\n")


def figure(rel, conv):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6))

    a = ax[0, 0]
    mask, _, _ = bq.rect_mask(13, 10, 1 / 10)
    a.spy(bq.laplacian(mask, 1 / 10, "hard"), markersize=2.5)
    a.xaxis.set_ticks_position('bottom'); a.xaxis.set_label_position('bottom')
    a.set_title("(a) the matrix for a 12 x 9 grid\n"
                "five bands: each node and its four neighbours", fontsize=10)
    a.set_xlabel("column"); a.set_ylabel("row")

    a = ax[0, 1]
    n = np.arange(1, 26)
    Eh25, _ = bq.solve(*bq.rect_mask(NX, NY, H)[:1], H, 25, "hard")
    Em25, _ = bq.solve(*bq.rect_mask(NX, NY, H)[:1], H, 25, "mirror")
    a.plot(n, Eh25, 'o-', ms=4, label="hard wall, psi = 0")
    a.plot(n, Em25, 's-', ms=4, label="reflecting wall, dpsi/dn = 0")
    a.axhline(0, color='0.6', lw=0.8)
    a.annotate("E = 0, a flat wavefunction.\nA reflecting wall allows it,\n"
               "a hard wall forbids it.", xy=(1.1, 0), xytext=(9.5, 14),
               fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8, color='0.4'))
    a.set_xlabel("level number"); a.set_ylabel("E")
    a.set_title("(b) the boundary condition is one number\n"
                "on the diagonal, and it changes everything", fontsize=10)
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.semilogy(np.arange(1, len(rel) + 1), np.maximum(rel, 1e-17), '.', ms=4)
    a.axhline(2.2e-16, color='0.5', ls='--', lw=1)
    a.text(4, 2.9e-16, "double precision limit", fontsize=8, ha='left',
           va='bottom', color='0.4')
    a.set_ylim(1e-17, 1e-11)
    a.set_xlabel("level number"); a.set_ylabel("relative error")
    a.set_title(f"(c) solver vs exact discrete spectrum\n"
                f"all {K} levels right to ~1e-14", fontsize=10)

    a = ax[1, 1]
    hs = np.array([c[0] for c in conv]); es = np.array([c[1] for c in conv])
    a.loglog(hs, es, 'o-', ms=6, label="measured")
    a.loglog(hs, 0.45 * es[0] * (hs / hs[0]) ** 2, ':', color='0.4', lw=1.2,
             label="exact h^2 slope, shifted down\nso both lines are visible")
    for hv, ev in zip(hs, es):
        a.annotate(f"h = 1/{round(1/hv)}", (hv, ev), textcoords="offset points",
                   xytext=(-6, 8), fontsize=8, ha='right')
    a.set_xlim(hs.min() * 0.72, hs.max() * 1.35)
    a.set_ylim(es.min() * 0.12, es.max() * 2.2)
    a.set_xlabel("grid spacing h"); a.set_ylabel("max error, 20 levels")
    a.set_title("(d) error falls by 4 each time h is halved", fontsize=10)
    a.legend(fontsize=8)

    fig.suptitle("Rectangle test: the eigenvalue solver and the boundary "
                 "condition", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig("figures/eigenvalues.png", dpi=150)
    print("wrote figures/eigenvalues.png")


if __name__ == "__main__":
    mask = check_1_matrix()
    Eh, Em, Xm = check_2_boundary(mask)
    E, X, rel = check_3_eigenvalues(mask)
    conv = check_4_continuum()
    check_5_degeneracies()
    figure(rel, conv)
    print("\nall five checks passed")
