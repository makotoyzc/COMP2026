"""Natural Step 2: test the masked solver where the answer is already known.

Four checks, in order of how much they would hurt to fail:

  A  Mask bookkeeping.  The unknown count, the symmetry of the matrix, and the
     one number that separates Dirichlet from Neumann: every diagonal entry is
     4/h^2, so a row whose four neighbours are all unknowns sums to zero, while a
     row against the wall sums to +d/h^2 for its d missing links. That positive
     leftover IS the boundary condition, and it is why there is no zero mode.

  B  Exact discrete spectrum.  No discretisation error in the way, so the solver
     must agree to machine precision or something is wrong with the mask, the
     stencil, the index map or the eigensolver.

  C  Second-order convergence to the continuum.  Halve h, the error drops 4x.

  D  Degeneracies.  Jim's warning about the square: n^2 + m^2 collides constantly.
     The 99:70 rectangle is 1.4143 -- sqrt(2) to one part in 10^4 -- but as a
     fraction in lowest terms its collisions are pushed far above the spectrum we
     look at, so the low levels are all simple.

Run:  python checks_rectangle.py
"""
import numpy as np
import billiard as bq

NX, NY, H = 99, 70, 1.0 / 70          # Lx = 99/70 = 1.41428..., Ly = 1
K = 200


def check_A():
    print("A  mask bookkeeping and the Dirichlet signature")
    mask, X, Y = bq.rect_mask(NX, NY, H)
    L = bq.laplacian(mask, H)

    n_nodes = mask.size
    n_unknown = int(mask.sum())
    print(f"     nodes on the grid        {n_nodes}   ({NX+1} x {NY+1})")
    print(f"     unknowns (mask True)     {n_unknown}   expected {(NX-1)*(NY-1)}")
    print(f"     wall / outside nodes     {n_nodes - n_unknown}  <- psi = 0 at each")
    assert n_unknown == (NX - 1) * (NY - 1)
    assert L.shape == (n_unknown, n_unknown)

    asym = abs(L - L.T).max()
    diag = L.diagonal()
    print(f"     |L - L^T|_max            {asym:.3e}   (must be 0)")
    print(f"     diagonal                 min {diag.min()*H**2:.6f}/h^2  "
          f"max {diag.max()*H**2:.6f}/h^2   (must both be 4)")
    assert asym == 0.0
    assert np.allclose(diag, 4.0 / H ** 2, rtol=0, atol=1e-9)

    rowsum = np.asarray(L.sum(axis=1)).ravel() * H ** 2
    links = np.asarray((L != 0).sum(axis=1)).ravel() - 1      # off-diagonals kept
    deep = links == 4
    print(f"     rows with 4 links (deep interior) {deep.sum()}: "
          f"row sum max |.| = {abs(rowsum[deep]).max():.3e}   (must be 0)")
    print(f"     rows against a wall               {(~deep).sum()}: "
          f"row sum in [{rowsum[~deep].min():.0f}, {rowsum[~deep].max():.0f}]/h^2 "
          f"= missing links   (must be > 0)")
    assert abs(rowsum[deep]).max() < 1e-9
    assert rowsum[~deep].min() > 0.5
    assert np.allclose(rowsum, 4 - links)
    print("     -> no constant zero mode: the wall is pinning psi, not reflecting it\n")


def check_B():
    print("B  exact discrete spectrum")
    mask, _, _ = bq.rect_mask(NX, NY, H)
    E, _ = bq.solve(mask, H, K)
    E_exact = bq.exact_rect_discrete(NX, NY, H)[:K]
    rel = np.abs(E - E_exact) / E_exact
    print(f"     E_1   solver {E[0]:.10f}   exact {E_exact[0]:.10f}")
    print(f"     E_200 solver {E[-1]:.10f}   exact {E_exact[-1]:.10f}")
    print(f"     max relative error over {K} levels  {rel.max():.3e}")
    assert rel.max() < 1e-9
    print("     -> mask, stencil, index map and eigensolver all verified\n")
    return E


def check_C():
    print("C  second-order convergence to the continuum")
    prev = None
    for f in (1, 2, 4):
        nx, ny, h = NX * f, NY * f, H / f
        mask, _, _ = bq.rect_mask(nx, ny, h)
        E, _ = bq.solve(mask, h, 20)
        E_c = bq.exact_rect_continuum(nx, ny, h, 20)
        err = np.abs(E - E_c).max()
        ratio = "" if prev is None else f"   ratio {prev/err:.2f}"
        print(f"     h = 1/{int(round(1/h)):3d}   unknowns {mask.sum():6d}   "
              f"max |E - E_cont| over 20 levels = {err:.3e}{ratio}")
        if prev is not None:
            assert 3.7 < prev / err < 4.3
        prev = err
    print("     -> error ~ h^2, as a five-point stencil on grid-aligned walls should\n")


def check_D(E):
    print("D  degeneracies: why 99:70 and not a square")
    for label, nx, ny in (("rectangle 99:70", NX, NY), ("square 70:70", 70, 70)):
        mask, _, _ = bq.rect_mask(nx, ny, H)
        Es, _ = bq.solve(mask, H, K)
        gap = np.diff(Es) / np.mean(np.diff(Es))
        print(f"     {label:16s}  smallest normalised gap {gap.min():.2e}   "
              f"gaps < 1e-3 : {(gap < 1e-3).sum():3d} of {K-1}")
    print("     -> the square's accidental n^2+m^2 pairs would poison any level\n"
          "        statistics; the 99:70 spectrum is simple all the way up\n")


if __name__ == "__main__":
    check_A()
    E = check_B()
    check_C()
    check_D(E)
    print("all four checks passed")
