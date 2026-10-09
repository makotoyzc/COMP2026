"""Quantum billiards: eigenvalues of a particle in a hard-walled 2D box.

Units: hbar = 1 and 2m = 1, so the Schrodinger equation is

    -lap psi = E psi     inside the table
         psi = 0         on the wall

and E is a pure number set by the shape and its size.

HOW THE EIGENVALUE PROBLEM IS BUILT
-----------------------------------
Put a square grid of spacing h over the table. Replace the Laplacian with the
five-point second difference:

    (-lap psi)_ij = [ 4 psi_ij - psi_i-1,j - psi_i+1,j
                             - psi_i,j-1 - psi_i,j+1 ] / h^2

Every grid node now holds one number, and -lap is a matrix acting on the list of
those numbers. So "solve Schrodinger" becomes "find eigenvectors of a matrix",
and the eigenvalues of that matrix are the energies.

The matrix is big (71,000 x 71,000 at h = 1/200) but has at most 5 nonzeros per
row, so it is stored sparse and solved with scipy's Lanczos routine eigsh.

HOW THE BOUNDARY CONDITION GETS IN
----------------------------------
The mask is a boolean array, one entry per grid node, with one job:

    mask True   ->  psi here is an unknown, the solver finds it
    mask False  ->  psi here is known, and it is 0

There is no third option. "On or past the wall" and "not an unknown" are the
same thing, so the matrix only ever has mask.sum() rows.

The stencil above needs four neighbour values. At a node against the wall, one
of them is off the table, so you have to say what it is. The two standard
choices do different things to the matrix:

    psi_outside = 0          the term is 0, so drop that off-diagonal entry
                             and keep the 4 on the diagonal.  Hard wall.

    psi_outside = psi_here   the term becomes +psi_here/h^2, which cancels one
                             unit of the diagonal, so the 4 becomes a 3.
                             Reflecting wall.

That is the entire difference. Same mask, same off-diagonal entries; a hard wall
has 4 on the diagonal everywhere and a reflecting wall has (number of links
kept). Nothing else in the code mentions the boundary.

The practical warning is that getting this wrong is silent. If you drop a
neighbour and also drop the 1 from the diagonal, the code runs fine and returns
a perfectly good spectrum of the wrong problem. checks_eigenvalues.py builds
both matrices on purpose and shows they disagree in the first digit.

The zeros show up in three places, and they are all the same zero:
  1. matrix assembly   a link to a masked-out neighbour is simply absent
  2. problem size      the matrix is N x N with N = mask.sum(), no more
  3. reconstruction    the eigenvector is scattered back into np.zeros(),
                       so every node off the table is literally 0.0
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------- grids & masks

def grid(x0, x1, y0, y1, h):
    """Nodes on [x0,x1] x [y0,y1] with spacing h, inclusive of both ends.

    The 0.5*h pad in the stop value is so that an endpoint that lands exactly on a
    multiple of h is kept; it never adds a spurious extra node.
    """
    x = np.arange(x0, x1 + 0.5 * h, h)
    y = np.arange(y0, y1 + 0.5 * h, h)
    return np.meshgrid(x, y, indexing='ij')


def rect_mask(nx, ny, h):
    """Rectangle (nx*h) x (ny*h), walls landing exactly on grid lines.

    Nodes run i = 0..nx, j = 0..ny. The four edges are wall, so the unknowns are
    the (nx-1)*(ny-1) strictly interior nodes. Because the wall is exactly on a
    grid line there is no staircase error here at all -- the discrete problem has
    a closed-form spectrum (see exact_rect_discrete), which is what makes this the
    test case.
    """
    X, Y = grid(0.0, nx * h, 0.0, ny * h, h)
    mask = np.zeros(X.shape, dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask, X, Y


def quarter_stadium_mask(h, a=1.0, r=1.0):
    """Quarter of a Bunimovich stadium: x >= 0, y >= 0, hard wall on all of it.

    Full stadium = rectangle |x| <= a, |y| <= r, capped by half discs of radius r
    at x = +-a. The full table has two reflection symmetries, so its spectrum is
    four independent spectra interleaved and the level statistics come out wrong.
    Solving the quarter with Dirichlet walls on the symmetry axes x = 0 and y = 0
    picks out one of those four -- the odd-odd sector -- and that is a single
    clean spectrum.

    Mask = nodes STRICTLY inside. A node exactly on x = 0, on y = 0, on y = r, or
    on the arc is wall, hence False, hence psi = 0 there. The arc is the one place
    where the wall does not follow grid lines, so the mask staircases and that
    edge carries O(h) error while the straight edges carry none.
    """
    X, Y = grid(0.0, a + r, 0.0, r, h)
    eps = 1e-12
    in_box = (X > eps) & (X <= a) & (Y > eps) & (Y < r - eps)
    in_cap = (X > a) & (Y > eps) & ((X - a) ** 2 + Y ** 2 < r ** 2 - eps)
    return (in_box | in_cap), X, Y


# ------------------------------------------------------------ masked Laplacian

def index_map(mask):
    """Node grid -> unknown number, with -1 meaning 'not an unknown' (psi = 0)."""
    idx = np.full(mask.shape, -1, dtype=np.int64)
    idx[mask] = np.arange(mask.sum())
    return idx


def laplacian(mask, h, wall="hard"):
    """-lap as a sparse matrix over the masked nodes only.

    The index grid is padded with a ring of -1 so the four neighbour slices can be
    taken without np.roll wrapping the right edge onto the left. A link is written
    only when BOTH ends are unknowns.

    wall = "hard" (psi = 0, Dirichlet) or "mirror" (dpsi/dn = 0, Neumann).

    The off-diagonal part of the matrix is IDENTICAL for the two. The only
    difference is the diagonal:

        hard    diagonal = 4/h^2 at every node, even next to a wall
        mirror  diagonal = (number of links kept)/h^2

    Why that is the whole boundary condition. The stencil at a node needs four
    neighbour values. For a node against the wall one of them is off the table,
    so you have to say what it is:

        psi_outside = 0           -> the term vanishes, drop the link,
                                     keep the 4.  This is psi = 0 on the wall.
        psi_outside = psi_here    -> the term becomes +psi_here/h^2, which
                                     cancels one unit of the diagonal, so the
                                     4 drops to 3.  This is a reflecting wall.

    So the same mask with 4 on the diagonal has hard walls and with 3 has mirror
    walls, and nothing else in the code changes. Use "mirror" only to show the
    difference -- checks_eigenvalues.py does exactly that, and the two spectra
    come out completely different, the mirror one starting at E = 0 because a
    constant is allowed.
    """
    idx = index_map(mask)
    pad = np.full(np.array(mask.shape) + 2, -1, dtype=np.int64)
    pad[1:-1, 1:-1] = idx

    here = pad[1:-1, 1:-1]
    n = int(mask.sum())
    kept = np.zeros(n)
    rows, cols, vals = [], [], []

    for nb in (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]):
        link = (here >= 0) & (nb >= 0)          # both ends are unknowns
        rows.append(here[link])
        cols.append(nb[link])
        vals.append(np.full(int(link.sum()), -1.0 / h ** 2))
        np.add.at(kept, here[link], 1.0)

    diag = np.full(n, 4.0) if wall == "hard" else kept
    rows.append(np.arange(n)); cols.append(np.arange(n))
    vals.append(diag / h ** 2)

    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def solve(mask, h, k, wall="hard"):
    """Lowest k eigenpairs. Returns E (k,) and psi (k, *mask.shape).

    eigsh with sigma=0 is shift-invert: it factors the matrix once and converges
    on the eigenvalues nearest zero, which for a positive operator are the lowest
    ones. Asking for 'SM' directly instead would converge painfully slowly.

    psi is scattered back onto the FULL node grid and is exactly 0.0 everywhere
    outside the mask, and normalised so that sum |psi|^2 h^2 = 1, i.e. the
    trapezoid-free discrete version of the integral over the table.
    """
    E, V = eigsh(laplacian(mask, h, wall), k=k, sigma=-1e-8)
    order = np.argsort(E)
    E, V = E[order], V[:, order]

    psi = np.zeros((k,) + mask.shape)
    psi[:, mask] = V.T / h                      # eigsh returns unit 2-norm columns
    return E, psi


def ipr(psi, h):
    """Inverse participation ratio I = sum |psi|^4 h^2, for psi normalised as above.

    Units of 1/area, so I*A is the dimensionless number to compare: a state spread
    evenly over the table gives I*A = 1, Gaussian random speckle gives 3, a
    separable sin*sin gives 9/4, and a state squeezed onto a periodic orbit gives
    more. Masked-out nodes contribute 0^4 = 0, so the sum over the full array is
    already the sum over the table.
    """
    return np.sum(psi ** 4, axis=(-2, -1)) * h ** 2


# ---------------------------------------------------- closed forms for the test

def exact_rect_discrete(nx, ny, h):
    """Eigenvalues of the DISCRETE rectangle problem, exactly.

    The five-point operator separates, and the 1D Dirichlet second difference on
    n-1 interior points has eigenvalues (4/h^2) sin^2(k pi / 2n), k = 1..n-1.
    This is what the solver must reproduce to machine precision -- it tests the
    mask, the stencil, the index map and the solver all at once, with no
    discretisation error in the way.
    """
    kx = np.arange(1, nx)[:, None]
    ky = np.arange(1, ny)[None, :]
    E = (4.0 / h ** 2) * (np.sin(kx * np.pi / (2 * nx)) ** 2
                          + np.sin(ky * np.pi / (2 * ny)) ** 2)
    return np.sort(E.ravel())


def exact_rect_continuum(nx, ny, h, count):
    """E_nm = pi^2 (n^2/Lx^2 + m^2/Ly^2), the answer the grid is converging to."""
    Lx, Ly = nx * h, ny * h
    n = np.arange(1, 400)[:, None]
    m = np.arange(1, 400)[None, :]
    E = np.pi ** 2 * (n ** 2 / Lx ** 2 + m ** 2 / Ly ** 2)
    return np.sort(E.ravel())[:count]


def exact_rect_mirror(nx, ny, h):
    """Eigenvalues of the same rectangle with REFLECTING walls, exactly.

    The mask is unchanged, so there are still (nx-1) by (ny-1) unknowns, but a
    reflecting wall sits half a cell outside the last node, so the 1D problem on
    N points has eigenvalues (4/h^2) sin^2(k pi / 2N) for k = 0 .. N-1 -- note k
    starts at 0, where the hard-wall version started at 1. The k = m = 0 mode is
    a constant with E = 0, which a hard wall forbids and a reflecting wall allows.
    """
    kx = np.arange(nx - 1)[:, None]
    ky = np.arange(ny - 1)[None, :]
    E = (4.0 / h ** 2) * (np.sin(kx * np.pi / (2 * (nx - 1))) ** 2
                          + np.sin(ky * np.pi / (2 * (ny - 1))) ** 2)
    return np.sort(E.ravel())
