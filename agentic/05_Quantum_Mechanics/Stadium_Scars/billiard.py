"""Quantum billiards by a masked finite-difference Laplacian.

Units: hbar = 1, 2m = 1, so the problem is  -lap psi = E psi  with psi = 0 on the
wall, and E is a pure number set by the shape and its size.

THE MASK IS THE BOUNDARY CONDITION
----------------------------------
Lay a uniform square grid of spacing h over a box that contains the table. Every
node of that grid carries a value of psi. The mask is a boolean array over those
nodes that answers one question per node:

    mask[i, j] == True   ->  psi here is an UNKNOWN, solved for.
    mask[i, j] == False  ->  psi here is KNOWN, and it is exactly 0.

False means the node is on the wall or outside it. There is no third category.
So the wall is not a term in the equation, not a penalty, not a large diagonal --
it is the set of nodes we refuse to make unknowns, and the value they carry is
zero. "Dirichlet" and "not in the mask" are the same statement.

That is why the matrix below has fewer entries than the stencil suggests. The
five-point form of the operator at an interior node is

    (-lap psi)_ij = [ 4 psi_ij - psi_i-1,j - psi_i+1,j - psi_i,j-1 - psi_i,j+1 ] / h^2

and when a neighbour is outside the mask its psi is 0, so that whole term is 0 and
we leave the off-diagonal entry out of the matrix. Nothing else changes. In
particular the diagonal stays 4/h^2 at EVERY unknown, including the ones sitting
right against a wall: dropping a neighbour must not also shrink the diagonal, or
you have quietly reflected the wavefunction instead of pinning it to zero and you
are solving a Neumann problem. The zeros live off the diagonal, never on it.

Three places the zeros show up, and all three are the same zero:
  1. matrix assembly   -- a link to a masked-out neighbour is simply absent,
  2. the unknown count -- the matrix is N x N with N = mask.sum(), nothing more,
  3. reconstruction    -- scatter the eigenvector back with np.zeros(mask.shape),
                          so every node outside the mask is literally 0.0 in the
                          array we plot.
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


def laplacian(mask, h):
    """-lap as a sparse matrix over the masked nodes only.

    The index grid is padded with a ring of -1 so the four neighbour slices can be
    taken without np.roll wrapping the right edge onto the left. A link is written
    only when BOTH ends are unknowns; a link to a -1 is the dropped term, i.e. the
    boundary condition.
    """
    idx = index_map(mask)
    pad = np.full(np.array(mask.shape) + 2, -1, dtype=np.int64)
    pad[1:-1, 1:-1] = idx

    here = pad[1:-1, 1:-1]
    n = int(mask.sum())
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0 / h ** 2)]

    for nb in (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]):
        link = (here >= 0) & (nb >= 0)          # both ends are unknowns
        rows.append(here[link])
        cols.append(nb[link])
        vals.append(np.full(int(link.sum()), -1.0 / h ** 2))

    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def solve(mask, h, k):
    """Lowest k eigenpairs. Returns E (k,) and psi (k, *mask.shape).

    eigsh with sigma=0 is shift-invert: it factors the matrix once and converges
    on the eigenvalues nearest zero, which for a positive operator are the lowest
    ones. Asking for 'SM' directly instead would converge painfully slowly.

    psi is scattered back onto the FULL node grid and is exactly 0.0 everywhere
    outside the mask, and normalised so that sum |psi|^2 h^2 = 1, i.e. the
    trapezoid-free discrete version of the integral over the table.
    """
    E, V = eigsh(laplacian(mask, h), k=k, sigma=0.0)
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
