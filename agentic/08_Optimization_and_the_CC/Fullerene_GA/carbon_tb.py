"""Carbon tight binding of Xu, Wang, Chan & Ho, J. Phys.: Condens. Matter 4, 6047 (1992).

    E_tot = 2 sum_occ eps_n + sum_i f( sum_j phi(r_ij) )

sp3 orthogonal basis, 4 valence electrons per atom, energies in eV, lengths in Angstrom.

energy(R)                                cluster energy, R is an (N, 3) array
energy_forces(R)                         cluster energy and analytic forces
levels(R)                                one-electron levels and HOMO-LUMO gap
energy_periodic(lattice, basis, kmesh)   crystal, energy per cell, band term k-averaged
binding(E, nat)                          binding energy per atom relative to free atoms
"""
import itertools

import numpy as np

# ---------------------------------------------------------------------------
# Parameters: text and Tables 1-2 of Xu et al. (1992), checked against the paper 2026-10-30
# ---------------------------------------------------------------------------
ES, EP = -2.99, 3.71
V0 = dict(ss_sigma=-5.0, sp_sigma=4.7, pp_sigma=5.5, pp_pi=-1.55)   # at r = R0
R0, N_S, NC_S, RC_S = 1.536329, 2.0, 6.5, 2.18                       # hopping s(r)
PHI0, M_P, MC_P, DC_P, D0_P = 8.18555, 3.30304, 8.6655, 2.1052, 1.64  # repulsion phi(r)
C = np.array([-2.5909765118191, 0.5721151498619, -1.7896349903996e-3,
              2.3539221516757e-5, -1.24251169551587e-7])               # f(x) = sum C_k x^k
R1, D1, RM = 2.45, 2.57, 2.6        # cubic tails start at R1 for s(r), at D1 for phi(r); 0 beyond RM
# Our change, not in the paper: the quartic f(x) peaks at x = 146.5 eV and then falls, so
# squeezed atoms would attract without limit. Beyond X_LIN, f continues as its tangent line.
# Real structures have x ~ 50 (C60 49, diamond 47); x > X_LIN needs atoms closer than ~1.2 A.
X_LIN = 100.0

ORBITALS = ("s", "px", "py", "pz")
AXIS = {"px": 0, "py": 1, "pz": 2}
ONSITE = np.array([ES, EP, EP, EP])


# ---------------------------------------------------------------------------
# Slater-Koster two-center integrals (copied from PHYS 7324 tb.py)
# ---------------------------------------------------------------------------
def sk(o1, o2, u, V):
    """<o1 on atom i | H | o2 on atom j> for the bond i -> j with unit vector u.

    E_s,s = V_ss_sigma;  E_s,pa = u_a V_sp_sigma;  E_pa,s = -u_a V_sp_sigma;
    E_pa,pb = u_a u_b (V_pp_sigma - V_pp_pi) + delta_ab V_pp_pi.
    """
    if o1 == "s" and o2 == "s":
        return V.get("ss_sigma", 0.0)
    if o1 == "s":
        return u[AXIS[o2]] * V.get("sp_sigma", 0.0)
    if o2 == "s":
        return -u[AXIS[o1]] * V.get("sp_sigma", 0.0)
    la, lb = u[AXIS[o1]], u[AXIS[o2]]
    return la * lb * (V.get("pp_sigma", 0.0) - V.get("pp_pi", 0.0)) \
        + (o1 == o2) * V.get("pp_pi", 0.0)


def sk_blocks(u):
    """4x4 blocks <a on i|H|b on j> at r = R0 for bonds with unit vectors u, shape (P, 3).
    sk() is called with u.T so each entry is evaluated for all P bonds at once."""
    B = np.empty((len(u), 4, 4))
    for a, o1 in enumerate(ORBITALS):
        for b, o2 in enumerate(ORBITALS):
            B[:, a, b] = sk(o1, o2, u.T, V0)
    return B


# ---------------------------------------------------------------------------
# Radial functions with the smooth cutoff
# ---------------------------------------------------------------------------
def radial(a0, r_ref, p, rc, pc, r1):
    """g(r) = a0 (r_ref/r)^p exp{p[-(r/rc)^pc + (r_ref/rc)^pc]} for r < r1; on r1 <= r < RM
    a cubic in (r - r1) that matches g and g' at r1 and reaches 0 with zero slope at RM
    (the paper's rule); 0 beyond. Returns a function r -> (g, dg/dr)."""
    def smooth(r):
        g = a0 * (r_ref / r)**p * np.exp(p * (-(r / rc)**pc + (r_ref / rc)**pc))
        return g, -p * g / r * (1 + pc * (r / rc)**pc)

    g1, dg1 = smooth(r1)
    L = RM - r1
    c2, c3 = np.linalg.solve([[L**2, L**3], [2 * L, 3 * L**2]], [-g1 - dg1 * L, -dg1])
    tail = np.array([g1, dg1, c2, c3])

    def g(r):
        r = np.asarray(r, float)
        x = r - r1
        gs, dgs = smooth(np.minimum(r, r1))          # clip so the far branch never overflows
        gt = tail[0] + tail[1] * x + tail[2] * x**2 + tail[3] * x**3
        dgt = tail[1] + 2 * tail[2] * x + 3 * tail[3] * x**2
        val = np.where(r < r1, gs, np.where(r < RM, gt, 0.0))
        der = np.where(r < r1, dgs, np.where(r < RM, dgt, 0.0))
        return val, der

    g.tail = tail
    return g


s_r = radial(1.0, R0, N_S, RC_S, NC_S, R1)
phi_r = radial(PHI0, D0_P, M_P, DC_P, MC_P, D1)

# Free atom: s2 p2 and f(0) = C0
E_ATOM = 2 * ES + 2 * EP + C[0]


F_POLY = np.poly1d(C[::-1])
F_DER = F_POLY.deriv()


def f_rep(x):
    """f(x) = sum_k C_k x^k for x <= X_LIN, its tangent line beyond. Returns (f, f')."""
    x = np.asarray(x, float)
    xc = np.minimum(x, X_LIN)
    return (F_POLY(xc) + F_DER(X_LIN) * np.maximum(x - X_LIN, 0.0),
            np.where(x <= X_LIN, F_DER(xc), F_DER(X_LIN)))


def binding(E, nat):
    """Binding energy per atom: E/nat - E_ATOM (negative = bound)."""
    return E / nat - E_ATOM


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------
def pairs(R):
    """All i < j with r_ij < RM. Returns i, j, d = R_j - R_i, r."""
    i, j = np.triu_indices(len(R), 1)
    d = R[j] - R[i]
    r = np.linalg.norm(d, axis=1)
    keep = r < RM
    return i[keep], j[keep], d[keep], r[keep]


def assemble(N, i, j, Hij):
    """4N x 4N real symmetric H, ordered (atom, orbital), from the i < j bond blocks."""
    H4 = np.zeros((N, N, 4, 4))
    H4[np.arange(N), np.arange(N)] = np.diag(ONSITE)
    H4[i, j] = Hij
    H4[j, i] = Hij.transpose(0, 2, 1)   # bond j -> i: sk(o2, o1, -u) = sk(o1, o2, u)
    return H4.transpose(0, 2, 1, 3).reshape(4 * N, 4 * N)


def hamiltonian(R):
    i, j, d, r = pairs(R)
    return assemble(len(R), i, j, sk_blocks(d / r[:, None]) * s_r(r)[0][:, None, None])


def levels(R):
    """One-electron levels (eV) and the HOMO-LUMO gap."""
    eps = np.linalg.eigvalsh(hamiltonian(np.asarray(R, float)))
    n = 2 * len(R)
    return eps, eps[n] - eps[n - 1]


def energy(R):
    """Total energy (eV) of the cluster with positions R, shape (N, 3)."""
    R = np.asarray(R, float)
    N = len(R)
    eps = np.linalg.eigvalsh(hamiltonian(R))
    E_band = 2 * eps[:2 * N].sum()               # 4N electrons fill the lowest 2N levels
    i, j, _, r = pairs(R)
    ph = phi_r(r)[0]
    x = np.bincount(i, ph, N) + np.bincount(j, ph, N)
    return E_band + f_rep(x)[0].sum()


def rep_x(R):
    """x_i = sum_j phi(r_ij), the argument of f for each atom of a cluster."""
    i, j, _, r = pairs(np.asarray(R, float))
    ph = phi_r(r)[0]
    return np.bincount(i, ph, len(R)) + np.bincount(j, ph, len(R))


def sk_grad(u):
    """dB[P, beta, a, b] = d B_ab / d u_beta for the sk_blocks entries (u treated as free).
    B_s,pa = u_a V_sp;  B_pa,s = -u_a V_sp;  B_pa,pc = u_a u_c (V_pp_sigma - V_pp_pi) + ..."""
    dB = np.zeros((len(u), 3, 4, 4))
    Vsp, dV = V0["sp_sigma"], V0["pp_sigma"] - V0["pp_pi"]
    for b in range(3):
        dB[:, b, 0, 1 + b] = Vsp
        dB[:, b, 1 + b, 0] = -Vsp
        dB[:, b, 1 + b, 1:] += dV * u      # d(u_a u_c)/du_b from a = b
        dB[:, b, 1:, 1 + b] += dV * u      # and from c = b
    return dB


def energy_forces(R):
    """Total energy (eV) and forces F = -dE/dR (eV/A), shape (N, 3).

    Band part by Hellmann-Feynman: dE_band = 2 Tr(rho dH), rho = sum_occ |psi><psi|.
    Each bond block X(d) = B(u) s(r) sits at (i, j) and, transposed, at (j, i), so
    dE_band/dd = 4 sum_ab G_ab dX_ab/dd with G = rho[i, :, j, :].
    dX/dd = s(r) dB/du . (1 - u u)/r + B s'(r) u.
    Repulsive part: dE_rep/dd = [f'(x_i) + f'(x_j)] phi'(r) u.
    Valid while the HOMO-LUMO gap is open (integer occupations)."""
    R = np.asarray(R, float)
    N = len(R)
    i, j, d, r = pairs(R)
    u = d / r[:, None]
    B = sk_blocks(u)
    s, ds = s_r(r)
    eps, psi = np.linalg.eigh(assemble(N, i, j, B * s[:, None, None]))
    occ = psi[:, :2 * N]
    G = (occ @ occ.T).reshape(N, 4, N, 4)[i, :, j, :]           # (P, 4, 4)
    W = np.einsum("pab,pkab->pk", G, sk_grad(u))
    GB = np.einsum("pab,pab->p", G, B)
    dEdd = 4 * (s[:, None] / r[:, None] * (W - u * (u * W).sum(1)[:, None])
                + (ds * GB)[:, None] * u)

    ph, dph = phi_r(r)
    x = np.bincount(i, ph, N) + np.bincount(j, ph, N)
    fx, fp = f_rep(x)
    dEdd += ((fp[i] + fp[j]) * dph)[:, None] * u

    F = np.zeros((N, 3))                    # d = R_j - R_i: dE/dR_i = -dE/dd
    np.add.at(F, i, dEdd)
    np.add.at(F, j, -dEdd)
    return 2 * eps[:2 * N].sum() + fx.sum(), F


# ---------------------------------------------------------------------------
# Crystal (Bloch H(k), following Model.H and Model.mesh in tb.py)
# ---------------------------------------------------------------------------
def periodic_terms(lattice, basis):
    """Every ordered pair (i, j) and translation T with 0 < |T + tau_j - tau_i| < RM.
    Returns i, j, d, r."""
    A = np.asarray(lattice, float)
    vol = abs(np.linalg.det(A))
    heights = [vol / np.linalg.norm(np.cross(A[(k + 1) % 3], A[(k + 2) % 3])) for k in range(3)]
    span = np.ptp(basis, axis=0).max()
    nmax = [int(np.ceil((RM + span) / h)) for h in heights]
    T = np.array(list(itertools.product(*[range(-n, n + 1) for n in nmax]))) @ A
    nat = len(basis)
    i, j = np.repeat(np.arange(nat), nat), np.tile(np.arange(nat), nat)
    d = (T[None, :, :] + basis[j][:, None, :] - basis[i][:, None, :]).reshape(-1, 3)
    i, j = np.repeat(i, len(T)), np.repeat(j, len(T))
    r = np.linalg.norm(d, axis=1)
    keep = (r > 1e-8) & (r < RM)
    return i[keep], j[keep], d[keep], r[keep]


def kmesh_points(lattice, kmesh):
    """Gamma-centered mesh, kmesh[a] points along reciprocal vector b_a."""
    B = 2 * np.pi * np.linalg.inv(np.asarray(lattice, float)).T
    f = np.array(list(itertools.product(*[np.arange(n) / n for n in kmesh])))
    return f @ B


def energy_periodic(lattice, basis, kmesh):
    """Total energy (eV) per cell. lattice rows are the lattice vectors, basis is (nat, 3)
    Cartesian, kmesh = (n1, n2, n3) with n = 1 along a vacuum direction."""
    basis = np.asarray(basis, float)
    nat = len(basis)
    i, j, d, r = periodic_terms(lattice, basis)
    B = sk_blocks(d / r[:, None]) * s_r(r)[0][:, None, None]
    k = kmesh_points(lattice, kmesh)
    ph = np.exp(1j * k @ d.T)                                    # (Nk, Nterms)
    H4 = np.zeros((len(k), nat, nat, 4, 4), complex)
    for a in range(nat):
        H4[:, a, a] = np.diag(ONSITE)
        for b in range(nat):
            m = (i == a) & (j == b)
            H4[:, a, b] += np.einsum("kt,tpq->kpq", ph[:, m], B[m])
    H = H4.transpose(0, 1, 3, 2, 4).reshape(len(k), 4 * nat, 4 * nat)
    eps = np.linalg.eigvalsh(H)
    E_band = 2 * eps[:, :2 * nat].sum(axis=1).mean()
    x = np.bincount(i, phi_r(r)[0], nat)                      # each ordered pair once
    return E_band + f_rep(x)[0].sum()
