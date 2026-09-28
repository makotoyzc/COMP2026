"""Both sides of the residue theorem, built from machinery that shares nothing.

    LHS  = contour integral of f(z) dz        -> numpy quadrature, float64
    RHS  = 2 pi i  sum_k  n(C, p_k) Res f(p_k) -> sympy finds the poles exactly,
                                                 residues come from Laurent series
                                                 (derivatives + series division),
                                                 winding numbers from counting
                                                 ray crossings (integers only).

Nothing on the RHS ever integrates anything.
"""
import numpy as np
import sympy as sp
import mpmath as mp

z = sp.symbols('z')
mp.mp.dps = 50


# ---------------------------------------------------------------- parsing

def parse(s):
    """String -> sympy expression in z. Floats become exact rationals, and
    tan/cot/sec/csc (and hyperbolic versions) are written as ratios, so every
    pole shows up as a zero of a denominator."""
    f = sp.sympify(s, locals={'z': z, 'i': sp.I, 'j': sp.I, 'e': sp.E})
    f = sp.nsimplify(f, rational=True)
    ratios = {sp.tan: lambda u: sp.sin(u) / sp.cos(u), sp.cot: lambda u: sp.cos(u) / sp.sin(u),
              sp.sec: lambda u: 1 / sp.cos(u), sp.csc: lambda u: 1 / sp.sin(u),
              sp.tanh: lambda u: sp.sinh(u) / sp.cosh(u), sp.coth: lambda u: sp.cosh(u) / sp.sinh(u),
              sp.sech: lambda u: 1 / sp.cosh(u), sp.csch: lambda u: 1 / sp.sinh(u)}
    for fn, rule in ratios.items():
        f = f.replace(fn, rule)
    return f


ENTIRE = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)


def why_not_meromorphic(f):
    """Reason the pole finder below cannot be trusted on f, or None. The pole
    finder only sees zeros of denominators; anything else is refused, loudly."""
    for g in f.atoms(sp.Function, sp.Pow):
        if not g.has(z):
            continue
        if isinstance(g, sp.Pow):
            if g.base.has(z) and (g.exp.has(z) or not g.exp.is_integer):
                return f'branch point: {g}'
            if g.exp.has(z) and sp.fraction(sp.together(g.exp))[1].has(z):
                return f'essential singularity: {g}'
        elif not isinstance(g, ENTIRE):
            return f'not meromorphic (or unsupported): {g}'
        elif sp.fraction(sp.together(g.args[0]))[1].has(z):
            return f'essential singularity: {g}'
    return None


def numeric(f):
    """sympy expression -> vectorized numpy function (principal branches)."""
    g = sp.lambdify(z, f, 'numpy')
    return lambda w: np.broadcast_to(g(w), np.shape(w)).astype(complex)


# ---------------------------------------------------------------- LHS

def lhs_param(f, gamma, dgamma, n):
    """Trapezoid rule for the integral of f(z) dz along z = gamma(t), t in [0, 2 pi).
    The integrand is periodic and analytic, so the error falls geometrically in n."""
    t = 2 * np.pi * np.arange(n) / n
    return 2 * np.pi / n * np.sum(f(gamma(t)) * dgamma(t))


def lhs_polygon(f, verts, n=32):
    """Gauss-Legendre with n nodes on every edge of the closed polygon `verts`."""
    x, w = np.polynomial.legendre.leggauss(n)
    a = np.asarray(verts, complex)
    b = np.roll(a, -1)
    mid, half = (a + b)[:, None] / 2, (b - a)[:, None] / 2
    return np.sum(f(mid + half * x) * w * half)


# ---------------------------------------------------------------- winding numbers

def winding(verts, p):
    """Winding number of the closed polygon about p, by signed crossings of the
    ray from p to +infinity (Sunday's algorithm). Pure integer bookkeeping."""
    a = np.asarray(verts, complex) - p
    b = np.roll(a, -1)
    left = a.real * b.imag - a.imag * b.real          # > 0 when p is left of edge a->b
    up = (a.imag <= 0) & (b.imag > 0) & (left > 0)
    down = (a.imag > 0) & (b.imag <= 0) & (left < 0)
    return int(up.sum() - down.sum())


def winding_by_angle(verts, p):
    """Independent cross-check: total turning angle of (z - p) around the polygon."""
    a = np.asarray(verts, complex) - p
    return np.sum(np.angle(np.roll(a, -1) / a)) / (2 * np.pi)


def distance_to_polygon(verts, p):
    a = np.asarray(verts, complex)
    b = np.roll(a, -1)
    s = np.clip(((p - a) * np.conj(b - a)).real / np.abs(b - a) ** 2, 0, 1)
    return np.min(np.abs(a + s * (b - a) - p))


# ---------------------------------------------------------------- poles

def _zeros_of_factor(fac, box):
    """Exact zeros of one irreducible factor of the denominator, inside box."""
    if fac.is_polynomial(z):
        P = sp.Poly(fac, z)
        if P.degree() <= 4:                  # closed-form roots, any coefficients
            roots = sp.roots(P)
            assert sum(roots.values()) == P.degree(), f'sympy missed roots of {fac}'
            return list(roots)
        if not all(c.is_real for c in P.all_coeffs()):
            # CRootOf needs real coefficients: use P * conj(P). The extra roots are
            # zeros of conj(P), not of P, and the order test below throws them out.
            P = P * sp.Poly(sp.conjugate(P.as_expr()).subs(sp.conjugate(z), z), z)
        return list(P.all_roots())
    return _enumerate(sp.solveset(fac, z, sp.S.Complexes), box)


def _enumerate(sol, box):
    """List the members of a solveset answer (finite sets, and periodic families
    a*n + b over the integers) that fall inside box = (xmin, xmax, ymin, ymax)."""
    if isinstance(sol, sp.FiniteSet):
        return list(sol)
    if isinstance(sol, sp.Union):
        return [p for s in sol.args for p in _enumerate(s, box)]
    if isinstance(sol, sp.ImageSet) and sol.base_sets == (sp.S.Integers,):
        n, = sol.lamda.variables
        a, b = complex(sol.lamda.expr.diff(n)), complex(sol.lamda.expr.subs(n, 0))
        if sol.lamda.expr.diff(n, 2) != 0:
            raise NotImplementedError(f'non-linear family of zeros: {sol}')
        lo, hi = -np.inf, np.inf
        for ak, bk, (cmin, cmax) in [(a.real, b.real, box[:2]), (a.imag, b.imag, box[2:])]:
            if ak == 0:
                if not cmin <= bk <= cmax:
                    return []
                continue
            n1, n2 = sorted([(cmin - bk) / ak, (cmax - bk) / ak])
            lo, hi = max(lo, n1), min(hi, n2)
        return [sol.lamda.expr.subs(n, k) for k in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
    raise NotImplementedError(f'cannot enumerate the zeros {sol}')


def _mp(p, dps=None):
    dps = dps or mp.mp.dps
    c = sp.N(p, dps + 10)
    return mp.mpc(str(sp.re(c)), str(sp.im(c)))


_I = sp.Dummy('I')


class _Taylor:
    """Taylor coefficients g^(k)(p) / k! of an expression, in mpmath arithmetic.
    I is passed in as an argument: lambdify would otherwise print it as the
    float64 constant 1j and quietly cap everything at 16 digits."""
    def __init__(self, g):
        self.derivs, self.funcs = [g], []

    def coeff(self, k, p):
        while len(self.funcs) <= k:
            if len(self.derivs) == len(self.funcs):
                self.derivs.append(self.derivs[-1].diff(z))
            g = self.derivs[len(self.funcs)].subs(sp.I, _I)
            self.funcs.append(sp.lambdify((z, _I), g, 'mpmath'))
        return self.funcs[k](p, mp.mpc(0, 1)) / mp.factorial(k)

    def order(self, p, kmax=30):
        """Order of the zero at the exact point p, working at 90 digits. Coefficient k
        counts as zero when it is > 1e40 smaller at p than at a point eta = 1e-3 away:
        near a zero of order m it grows like eta^(m-k), while at p only rounding
        noise (~1e-90) is left. A genuine value barely changes over eta.
        (A fixed threshold fails: coefficients here range from 1e-130 to 1e+22.)"""
        with mp.workdps(90):
            pm = _mp(p, 90)
            eta = mp.mpf('1e-3') * (1 + abs(pm)) * mp.expjpi(mp.mpf('0.2371'))
            for k in range(kmax):
                if abs(self.coeff(k, pm)) > mp.mpf('1e-40') * abs(self.coeff(k, pm + eta)):
                    return k
        raise ValueError(f'expression vanishes to order >= {kmax} at {p}')


def poles(f, box):
    """All poles of f inside box = (xmin, xmax, ymin, ymax).
    Returns a list of (location: complex, order: int, residue: complex).

    The residue is read off the Laurent series: with h = z - p,
        f = N/D = h^(-m) * A(h)/B(h),   A, B = Taylor series of N, D past their zeros,
    and Res = coefficient of h^(m-1) in A/B, found by power-series division."""
    N, D = sp.fraction(sp.together(f))
    TN, TD = _Taylor(N), _Taylor(D)
    found = []
    for fac, _ in sp.factor_list(D)[1]:
        for p in _zeros_of_factor(fac, box):
            pm = _mp(p)
            if not (box[0] <= pm.real <= box[1] and box[2] <= pm.imag <= box[3]):
                continue
            if any(abs(pm - q) < 1e-25 for q, _, _ in found):
                continue
            kD = TD.order(p)
            kN = TN.order(p)
            m = kD - kN
            if m <= 0:                     # not a zero of D, or a removable singularity
                continue
            A = [TN.coeff(kN + k, pm) for k in range(m)]
            B = [TD.coeff(kD + k, pm) for k in range(m)]
            C = []
            for k in range(m):
                C.append((A[k] - sum(B[j] * C[k - j] for j in range(1, k + 1))) / B[0])
            found.append((pm, m, C[m - 1]))
    return [(complex(p), m, complex(r)) for p, m, r in found]


# ---------------------------------------------------------------- RHS

def bbox(verts, pad=0.0):
    v = np.asarray(verts, complex)
    return (v.real.min() - pad, v.real.max() + pad, v.imag.min() - pad, v.imag.max() + pad)


def rhs(f, verts, min_dist=1e-8):
    """2 pi i sum n(C, p) Res_p f over the closed polygon `verts`.
    Returns (value, [(pole, order, residue, winding), ...])."""
    reason = why_not_meromorphic(f)
    if reason:
        raise ValueError(f'residue theorem does not apply -- {reason}')
    table = []
    for p, m, r in poles(f, bbox(verts)):
        if distance_to_polygon(verts, p) < min_dist:
            raise ValueError(f'pole at {p:.6g} lies on the contour: the theorem does not apply')
        table.append((p, m, r, winding(verts, p)))
    return 2j * np.pi * sum(r * n for _, _, r, n in table), table
