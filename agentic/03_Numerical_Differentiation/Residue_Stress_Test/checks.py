"""Stress tests of the residue "conjecture". Run:  python checks.py
Prints a pass/fail line per check and writes figures/*.png."""
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from residue import parse, numeric, lhs_param, lhs_polygon, rhs, poles, winding, winding_by_angle

rng = np.random.default_rng(7326)
T = 2 * np.pi * np.arange(4000) / 4000          # polyline samples for winding numbers


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def circle(R, c=0):
    return (lambda t: c + R * np.exp(1j * t)), (lambda t: 1j * R * np.exp(1j * t))


def compare(s, gamma, dgamma, n=2048):
    f = parse(s)
    L = lhs_param(numeric(f), gamma, dgamma, n)
    R, table = rhs(f, gamma(T))
    return L, R, table


# ------------------------------------------------------------ 1. residues known by hand
def check_hand():
    cases = [  # f, contour, exact answer (worked out on paper)
        ('1/z', circle(1), 2j * np.pi),
        ('1/(z**2+1)', circle(1, 1j), np.pi),                       # only +i inside: Res = 1/(2i)
        ('exp(z)/z**3', circle(1), 1j * np.pi),                     # Res = 1/2!
        ('1/(z**2+1)**4', circle(1, 1j), 2j * np.pi * (-5j / 32)),  # 4th-order pole
        ('tan(z)', circle(2), -4j * np.pi),                         # two poles, Res = -1 each
        ('1/sin(z)**2', circle(1), 0),                              # double pole, zero residue
        ('pi*cot(pi*z)/z**2', circle(0.5), 2j * np.pi * (-np.pi ** 2 / 3)),
    ]
    worst = 0
    for s, (g, dg), exact in cases:
        L, R, _ = compare(s, g, dg)
        worst = max(worst, abs(L - exact), abs(R - exact))
        print(f'    {s:22s} LHS={L:+.10f}  RHS={R:+.10f}  exact={complex(exact):+.10f}')
    report('hand-computed residues', worst < 1e-12, f'worst |side - exact| = {worst:.1e}')


# ------------------------------------------------------------ 2. winding numbers
def limacon(t):     # r = 1/2 + cos t: an inner loop, so points in it are wound twice
    return (0.5 + np.cos(t)) * np.exp(1j * t)


def dlimacon(t):
    return (-np.sin(t) + 1j * (0.5 + np.cos(t))) * np.exp(1j * t)


def figure8(t):     # lemniscate of Gerono: +1 around one lobe, -1 around the other
    return np.sin(t) + 0.5j * np.sin(2 * t)


def dfigure8(t):
    return np.cos(t) + 1j * np.cos(2 * t)


def check_winding():
    grid = [x + 1j * y for x in np.linspace(-1.2, 1.7, 41) for y in np.linspace(-1.2, 1.2, 41)]
    bad = 0
    for curve in (limacon, figure8):
        v = curve(T)
        for p in grid:
            ang = winding_by_angle(v, p)
            if abs(ang - round(ang)) < 1e-6 and winding(v, p) != round(ang):
                bad += 1
    report('crossing count == turning angle', bad == 0, f'{bad} disagreements on {2 * len(grid)} grid points')

    cases = [  # poles planted in regions with winding 2, 1, 0 (limacon) and +1, -1 (figure 8)
        ('1/(z-0.2) + 2/(z-1.2) + 5/(z+0.8)', limacon, dlimacon, 2 * 1 + 1 * 2 + 0 * 5),
        ('1/(z-0.5)**2 + 3/(z-0.5) + 1j/(z+0.5)', figure8, dfigure8, None),
    ]
    for s, g, dg, expect in cases:
        L, R, table = compare(s, g, dg, n=4096)
        ns = [n for *_, n in table]
        print(f'    {s:40s} windings {ns}  LHS={L:+.10f}  RHS={R:+.10f}')
        ok = abs(L - R) < 1e-10 and (expect is None or abs(R - 2j * np.pi * expect) < 1e-10)
        report(f'multi-winding contour ({g.__name__})', ok, f'|LHS-RHS| = {abs(L - R):.1e}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    X, Y = np.meshgrid(np.linspace(-1.2, 1.7, 300), np.linspace(-1.2, 1.2, 250))
    for ax, curve, s in zip(axes, (limacon, figure8), (cases[0][0], cases[1][0])):
        v = curve(T[::8])
        W = np.vectorize(lambda x, y: winding(v, x + 1j * y))(X, Y)
        im = ax.pcolormesh(X, Y, W, cmap='coolwarm', vmin=-2, vmax=2, shading='auto')
        ax.plot(v.real, v.imag, 'k', lw=1)
        for p, *_ in poles(parse(s), (-2, 2, -2, 2)):
            ax.plot(p.real, p.imag, 'kx', ms=9, mew=2)
        ax.set_aspect('equal')
        ax.set_title(curve.__name__)
    fig.colorbar(im, ax=axes, label='winding number n(C, z)')
    fig.savefig('figures/winding.png', dpi=130, bbox_inches='tight')


# ------------------------------------------------------------ 3. infinitely many poles: Basel
def square(N, per_unit=1):
    """Square with corners (N+1/2)(+-1 +-i), vertices every unit so no edge is long."""
    a = N + 0.5
    s = np.linspace(-a, a, int(2 * a * per_unit) + 1)[:-1]
    return np.concatenate([s - 1j * a, a + 1j * s, -s + 1j * a, -a - 1j * s])


def check_basel():
    f = parse('pi*cot(pi*z)/z**2')
    fn = numeric(f)
    rows = []
    for N in (1, 2, 5, 10, 20, 40):
        v = square(N)
        L = lhs_polygon(fn, v, n=24)
        R, table = rhs(f, v)
        partial = (L / (2j * np.pi) + np.pi ** 2 / 3) / 2     # what the LHS says sum 1/n^2 is
        rows.append((N, len(table), abs(L - R), partial))
        print(f'    N={N:3d}  poles={len(table):3d}  |LHS-RHS|={abs(L - R):.1e}  '
              f'LHS -> sum_(n<=N) 1/n^2 = {partial.real:.10f}  (pi^2/6 = {np.pi ** 2 / 6:.10f})')
    worst = max(r[2] for r in rows)
    report('81 poles of cot(pi z)/z^2 inside a square', worst < 1e-10, f'worst |LHS-RHS| = {worst:.1e}')
    exact_partial = [sum(1 / k ** 2 for k in range(1, N + 1)) for N, *_ in rows]
    report('LHS alone reproduces the partial sums of 1/n^2', max(abs(r[3] - e) for r, e in zip(rows, exact_partial)) < 1e-10,
           'so the contour integral knows the Basel sum')


# ------------------------------------------------------------ 4. quadrature error: theory vs measured
def check_convergence():
    """Trapezoid on the unit circle, n nodes. Geometric series over the roots of unity give
    exactly  LHS_n = 2 pi i / (1 - a^n)  for 1/(z-a), |a|<1, and  -2 pi i b^-n / (1 - b^-n)
    for 1/(z-b), |b|>1. So the error is predicted to the last digit, not just in scaling."""
    a, b = 0.6 * np.exp(0.7j), 1.25 * np.exp(-2.0j)
    f = numeric(parse(f'1/(z-({a.real}+{a.imag}*I)) + 1/(z-({b.real}+{b.imag}*I))'))
    g, dg = circle(1)
    ns = np.arange(4, 161, 4)
    err = np.array([abs(lhs_param(f, g, dg, n) - 2j * np.pi) for n in ns])
    pred = np.abs(2j * np.pi / (1 - a ** ns) - 2j * np.pi - 2j * np.pi * b ** -ns / (1 - b ** -ns))
    gap = np.max(np.abs(err - pred))      # errors span 1e0 .. 1e-16, so compare absolutely
    report('trapezoid error == closed-form prediction', gap < 1e-13,
           f'max |measured - predicted| = {gap:.1e} while the error itself spans {err.max():.0e} .. {err.min():.0e}')

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(ns, err, 'o', label='measured |LHS - 2πi|')
    ax.semilogy(ns, pred, '-', label='predicted, exact formula')
    ax.semilogy(ns, 2 * np.pi * (1 / abs(b)) ** ns, '--', label=f'(1/|b|)^n, |b| = {abs(b)}')
    ax.semilogy(ns, 2 * np.pi * abs(a) ** ns, ':', label=f'|a|^n, |a| = {abs(a)}')
    ax.set_xlabel('quadrature nodes n')
    ax.set_ylabel('error')
    ax.set_ylim(1e-17, 10)
    ax.legend()
    fig.savefig('figures/convergence.png', dpi=130, bbox_inches='tight')


# ------------------------------------------------------------ 5. random functions, random contours
def random_case():
    """Planted poles at exact rational points, random orders, random numerator,
    random smooth star-shaped contour. Returns everything needed to check."""
    k = rng.integers(1, 6)
    locs = np.round(rng.uniform(-1.6, 1.6, k) + 1j * rng.uniform(-1.6, 1.6, k), 2)
    orders = rng.integers(1, 4, k)
    num = sum(int(c) * sp.Symbol('z') ** j for j, c in enumerate(rng.integers(-3, 4, rng.integers(1, 4))))
    num = num if num != 0 else sp.Integer(1)
    den = ' * '.join(f'(z - ({sp.nsimplify(p.real)} + {sp.nsimplify(p.imag)}*I))**{m}' for p, m in zip(locs, orders))
    s = f'({num}) / ({den})'
    c = 0.3 * (rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1))
    R0, eps, phi = rng.uniform(0.7, 1.6), rng.uniform(0, 0.12, 3), rng.uniform(0, 2 * np.pi, 3)
    ks = np.array([2, 3, 5])
    r = lambda t: R0 * (1 + np.sum(eps[:, None] * np.cos(ks[:, None] * t + phi[:, None]), 0))
    dr = lambda t: -R0 * np.sum(eps[:, None] * ks[:, None] * np.sin(ks[:, None] * t + phi[:, None]), 0)
    g = lambda t: c + r(t) * np.exp(1j * t)
    dg = lambda t: (dr(t) + 1j * r(t)) * np.exp(1j * t)
    return s, locs, orders, g, dg


def check_random(trials=300):
    dist, err, planted_ok = [], [], 0
    for _ in range(trials):
        s, locs, orders, g, dg = random_case()
        f = parse(s)
        found = sorted((round(p.real, 6), round(p.imag, 6), m) for p, m, _ in poles(f, (-3, 3, -3, 3)))
        planted = sorted((round(p.real, 6), round(p.imag, 6), int(m)) for p, m in zip(locs, orders))
        planted_ok += found == planted
        L = lhs_param(numeric(f), g, dg, 1024)
        R, table = rhs(f, g(T))
        scale = max(1.0, sum(abs(2 * np.pi * r * n) for _, _, r, n in table))
        v = g(np.linspace(0, 2 * np.pi, 20000))
        dist.append(min(np.min(np.abs(v - p)) for p in locs))
        err.append(abs(L - R) / scale)
    dist, err = np.array(dist), np.array(err)
    report('pole finder recovers the planted poles and orders', planted_ok == trials, f'{planted_ok}/{trials}')
    far = dist > 0.1
    report('random functions, poles >= 0.1 from contour', err[far].max() < 1e-10,
           f'{far.sum()} cases, worst relative |LHS-RHS| = {err[far].max():.1e}')
    print(f'    closer than 0.1: {(~far).sum()} cases, worst {err[~far].max():.1e} '
          '(quadrature needs more nodes, see figure)')

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(dist, np.maximum(err, 1e-17), '.', alpha=0.6)
    ax.set_xlabel('distance from nearest pole to contour')
    ax.set_ylabel('relative |LHS - RHS|  (n = 1024)')
    ax.set_title(f'{trials} random meromorphic functions and contours')
    fig.savefig('figures/random.png', dpi=130, bbox_inches='tight')


# ------------------------------------------------------------ 6. break the hypotheses on purpose
def check_failures():
    g, dg = circle(1)
    for s in ('sqrt(z)', 'log(z)', 'conjugate(z)', 'exp(1/z)'):
        f = parse(s)
        L = lhs_param(numeric(f), g, dg, 4096)
        try:
            rhs(f, g(T))
            verdict = 'RHS computed (should have refused!)'
        except ValueError as e:
            verdict = str(e).split('-- ')[1]
        print(f'    {s:14s} LHS = {L:+.8f}   RHS refused: {verdict}')
    report('non-meromorphic inputs are refused, not silently summed', 'should' not in verdict, 'see lines above')

    # a pole ON the contour: trapezoid nodes placed symmetrically around it -> principal value,
    # which is HALF the residue contribution: pi i instead of 2 pi i.
    f = numeric(parse('1/(z-1)'))
    for n in (64, 1024, 16384):
        t = 2 * np.pi * (np.arange(n) + 0.5) / n
        L = 2 * np.pi / n * np.sum(f(np.exp(1j * t)) * 1j * np.exp(1j * t))
        print(f'    1/(z-1), pole on |z|=1, n={n:5d}:  LHS = {L:+.10f}   (pi i = {1j * np.pi:+.10f})')
    report('pole on the contour gives the principal value pi*i', abs(L - 1j * np.pi) < 1e-10, 'half of 2 pi i Res')


if __name__ == '__main__':
    for check in (check_hand, check_winding, check_basel, check_convergence, check_random, check_failures):
        print(f'\n== {check.__name__}')
        check()
