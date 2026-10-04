"""Phase 0 gate. Run: python phase0_gate.py
alpha = 0 in a static field, 10^6 steps: the true flow conserves |m|, the Larmor
frequency and the Zeeman energy exactly, so every deviation belongs to the integrator.
Prints a pass/fail line per check and writes figures/phase0_norm.png.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from macrospin import field, run, step_rk4, step_rot

H_DC, THETA = 0.1, 45.0              # T, deg off the easy axis (settled in PLAN.md)
F_L = 28.0 * H_DC                    # GHz, Larmor frequency
PER = 200                            # steps per precession period
DT = 1.0 / (F_L * PER)               # ns
N, STRIDE = 10 ** 6, 1000            # 5000 periods
M0 = np.array([0.0, 0.0, 1.0])       # 45 deg cone about H
SCHEMES = (('RK4', step_rk4), ('rotation', step_rot))
report = lambda name, ok, detail: print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

h_free = field(H_DC, THETA, h_k=0.0, h_rf=0.0, f_rf=0.0)
hhat = h_free(M0, 0.0) / H_DC

# ------------------------------------------------- 1, 2. the norm over 10^6 steps
long = {name: run(step, M0, DT, N, h_free, 0.0, STRIDE) for name, step in SCHEMES}
drift = {name: np.abs(np.linalg.norm(m, axis=1) - 1.0) for name, (_, m) in long.items()}
n = STRIDE * np.arange(1, N // STRIDE + 1)
slope = {name: np.polyfit(np.log(n[d[1:] > 0]), np.log(d[1:][d[1:] > 0]), 1)[0]
         for name, d in drift.items()}
d_rk4, d_rot = drift['RK4'][-1], drift['rotation'][-1]
theory = 0.5 * N * (2 * np.pi / PER) ** 6 / 144      # |amplification| = 1 - th^6/144
report('RK4 leaks the norm', d_rk4 > 1e-8,
       f'||m|-1| = {d_rk4:.3e} after {N} steps (hand: {theory:.3e}), slope {slope["RK4"]:.2f}')
report('rotation keeps it', d_rot < 1e-12,
       f'||m|-1| = {d_rot:.3e}, slope {slope["rotation"]:.2f} (roundoff, not truncation), '
       f'{d_rk4 / d_rot:.0e}x better')

# ----------------------------------------------------- 3. Larmor frequency, energy
e1 = M0 - (M0 @ hhat) * hhat
e1 /= np.sqrt(e1 @ e1)
e2 = np.cross(hhat, e1)
for name, step in SCHEMES:
    t, m = run(step, M0, DT, PER * 200, h_free, 0.0, 1)
    f = np.polyfit(t, np.unwrap(np.arctan2(m @ e2, m @ e1)), 1)[0] / (2 * np.pi)
    report(f'{name} precesses at the Larmor rate', abs(f / F_L - 1) < 1e-6,
           f'f = {f:.9f} GHz vs gamma H / 2pi = {F_L:.9f} GHz ({f / F_L - 1:+.1e})')
for name, (_, m) in long.items():
    e = -(m @ (H_DC * hhat))
    report(f'{name} conserves the Zeeman energy', abs(e[-1] / e[0] - 1) < 1e-12,
           f'max |dE/E| = {np.abs(e / e[0] - 1).max():.1e} over {N} steps')

# -------------------------------- 4. observed order, anisotropy on so Omega(m) bites
h_an = field(H_DC, THETA, h_k=0.05, h_rf=0.0, f_rf=0.0)
for (name, step), want in zip(SCHEMES, (4.0, 2.0)):
    end = [run(step, [1.0, 0.0, 0.0], 2.0 / k, k, h_an, 0.01, k)[1][-1]
           for k in (500, 1000, 2000, 4000)]
    e = [np.linalg.norm(end[i] - end[i + 1]) for i in range(3)]
    p = [np.log2(e[i] / e[i + 1]) for i in range(2)]
    report(f'{name} converges at order {want:.0f}', abs(p[1] - want) < 0.3,
           f'observed {p[0]:.2f} then {p[1]:.2f}')

a_num = (2 * np.pi / PER) ** 5 / 144
print(f'\n    RK4 at {PER} steps/period acts like a spurious Gilbert damping alpha_num ='
      f' {a_num:.1e}, {0.01 / a_num:.0e}x below the alpha = 0.01 settled for the physics.')

# ------------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for name, (t, m) in long.items():
    ax[0].loglog(n, drift[name][1:], label=name)
    par, nrm = m @ hhat, np.linalg.norm(m, axis=1)
    perp = np.sqrt(np.abs(nrm ** 2 - par ** 2))
    ax[1].semilogy(t, np.abs(par / par[0] - 1) + 1e-18, ls='--', label=f'{name} $m_\\parallel$')
    ax[1].semilogy(t, np.abs(perp / perp[0] - 1) + 1e-18, label=f'{name} $m_\\perp$')
ax[0].loglog(n, d_rk4 * n / n[-1], 'k:', label=r'$\propto n$')
ax[0].set(xlabel='steps', ylabel=r'$|\,|m|-1\,|$', title=r'$\alpha = 0$, static field')
ax[1].set(xlabel='t (ns)', ylabel='relative error', title='the leak is a fake damping')
[a.legend(fontsize=8) for a in ax]
fig.savefig('figures/phase0_norm.png', dpi=130, bbox_inches='tight')
print('    wrote figures/phase0_norm.png')
