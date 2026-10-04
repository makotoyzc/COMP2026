"""Solving for the dynamics: the magnetization itself, before any chaos.
Run: python dynamics.py   (writes dynamics.npz, which viz3d.py turns into HTML)

1. the damped cone against a closed-form solution. This is the M1 gate: the damping
   term's sign and size are both in one line of algebra, and a flipped sign opens the
   cone instead of closing it;
2. the equilibrium, and the small-signal FMR frequency it rings at;
3. three driven trajectories on the sphere, weak to strong.
"""
import numpy as np
from macrospin import GAMMA, field, rhs, run, step_rot

H_DC, THETA, H_K, ALPHA = 0.1, 45.0, 0.05, 0.01   # T, deg, T, - (settled in PLAN.md)
PER = 200                                         # steps per precession period
DT = 1.0 / (28.0 * H_DC * PER)                    # ns
EY = np.array([0.0, 1.0, 0.0])
report = lambda n, ok, d: print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")

h_zee = field(H_DC, THETA, h_k=0.0, h_rf=0.0, f_rf=0.0)
hhat = h_zee(np.zeros(3), 0.0) / H_DC
e1 = np.cross(EY, hhat)                           # unit, perpendicular to H
e2 = np.cross(hhat, e1)

# ------------------------------------------ 1. the damped cone, solved in closed form
# Pure Zeeman: the damping torque is purely meridional, so the polar angle about H obeys
#   dtheta/dt = -k sin(theta),  k = alpha gamma H / (1 + alpha^2)
#   -> tan(theta/2) = tan(theta0/2) exp(-k t),
# while the azimuth advances at exactly gamma H / (1 + alpha^2). Both are exact.
th0 = np.radians(60.0)
m_cone = np.cos(th0) * hhat + np.sin(th0) * e1
for alpha in (ALPHA, 0.1):
    k = alpha * GAMMA * H_DC / (1 + alpha ** 2)
    t, m = run(step_rot, m_cone, DT, int(3 / (k * DT)), h_zee, alpha, PER // 8)
    th = np.arccos(np.clip(m @ hhat, -1.0, 1.0))
    err = np.abs(th - 2 * np.arctan(np.tan(th0 / 2) * np.exp(-k * t))).max()
    w = abs(np.polyfit(t, np.unwrap(np.arctan2(m @ e2, m @ e1)), 1)[0])
    report(f'cone closes exactly as the closed form says (alpha = {alpha})', err < 1e-4,
           f'max |theta - theta_exact| = {err:.1e} rad over {t[-1]:.1f} ns, '
           f'theta {np.degrees(th[0]):.1f} -> {np.degrees(th[-1]):.2f} deg')
    report(f'azimuth advances at gamma H / (1 + alpha^2) (alpha = {alpha})',
           abs(w / (GAMMA * H_DC / (1 + alpha ** 2)) - 1) < 1e-5,
           f'{w:.6f} vs {GAMMA * H_DC / (1 + alpha ** 2):.6f} rad/ns '
           f'({w / (GAMMA * H_DC / (1 + alpha ** 2)) - 1:+.1e}); damping slows precession')

# ------------------------------------- 2. the equilibrium, and the frequency it rings at
h_free = field(H_DC, THETA, h_k=H_K, h_rf=0.0, f_rf=0.0)
m_eq = run(step_rot, hhat, DT, 20000, h_free, 0.5, 20000)[1][-1]
v_eq = np.linalg.norm(rhs(m_eq, 0.0, h_free, ALPHA))
report('equilibrium reached', v_eq < 1e-9,
       f'|dm/dt| = {v_eq:.1e} /ns at m_eq = ({m_eq[0]:+.4f}, {m_eq[1]:+.4f}, {m_eq[2]:+.4f}), '
       f'{np.degrees(np.arccos(m_eq @ hhat)):.2f} deg off H, pulled toward the easy axis')

kick = m_eq + 0.005 * np.cross(m_eq, hhat)
kick /= np.linalg.norm(kick)
_, m = run(step_rot, kick, DT, 2 ** 16, h_free, 0.0, 1)
amp = np.abs(np.fft.rfft((m[:, 1] - m_eq[1]) * np.hanning(len(m))))
fax = np.fft.rfftfreq(len(m), DT)
i = np.argmax(amp[1:]) + 1
F0 = fax[i] + 0.5 * (amp[i-1] - amp[i+1]) / (amp[i-1] - 2*amp[i] + amp[i+1]) * (fax[1] - fax[0])
print(f'    small-signal FMR frequency f0 = {F0:.4f} GHz, against a bare Larmor '
      f'gamma H / 2pi = {28.0 * H_DC:.3f} GHz -- the anisotropy moves it')

# ------------------------------------------ 3. three driven regimes, weak to strong
NPER, SUB = 400, 200                              # drive periods, steps per drive period
out = {'m_eq': m_eq, 'hhat': hhat, 'f0': F0}
for tag, h_rf in (('weak', 5e-5), ('moderate', 5e-4), ('strong', 5e-3)):
    h = field(H_DC, THETA, h_k=H_K, h_rf=h_rf, f_rf=F0)
    _, m = run(step_rot, m_eq, 1.0 / (F0 * SUB), NPER * SUB, h, ALPHA, 5)
    out[tag] = m
    cone = np.degrees(np.arccos(np.clip(m[len(m) // 2:] @ m_eq, -1.0, 1.0)))
    strobe = m[len(m) // 2::SUB // 5]             # Poincare section, steady state only
    print(f'    {tag:8s} h_rf = {h_rf * 1e3:5.2f} mT -> cone about m_eq reaches '
          f'{cone.max():6.2f} deg; {len(strobe)} strobe points collapse to '
          f'{np.linalg.norm(strobe - strobe.mean(0), axis=1).max():.1e} -> period 1')

# ------------------------------- M1 frames: m and the two Landau-Lifshitz torques
_, m_m1 = run(step_rot, m_cone, DT, 2000, h_zee, ALPHA, 12)
hv = H_DC * hhat
out['m1_m'] = m_m1
out['m1_tp'] = -(GAMMA / (1 + ALPHA ** 2)) * np.cross(m_m1, hv)
out['m1_td'] = -(ALPHA * GAMMA / (1 + ALPHA ** 2)) * np.cross(m_m1, np.cross(m_m1, hv))
print(f'    M1: {len(m_m1)} frames, |T_damp|/|T_prec| = '
      f'{np.linalg.norm(out["m1_td"][0]) / np.linalg.norm(out["m1_tp"][0]):.4f} = alpha')
np.savez('dynamics.npz', **out)
print('    wrote dynamics.npz')
