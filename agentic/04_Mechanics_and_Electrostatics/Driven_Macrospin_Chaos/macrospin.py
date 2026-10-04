"""Landau-Lifshitz macrospin: the right-hand side and two integrators.
Units: ns, T, gamma/2pi = 28.0 GHz/T (g = 2), |m| = 1.

    dm/dt = -(g/(1+a^2)) m x H - (a g/(1+a^2)) m x (m x H)  ==  m x Omega,
            Omega = -(g/(1+a^2)) (H + a m x H)

is an identity for any H and any alpha, damping included. The flow is therefore a
rotation at every instant and |m| = 1 is structural, not a lucky parameter choice.
An integrator either inherits that or it does not; Phase 0 tells the two apart.
"""
import numpy as np

GAMMA = 2 * np.pi * 28.0                 # rad / (ns T)
EX = np.array([1.0, 0.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])

def cross(a, b):                         # 10x faster than np.cross on 3-vectors
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]))

def field(h_dc, theta_dc, h_k, h_rf, f_rf):
    """h_eff(m, t) in T: Zeeman along a static field tilted theta_dc degrees off the
    easy axis, uniaxial anisotropy h_k along z, linearly polarized RF along x."""
    th = np.radians(theta_dc)
    h0 = h_dc * np.array([np.sin(th), 0.0, np.cos(th)])
    w_rf = 2 * np.pi * f_rf

    def h_eff(m, t):
        return h0 + h_k * m[2] * EZ + h_rf * np.cos(w_rf * t) * EX
    return h_eff

def omega(m, t, h_eff, alpha):
    """The rotation generator Omega, as in the module docstring."""
    h = h_eff(m, t)
    return -(GAMMA / (1.0 + alpha ** 2)) * (h + alpha * cross(m, h))

def rhs(m, t, h_eff, alpha):
    return cross(m, omega(m, t, h_eff, alpha))                        # dm/dt in 1/ns

def step_rk4(m, t, dt, h_eff, alpha):
    """Classic RK4: fourth order, and nothing in it knows that |m| = 1."""
    k1 = rhs(m, t, h_eff, alpha)
    k2 = rhs(m + 0.5 * dt * k1, t + 0.5 * dt, h_eff, alpha)
    k3 = rhs(m + 0.5 * dt * k2, t + 0.5 * dt, h_eff, alpha)
    k4 = rhs(m + dt * k3, t + dt, h_eff, alpha)
    return m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rotate(m, w, dt):
    """Exact flow of dm/dt = m x w at fixed w, by Rodrigues: orthogonal by construction,
    so it cannot touch |m| except through roundoff. np.sinc keeps |w| dt -> 0 exact."""
    a = -w * dt                                       # m x w = (-w) x m: axis is -w
    th = np.sqrt(a @ a)
    return (np.cos(th) * m
            + np.sinc(th / np.pi) * cross(a, m)
            + 0.5 * np.sinc(th / (2 * np.pi)) ** 2 * (a @ m) * a)

def step_rot(m, t, dt, h_eff, alpha):
    """Midpoint rotation: half a step to locate Omega at the midpoint, then one exact
    rotation through it. Second order, and |m| is a property of the map itself."""
    half = rotate(m, omega(m, t, h_eff, alpha), 0.5 * dt)
    return rotate(m, omega(half, t + 0.5 * dt, h_eff, alpha), dt)

def run(step, m0, dt, nsteps, h_eff, alpha, stride):
    """Integrate nsteps and return (t, m) sampled every stride steps."""
    m = np.array(m0, float)
    out = np.empty((nsteps // stride + 1, 3))
    out[0] = m
    for i in range(1, nsteps + 1):
        m = step(m, (i - 1) * dt, dt, h_eff, alpha)
        if i % stride == 0:
            out[i // stride] = m
    return dt * stride * np.arange(len(out)), out
