"""Exact symbolic theta/phi derivatives of the FMR free energy.

Nested finite differences (FD Hessian inside a Newton solve inside an FD outer
Jacobian) lose too much precision and stall the parameter fit. sympy generates the
gradient and Hessian once at import; everything downstream is then exact to machine
precision and smooth in the parameters.
"""
import numpy as np, sympy as sp

th, ph, H, thH, phH = sp.symbols("th ph H thH phH", real=True)
Hc1, Hc2, Hu2, Hip, Meff = sp.symbols("Hc1 Hc2 Hu2 Hip Meff", real=True)

m = sp.Matrix([sp.sin(th)*sp.cos(ph), sp.sin(th)*sp.sin(ph), sp.cos(th)])
hv = sp.Matrix([sp.sin(thH)*sp.cos(phH), sp.sin(thH)*sp.sin(phH), sp.cos(thH)])

_s3, _s2 = sp.sqrt(3), sp.sqrt(2)
R111 = sp.Matrix([[1/_s2, -1/_s2, 0],
                  [1/sp.sqrt(6), 1/sp.sqrt(6), -2/sp.sqrt(6)],
                  [1/_s3, 1/_s3, 1/_s3]])

def build(normal):
    a = m if normal == "001" else R111.T * m
    a2 = [a[i]**2 for i in range(3)]
    s2 = a2[0]*a2[1] + a2[1]*a2[2] + a2[2]*a2[0]
    s3 = a2[0]*a2[1]*a2[2]
    f = (-H*(m.dot(hv)) + sp.Rational(1, 2)*Meff*m[2]**2
         + Hc1*s2 + Hc2*s3 + Hu2*m[2]**4 + Hip*(m[0]**2 - m[1]**2))
    args = (th, ph, H, thH, phH, Hc1, Hc2, Hu2, Hip, Meff)
    d = {"f": f, "ft": sp.diff(f, th), "fp": sp.diff(f, ph),
         "ftt": sp.diff(f, th, 2), "fpp": sp.diff(f, ph, 2),
         "ftp": sp.diff(f, th, ph)}
    return {k: sp.lambdify(args, v, "numpy") for k, v in d.items()}

FUNCS = {"001": build("001"), "111": build("111")}
