"""What local relaxation alone can and cannot do (pictures for Step 3).

Run A: the reference C60, every coordinate shaken by 0.1 A. L-BFGS brings it back.
Run B: the same, shaken by 0.3 A. Bonds break and it relaxes into a defective cage.
Run C: 60 atoms at random in a sphere. L-BFGS finds the nearest local minimum,
       a tangle of chains far above C60. This is why we need a global search.

Usage:  python3 demo_relax.py   -> relax_snapshots.png, relax_random.gif, viewer.html
"""
import numpy as np

import carbon_tb as tb
import structures as st
import viz

rng = np.random.default_rng(30)
ref = np.load("c60_ref.npy")
E_ref = tb.binding(tb.energy(ref), 60)

starts = {
    "A: C60 shaken 0.1 A": ref + rng.normal(0, 0.1, ref.shape),
    "B: C60 shaken 0.3 A": ref + rng.normal(0, 0.3, ref.shape),
    "C: random atoms": st.random_cluster(60, rng, radius=4.6),
}
runs = []
for label, R0 in starts.items():
    R, E, info = st.relax(R0, fmax=1e-4, record=True)
    Es = [tb.binding(tb.energy(p), 60) for p in info["path"]]
    print(f"{label}: {info['nit']} steps, {info['nfev']} calls, converged = {info['converged']},"
          f" max |F| = {info['fmax']:.1e} eV/A")
    print(f"   start: {st.describe(R0)}")
    print(f"   end  : {st.describe(R)}")
    print(f"   end - E_ref = {Es[-1] - E_ref:+.4f} eV/atom")
    runs.append((label, info["path"], Es))

viz.snapshots(runs, "relax_snapshots.png", E_ref)
label, path, Es = runs[2]
viz.movie(path, Es, "relax_random.gif", every=max(1, len(path) // 60))
viz.viewer(runs + [("reference C60 (E_ref)", [ref], [E_ref])], "viewer.html", "L-BFGS step")
