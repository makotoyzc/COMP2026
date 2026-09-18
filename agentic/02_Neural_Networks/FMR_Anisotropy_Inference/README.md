# When Is a Neural Network the Wrong Answer?

### Inferring magnetic anisotropy from ferromagnetic resonance

Friday agentic session, 9/18. A self-chosen project, connected to my research group's
work on Er-doped yttrium iron garnet (Er:YIG). See [../../README.md](../../README.md).

## The Problem

My group wants the magnetocrystalline anisotropy constants of Er:YIG at several doping
levels whose magnetic properties are unknown. The measurement available is
angle-resolved ferromagnetic resonance: sweep the applied field until the sample
resonates, repeat at many field angles, record the resonance field.

The literature approach (Tomczak & Puszkarski, *Phys. Rev. B* **98**, 144415 (2018) and
*JMMM* **507**, 166824 (2020); Napierała-Batygolska & Tomczak, *Sci. Rep.* **15**, 11277
(2025)) is advertised as machine learning. Reading the three papers, **none of them
contains a neural network.** They use constrained nonlinear least squares with
leave-one-out cross-validation. That surprised me enough to be worth the session: *when
does a network actually earn its place, and when does it not?*

The answer turns on the fact that here the forward model is known exactly. Magnetization
sits at a minimum of the free energy $f$, and resonance occurs when the Gaussian
curvature of that free-energy surface matches the drive frequency — the Smit-Beljers
condition:

$$\left(\frac{hf_{\text{drive}}}{\mu_B}\right)^{2} = g^{2}\cdot\frac{1}{\sin^{2}\theta}\left(f_{\theta\theta}f_{\phi\phi}-f_{\theta\phi}^{2}\right)$$

subject to $\partial_\theta f = \partial_\phi f = 0$. With an exact forward model you do
not *learn* the map from anisotropy to resonance field — you invert it. A network would
be approximating something you can already write down, from 44 data points, to produce
quantities you need to be physically interpretable. Three strikes.

## Your Checks — decided before writing code

The thing that makes this tractable is that **I don't need the real data to build the
pipeline.** Pick anisotropy constants, run the physics forward, add noise: now I have a
dataset whose answer I already know. Real data cannot do this, because when the fit
returns something odd you cannot tell whether the bug is in your code or the sample.

1. **Round-trip recovery.** Fit the synthetic data starting from $0.7\times$ the planted
   values. Every parameter must come back. This is non-trivial: the inverse problem runs
   through an inner equilibrium solve and a root-find, and nothing guarantees it inverts.
2. **RMS at truth vs RMS at the fit.** Compute the residual at the *planted* parameters
   and at the fitted ones. On noisy data the fit must land slightly *below* truth
   (it absorbs some noise). If it lands above, the optimizer failed — this cleanly
   separates "optimizer broken" from "data underconstrains," which was the single most
   useful diagnostic in the session.
3. **Noise floor.** Parameters smaller than the noise must *not* be recoverable. If they
   come back looking precise, something is wrong.

## What Happened

Two planted-truth sets: a (001) film at 9.46 GHz matching the 2018 paper's magnitudes,
and a (111) film at 8 GHz — the real Er:YIG-on-GGG(111) geometry.

Recovery from $0.7\times$ truth, 44 points, 2 Oe noise:

| param | (001) truth | recovered | err | (111) truth | recovered | abs err |
|---|---|---|---|---|---|---|
| Hc1 | 78.070 | 78.014 | -0.07% | -61.000 | -60.629 | 0.37 Oe |
| Hc2 | -534.000 | -534.639 | +0.12% | 5.000 | 6.237 | 1.24 Oe |
| Hu2 | 43.900 | 43.431 | -1.07% | 18.000 | 17.983 | 0.02 Oe |
| Hip | 66.300 | 65.937 | -0.55% | 6.100 | 5.653 | 0.45 Oe |
| Meff | 4811.000 | 4814.183 | +0.07% | 826.000 | 826.592 | 0.59 Oe |
| g | 1.985 | 1.985 | -0.01% | 2.005 | 2.005 | — |

Check 2 passes: fitted RMS $1.099\times10^{-3}$ sits below truth's $1.240\times10^{-3}$.

Check 3 passes in an instructive way. In the (111) set Hc2 is 25% off — but that is
1.24 Oe on a 5 Oe parameter, *below the 2 Oe noise*. The percentage is misleading and
the absolute error is the honest column. This is exactly the situation cross-validation
exists to catch: held-out error will not improve when you add a term the data cannot
resolve, so CV tells you to drop Hc2 rather than letting you quote 6.2 Oe with a
confident-looking error bar.

## The Part That Actually Cost the Time

**The inner equilibrium solve must be exact and smooth, not merely converged.** The
outer parameter fit takes finite differences *through* that inner solve, so imprecision
there destroys the outer Jacobian. Three attempts:

1. Nelder-Mead inner solve → outer Jacobian is noise, fit never leaves its initial
   guess. Every parameter returned exactly $0.7\times$ truth.
2. Newton with finite-difference gradients → Meff and g recover, small anisotropy terms
   stay 35-47% off, and the fit stops at a residual *worse than the known truth*.
3. Newton with exact symbolic derivatives (sympy, `exact_derivs.py`) → ~1%.

Nesting finite differences inside finite differences inside a root-find is the obvious
way to build this, and it silently does not work. None of the three papers mentions it;
their entire software disclosure is one footnote reading "we use Python packages
SCIPY.OPTIMIZE and NUMDIFFTOOLS."

One more thing worth recording: the 2018 paper's constant 45.6834 kOe² is **not**
$(\omega/\gamma)^2$. It is $hf/\mu_B$ with no $g$ — $(9.46 / 1.39962\times10^{-3})^2$.
That is why $g$ appears on the right-hand side above, multiplying the curvature, and it
is what makes $g$ a cleanly fitted parameter rather than a fixed constant. Getting this
wrong puts every number off by $g^2$.

## Where a Network *Would* Earn Its Place

Not for the inversion. But:

- **Amortized inference.** Simulate the forward model across parameter space, train a
  network to invert it, replace ~500 nonconvex fits per sample with one forward pass.
  Justified by throughput across many dopings × temperatures, not by statistics.
- **Peak extraction** from raw spectra, which is a genuinely hard pattern-recognition
  problem and the real bottleneck in my group's pipeline.
- **Across composition**, once there are enough samples. With 4-10 dopings it is a
  physically-motivated curve fit; 10-50 calls for Gaussian process regression (small
  data, calibrated uncertainty); a network needs hundreds, realistically a
  composition-gradient wedge sample.

## Files

- `exact_derivs.py` — sympy-generated exact $\theta,\phi$ gradient and Hessian
- `sb_synthetic.py` — generator, Newton equilibrium solver, curvature, fitter, LOOCV
- `synthetic_001.csv`, `synthetic_111.csv` — columns `thetaH_deg, phiH_deg, Hr_Oe`

Run: `python sb_synthetic.py 001` (~27 s) or `python sb_synthetic.py 111` (~210 s; the
3-fold landscape about [111] has more local minima, so the multi-start rarely
short-circuits). LOOCV folds are independent and should be parallelized before scaling.

## Next

Add the nested model ladder (C1–C4, U1–U3) and run LOOCV on each to select the
expansion order from data rather than assuming it; add bootstrap error bars; then run
the 2020 paper's validation test — fit $g$ and Meff freely and see whether they land on
physically sane values. If they don't, the static free-energy ansatz is wrong for
Er:YIG, which is a real risk given that Er³⁺ is a slow relaxer.

A candidate portfolio project.
