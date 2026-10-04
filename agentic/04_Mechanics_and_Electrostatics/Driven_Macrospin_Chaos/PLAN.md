# Chaos in a Driven Macrospin

Friday agentic session, week 4. Prompt: [../Chaos_and_the_Butterfly_Effect.md](../Chaos_and_the_Butterfly_Effect.md).

**Status: planning done, starting Phase 0.**

## Today's scope — hard stop 11:30

Starting point, before end of class on 11:30 am
1. work on solving the diff.eq.
2. extract the Lyapunov exponent.
3. solve for dynamics

That is **Phase 0, Phase 4, and the sphere/torque visualizations**, and nothing else.
The seven phases below are the full project, which is several hours of work. Anything
not in those three lines is after class. In particular the Kittel and linewidth checks
in Phase 1 are the best verification in this project, and they are *still* not today.

**Tactical note on item 2.** Do not hunt for chaos by eye — I can burn the whole class
sweeping drive amplitudes and find nothing. Write the $\lambda$ measurement *first*,
then run a coarse scan over drive amplitude computing $\lambda$ at each point. The scan
is the result either way: $\lambda > 0$ somewhere means I found chaos, $\lambda \le 0$
everywhere is a real finding with a plot to back it.

## The idea

Jim's prompt uses Lorenz. I'm using the Landau–Lifshitz–Gilbert equation instead,
because it is the same kind of object — a low-dimensional nonlinear ODE with a
dissipative term — and because I already know what its *linear* limit is supposed to
do, which hands me a verification the Lorenz system can't offer.

$$\frac{d\mathbf{m}}{dt} = -\gamma\,\mathbf{m}\times\mathbf{H}_{\rm eff}
+ \alpha\,\mathbf{m}\times\frac{d\mathbf{m}}{dt}$$

Solved for $\dot{\mathbf m}$ (the Landau–Lifshitz form, which is what I'll actually integrate):

$$\frac{d\mathbf{m}}{dt} = -\frac{\gamma}{1+\alpha^{2}}\,\mathbf{m}\times\mathbf{H}_{\rm eff}
-\frac{\alpha\gamma}{1+\alpha^{2}}\,\mathbf{m}\times(\mathbf{m}\times\mathbf{H}_{\rm eff})$$

At small RF drive this is ordinary FMR: a Kittel resonance with a Gilbert linewidth.
At large drive it goes nonlinear — foldover, bistability, period doubling, and
(the claim to be tested) chaos with a positive Lyapunov exponent.

**The question for the afternoon:** does a driven macrospin actually become chaotic,
what is $\lambda$, and once the trajectory is unpredictable, what about the
*measurement* survives?

## Why this fits the week

- It is a mechanics problem. $\mathbf m$ precesses like a gyroscope; the damping term
  is the dissipation. Tuesday's Problem 5 was "one integrator conserves the invariant
  and the other doesn't" — here the invariant is $|\mathbf m| = 1$ and the same lesson
  applies with teeth.
- $\mathbf H_{\rm eff}$ contains the demagnetizing field, which is a solved
  electrostatics problem (magnetostatics, same Laplace equation, same boundary-value
  structure as Tuesday's Problem 2).
- It is Jim's butterfly-effect prompt done on a system where I can check the answer.

## A structural fact worth getting right before coding

$|\mathbf m| = 1$ is conserved exactly, so the phase space is the 2-sphere. An
*autonomous* flow on a 2-manifold cannot be chaotic (Poincaré–Bendixson). So:

- Static field only → no chaos, ever. Limit cycles at best.
- **Circularly** polarized RF with uniaxial symmetry about $z$ → go to the frame
  rotating at the drive frequency and the equation becomes autonomous again → still
  no chaos. Only fixed points and limit cycles, which is exactly why the rotating-frame
  trick is so useful in the nonlinear-FMR literature.
- **Linearly** polarized RF → genuinely non-autonomous, phase space is
  $S^{2}\times S^{1}$, three-dimensional → chaos is allowed.

This is not a detail. It means the drive polarization is the control knob that turns
chaos on and off, and I should demonstrate exactly that rather than assume it.

## Conventions

| quantity | value / unit |
|---|---|
| time | ns |
| field | T (or mT where it reads better) |
| $\gamma/2\pi$ | 28.0 GHz/T ($g = 2$) |
| $\alpha$ | start at 0.005, sweep later |
| $\mathbf m$ | unit vector, $\lvert\mathbf m\rvert = 1$ |
| $\lambda$ | ns$^{-1}$ |

No `try`/`except`, no silent fallbacks, no magic defaults. Short readable functions.

---

# Phases

Each phase has a gate. **Do not start the next phase until the gate passes and I
understand why it passes.**

## Phase 0 — integrator and the norm

Write the LL right-hand side and two integrators: plain RK4, and one that preserves
$|\mathbf m| = 1$ (projection after each step, or a midpoint/Cayley scheme).

**Gate.** Run $10^{6}$ steps with $\alpha = 0$ in a static field.
1. RK4 drifts: $\bigl\lvert |\mathbf m| - 1\bigr\rvert$ grows as a power of step count. Plot it.
2. The norm-preserving scheme sits at $10^{-16}$ forever.
3. With $\alpha = 0$ the precession frequency must equal the Larmor value
   $\gamma H/2\pi$ to the digits the timestep deserves, and the Zeeman energy
   $-\mathbf m\cdot\mathbf H$ must be constant.

This is Tuesday's Euler vs Euler–Cromer plot with a different invariant. It is the
cheapest possible proof that the integrator is not lying.

## Phase 1 — the linear limit, against two independent references

Small transverse RF drive. Sweep field at fixed frequency, record the absorbed power,
fit a Lorentzian.

**Gate — and this is the strongest check in the whole project.**
1. The resonance field must satisfy Kittel. In-plane:
   $$f = \frac{\gamma}{2\pi}\,\mu_0\sqrt{H\,(H + M_{\rm eff})}$$
   out-of-plane:
   $$f = \frac{\gamma}{2\pi}\,\mu_0\,(H - M_{\rm eff})$$
2. The fitted FWHM linewidth must obey the Gilbert form with the $\alpha$ I put in:
   $$\mu_0\,\Delta H = \frac{4\pi\alpha f}{\gamma} + \mu_0\,\Delta H_0$$
   with $\Delta H_0 = 0$, since a macrospin has no inhomogeneous broadening.
3. **Run my own FMR_Engine_v15 fitter on the synthetic spectra and confirm it returns
   the input $\alpha$.** The simulator and the fitter share no code. If they agree,
   both are probably right; if they disagree, I've learned something either way.

Check 3 is the point of choosing LLG over Lorenz. Write the synthetic sweeps out as
CSV in the same column format the real pipeline eats.

## Phase 2 — nonlinear: foldover and bistability

Raise the RF amplitude. The Lorentzian leans over, then becomes multivalued.

**Gate.** Sweep the field up and then back down at fixed drive. Above a critical
amplitude the two sweeps must *not* coincide — a hysteresis loop in the absorption,
with jumps. Locate the critical amplitude where the loop first opens and compare to
the Duffing foldover threshold. Still no chaos here: this is a bent resonance, not a
strange attractor, and I should be able to say why.

## Phase 3 — the route to chaos

Stroboscope the trajectory once per drive period (a Poincaré section), plot $m_z$
against drive amplitude.

**Gate.**
1. A bifurcation diagram showing period-1 → period-2 → period-4 → chaos.
2. The same sweep with a *circularly* polarized drive must show **no** chaos, for the
   rotating-frame reason above. This is the control experiment and it is a real
   prediction, not a sanity check.
3. If the period-doubling cascade is clean enough, estimate the Feigenbaum constant
   4.669 from the bifurcation spacings. It is universal, so it is a check on nothing
   but the physics.

## Phase 4 — measure $\lambda$

Two trajectories separated by $\delta$, renormalize the separation at fixed intervals
(Benettin), accumulate the log stretching.

**Gate.**
1. $\lambda > 0$ in the chaotic window, $\lambda \approx 0$ on a limit cycle,
   $\lambda < 0$ at a fixed point. The sign must track the bifurcation diagram.
2. $\lambda$ independent of $\delta$ over several decades and of the renormalization
   interval.
3. $\lambda$ independent of timestep.

## Phase 5 — the exact constraint

Lorenz gets a free exact check because the flow has constant divergence. LLG has an
analogue and it is better, because it is *structural*: the precession term is
Hamiltonian on the sphere and therefore divergence-free with respect to the area
measure, so **every bit of phase-space contraction comes from $\alpha$.**

Derive $\nabla\cdot\dot{\mathbf m}$ by hand. It should come out proportional to
$\alpha\gamma/(1+\alpha^{2})$ times the surface Laplacian of the energy on the sphere.
Then compute the full Lyapunov spectrum (Benettin with QR re-orthonormalization) and
check that the exponents sum to the time average of that divergence along the attractor.

**Gate.**
1. Derive the coefficient myself and write the derivation in the README. Do not take
   it on faith from the agent — this is the one piece of algebra the whole phase rests on.
2. At $\alpha = 0$ the sum must be exactly zero, and the measured spectrum must
   confirm it.
3. One exponent must be exactly zero (the drive-phase direction). That one is free and
   it is a good check that the QR implementation is right.

## Phase 6 — what is still reproducible

Jim's step 4, and the actual physics question. The trajectory $\mathbf m(t)$ is
garbage after a few $1/\lambda$. But the *measurement* is a time average.

**Gate.** Take two initial conditions a distance $10^{-12}$ apart, confirm the
trajectories are uncorrelated by $t \gg 1/\lambda$, and then show that
1. the time-averaged absorbed power agrees between them to within the statistical error, and
2. the stroboscopic density on the sphere is the same attractor.

Then state the predictability horizon: improving the initial data by a factor of ten
buys $\ln(10)/\lambda$ more nanoseconds, full stop. Put a number on it in ns.

---

# What each phase produces

Every phase owes me **two** pictures: one about the dynamical system, one about the
magnet. If a phase produces neither, I've written code without learning anything.

| phase | figure | why it earns its place |
|---|---|---|
| phase | dynamics picture | magnetization picture |
|---|---|---|
| 0 | norm drift vs step count, two integrators | **M1** torque decomposition animated |
| 1 | absorption + Lorentzian fit; linewidth vs frequency | **M2** precession ellipse + phase lag |
| 2 | field sweep up and down at several powers | **M3** cone angle snapping open |
| 3 | bifurcation diagram, linear vs circular drive | **M3** stroboscoped cone angle |
| 4 | $\ln d(t)$ with the fitted $\lambda$ | two moments on one cone, decorrelating |
| 5 | exponent sum vs derived divergence | **M4** energy landscape + orbit |
| 6 | two trajectories diverging, same mean power | time-averaged cone distribution |
| A | $k$-space spectrum leaking out of $k=0$ | **M5** arrows on a chain, spin wave breakup |

## The magnetization itself — this comes first

The pictures below are dynamical-systems pictures. They are not the physics. The
physics is a magnetic moment precessing in a field, and every phase owes me a picture
of *that* too, or I've built a chaos project that happens to use magnetic notation.

**M1. The torque decomposition, animated. (Phase 0 — and it is a verification, not decoration.)**
Draw four vectors at each instant: $\mathbf m$, $\mathbf H_{\rm eff}$, the precessional
torque $-\gamma\,\mathbf m\times\mathbf H_{\rm eff}$ (tangent to the cone), and the damping
torque $-\alpha\gamma\,\mathbf m\times(\mathbf m\times\mathbf H_{\rm eff})$ (pointing along
the meridian toward $\mathbf H_{\rm eff}$). Watch them move.

This is the whole equation in one animation: precession carries $\mathbf m$ around the
cone, damping slides the cone shut. It is also the fastest sign-error detector there is
— a flipped sign in the damping term makes the cone *open* instead of close, and I will
see it in two seconds instead of debugging a linewidth that came out negative. Sign
conventions are the most common LLG bug by a wide margin. Make this before anything else.

**M2. The precession ellipse, and its aspect ratio is a quantitative gate. (Phase 1)**
The orbit is not a circle. For an in-plane magnetized film the out-of-plane excursion is
squashed by the demagnetizing field, and the ratio of the two axes is set by the same
quantities that set the Kittel frequency — roughly $\sqrt{(H+M_{\rm eff})/H}$, which I
should derive rather than trust. So: plot the traced ellipse, measure the aspect ratio,
and check it against the analytic value. A picture that doubles as a gate is worth two
pictures. Also show the **phase lag** between $\mathbf m$ and the RF field sweeping
through resonance, passing through 90 degrees exactly on resonance — that phase lag is
*why* there is absorption at all, and it makes the Lorentzian feel earned.

**M3. Cone angle as the physical observable. (Phases 2, 3)**
$\theta(t)$, the angle off equilibrium, is what an experimentalist actually thinks in:
a degree or two in linear FMR, large-angle once it goes nonlinear. Track it.
- Phase 2: $\theta$ versus field shows the foldover as the cone **snapping open** at the
  bistability edge, which is far more legible than a kink in an absorption curve.
- Phase 3: stroboscope $\theta$ once per drive period. Period doubling then has a plain
  physical statement — *the magnet traces a different cone on alternate RF cycles.*
  That sentence is the reason to do this project in magnetic language instead of Lorenz.

**M4. The energy landscape on the sphere, with the trajectory on it. (Phase 5)**
Color the sphere by the energy $E(\mathbf m)$ — Zeeman plus demag plus anisotropy — and
overlay the orbit. Precession runs **along** the contours; damping crosses them
**downhill**. That is "the precession term is Hamiltonian and the damping term is
gradient flow" made visible, which is exactly the structural claim Phase 5's divergence
derivation rests on. Draw this before doing the algebra and the algebra will be obvious.

**M5. Real space, not just a unit vector. (Phase 1 onward, and direction A)**
Show the moment inside an actual slab with the demag field drawn, so the geometry that
sets $M_{\rm eff}$ is visible rather than a parameter in a dict. For direction A this
becomes the classic magnetization-dynamics animation: a row of arrows on a lattice doing
the Mexican wave, the uniform mode with every arrow in phase, and then the Suhl
instability as that uniform wave visibly breaks up into a short-wavelength ripple at
half the drive frequency. Watching a spin wave grow out of the uniform mode is the
single best figure available anywhere in this project.

## Three-dimensional, and this is the real advantage over Lorenz

Lorenz lives in an unbounded $\mathbb{R}^3$ and everyone has seen the butterfly. This
system lives **on a sphere** — bounded, familiar, and genuinely nicer to look at.

1. **Trajectory on the unit sphere.** One 3D view, three regimes: a tight spiral into
   equilibrium at low drive, a clean closed loop on the limit cycle, a tangle that
   never closes in the chaotic window. Same code, three drive amplitudes.
2. **The butterfly effect, animated.** Two dots $10^{-12}$ apart on the sphere. They
   move as one, and then they don't. This is the single most convincing thing in the
   project and it takes ten lines once the integrator works. Lorenz cannot do this as
   cleanly because the attractor is unbounded and the eye loses the dots.
3. **Stroboscopic section draped on the sphere.** Sample once per drive period and
   accumulate. Period-1 is one dot, period-2 is two, chaos is a fractal-looking curtain
   hanging on the sphere. Rotate it and the structure is obvious.
4. **Animated foldover.** Sweep the field up and back with the orbit shown live; the
   jump at the bistability edge is a visible snap, not a kink in a curve.
5. **$\lambda$ as a colormap over (drive amplitude, field).** The nonlinear phase
   diagram, colored by Lyapunov exponent. This is the portfolio centerpiece and it is
   expensive to compute, so it's a Phase 4 overnight job, not an in-class one.
6. **For direction A:** space-time plot of $m_z(x,t)$ down the spin chain, plus the
   $k$-space energy spectrum animating as the uniform mode bleeds into $k \neq 0$.
   Watching the Suhl instability happen is worth a lot.

## Format

The portfolio is a `github.io` site, so the 3D things should be **web-native, not PNG**
— they'll be embedded live. I already shipped an `index.html` GUI for week 3, so this
is a known pattern: `plotly` writes standalone interactive HTML with almost no effort,
and `three.js` is the move if I want a real orbit viewer with a drive-amplitude slider.
Matplotlib stays for the quantitative 2D plots, which should look like my usual plots.

Animations: `FuncAnimation` to mp4 for anything that goes in a talk, since the
presentation is ten minutes on December 8.


---

# Push harder — where this goes past 100 minutes

Ranked by how much I actually want to do them.

**A. Chaos is the macrospin caricature of the Suhl instability.**
The real reason high-power FMR in YIG goes nonlinear is not macrospin chaos — it's that
the uniform mode becomes unstable to *pairs of spin waves* at $k \neq 0$, half the drive
frequency for the first-order process. A macrospin cannot see this at all, by construction.
Go to a chain of $N$ exchange-coupled spins, drive it uniformly, and watch energy leak
into $k \neq 0$. Verification: the Suhl threshold has an analytic form, and the
instability must appear at exactly $f_{\rm drive}/2$. This is the highest-ceiling
direction and it is the actual physics of the films I measure.

**B. The nonlinear FMR phase diagram.**
Map drive power against field into regions: linear / foldover / period-doubled /
chaotic / spin-wave unstable. One figure, and it's the portfolio centerpiece. The
experimental arm, if I want one, is a VNA power sweep on a real film — the foldover
threshold is measurable.

**C. The slow relaxer as a memory kernel — the one that breaks the dimension count.**

Two *different* temperature effects get confused here and I should keep them apart.

*C1, the deterministic one, and the interesting one.* The Er:YIG damping peak at 40 K is
not thermal noise. It is the Van Vleck-Orbach **slow relaxation mechanism**: the Er
ion's level populations relax toward the instantaneous magnetization direction with a
characteristic time $\tau(T)$, and the linewidth contribution has the shape

$$\Delta H_{\rm sr} \;\propto\; \frac{\omega\tau}{1+\omega^{2}\tau^{2}}\;
\frac{1}{T}\,\mathrm{sech}^{2}\!\left(\frac{\Delta}{2k_{B}T}\right),
\qquad \tau = \tau(T)$$

which peaks where $\omega\tau = 1$. That is the 40 K peak. (Confirm the prefactor from
Sparks / Van Vleck rather than letting the agent assert it; the $\omega\tau/(1+\omega^2\tau^2)$
shape is the part I'm sure of.)

Here is why this matters for *chaos* and not just for fitting. A relaxation time means
the Er contribution to $\mathbf H_{\rm eff}$ **lags** the magnetization — it is not a
function of $\mathbf m(t)$, it is a functional of the whole history:

$$\mathbf H_{\rm Er}(t) = \int_{-\infty}^{t} \frac{e^{-(t-t')/\tau}}{\tau}\;
\mathbf H\!\left[\mathbf m(t')\right]\,dt'$$

Gilbert damping is the $\tau \to 0$ limit of this, where the kernel becomes a delta
function and the lag collapses into a term proportional to $\dot{\mathbf m}$. But at
finite $\tau$ the phase space is no longer the 2-sphere — a memory kernel is formally
infinite-dimensional. **The Poincare-Bendixson argument from the top of this file does
not apply.** An autonomous slow-relaxer magnet could in principle be chaotic with no
drive at all, which the macrospin flatly cannot.

And the regime where this is strongest is handed to me by the experiment: $\omega\tau = 1$
at the peak means $\tau \approx 16$ ps at 10 GHz, against a precession period near 100 ps.
The memory time is a serious fraction of the period exactly where the linewidth peaks.

**Gate.** Two exact checks, both available.
1. $\tau \to 0$ must reproduce ordinary Gilbert damping with an effective $\alpha$, and
   the numbers must match Phase 1.
2. Linearize the memory model and the resulting linewidth must reproduce the
   $\omega\tau/(1+\omega^{2}\tau^{2})$ slow-relaxer shape, peaking at $\omega\tau = 1$.
   If it does, the memory kernel is the *same physics* my v13/v15 fits have been
   parameterizing all along, and I can feed it the real $\tau(T)$ from 13 temperatures
   of data rather than inventing one.

Then: does $\lambda$ inherit the 40 K peak? Is there a chaotic window that only exists
near $\omega\tau = 1$?

*C2, the stochastic one, kept separate.* Langevin field, amplitude fixed by
fluctuation-dissipation from $\alpha$ and $T$ with no free parameter. Exact check: drive
off, the sampled distribution must be Boltzmann in the Zeeman energy. The question here
is only "when does noise swamp the deterministic chaos," which is a real question but a
smaller one than C1. Do C1 first.

**D. The magnet as a reservoir computer.**
A driven nonlinear system at the edge of chaos is a reservoir: feed an input signal in
through the drive, read $\mathbf m(t)$ out linearly, train only the readout weights.
Spintronic reservoir computing is a live field. The hook is that performance should
*peak* near the edge of chaos — too ordered and there's no memory, too chaotic and
there's no reproducibility. That ties week 4 to the ML half of the course.

**E. Two coupled macrospins.**
Exchange or dipolar coupled bilayer: acoustic and optic modes, more room for chaos,
and a direct map onto real multilayer FMR spectra.

---

# Decisions — settled, not up for debate today

- [x] **Uniaxial easy axis along $z$; static field at 45 degrees to it.** The
      misalignment supplies the nonlinearity. Aligned fields are too symmetric.
- [x] **Linearly polarized RF.** Required for chaos at all — see the
      Poincare-Bendixson argument at the top. Circular drive is the Phase 3 control.
- [x] **No demagnetizing tensor yet.** Uniaxial anisotropy plus Zeeman only. Film
      geometry arrives with Phase 1, when $M_{\rm eff}$ has to mean something.
- [x] **$\alpha = 0.01$.** Matters more than it looks: real YIG is near $10^{-4}$, so
      ringdown takes $\sim 10^{4}$ precession periods and I would spend the whole class
      watching transients. Note in the README that 0.01 is not YIG's value. Go small
      only once everything works.
- [x] **Generic parameters today.** Er:YIG numbers arrive with the v15 cross-check.
- [ ] Phase 5 gets its derivation written out longhand before I let the agent near it.
      *(Still open. Not today.)*

# Rules for myself today

- Commit before every agent turn, start clean.
- One phase at a time. The gate is not optional and "it looks right" is not a gate.
- Ask for short code. If a phase produces more than ~150 lines I over-asked.
- I am the architect.
