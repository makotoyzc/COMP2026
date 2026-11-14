# Can a CNN See the BKT Transition?

Week 10 agentic, Convolutions and Galaxy Classification. Started Friday 2026-11-13.
Numbers and figures for each step go in [RESULTS.md](RESULTS.md).
Builds on [XY_BKT](../../07_Stat_Mech_and_Ising/XY_BKT/) (week 7), which has the
Ising and XY samplers and a measured T_BKT.

## Status

| Step | What | Status |
|---|---|---|
| 0 | Snapshots: Ising and XY, several sizes | done 11/13 (L = 16, 32) |
| 1 | Ising, told the answer | done 11/13 |
| 2 | Ising, not told (learning by confusion) | run 11/13; check 2.2 fails by 0.006 (flat top) |
| 3 | XY from raw spins | 3a, 3b done 11/13 (3.1 fails at L = 16, 64: flat curves); 3c not run |
| 4 | XY from vortices only | to do |
| 5 | Can a small CNN compute vorticity? | stretch |
| 6 | Figures and write-up | to do |
| 7 | Disorder: random-bond XY | to do |

Today (11/13): Steps 0–2, and Step 3 if time is left. Stop at each step's check.
Nothing is committed yet.

## The question in one paragraph

Train a CNN on Ising snapshots labeled "below T_c" or "above T_c" and it finds
T_c (Carrasquilla and Melko 2017; Jim has assigned this as homework before).
That is easy, because an Ising snapshot below T_c is mostly one color: the
magnetization gives it away. The 2D XY model is different. An infinite XY
system has no magnetization at any T > 0, and its transition is
vortex-antivortex pairs coming apart. So what does a CNN use there? A finite
lattice still has some leftover magnetization. If the network reads that, it is
reading a finite-size effect. If it reads vortices, it has learned the real
mechanism. And can it find T_BKT without being told the answer? This matters for
thin superconducting films: the phase of the order parameter is an XY angle, and
the resistive transition is vortex unbinding (XY_BKT Step 3).

## What to expect (predictions, to be tested)

Learning by confusion (Step 2) finds the temperature where the data changes most,
without being told.

**What our network can measure.** Its output is a sigmoid of the lattice average
of *one local quantity*, seen through its window (5×5 for the standard network).
So it is a meter for a local observable: energy per site, how well spins line up
inside one window, or vortex number. It cannot compute the global |m| exactly,
because |m| is the absolute value of an average, not an average of something. For
Ising this hardly matters (Step 1: output 98% explained by |m|), because order
inside a 5×5 block tracks global order there. For XY it matters a lot, because
nothing local jumps at T_BKT.

A confusion peak lands where the quantity the network reads changes fastest with T:

| If the network reads... | its confusion peak should land | what moves it |
|---|---|---|
| energy per site | the specific-heat peak, T ≈ 1.05–1.1 (C = dE/dT; XY_BKT Step 4) | nothing: fixed |
| vortex number | near the same bump, where vortex number rises fastest | nothing: fixed |
| alignment within its window | where the correlation length shrinks to the window size, roughly T ≈ 1 for 5×5 | a wider window moves it down toward 0.893; L barely matters |
| bound vs free vortex pairs | T_BKT = 0.893 | sharpens as L grows |

For this network the window size plays the role L would play for a global
measurement. So the main prediction for Step 3 is a peak **above** 0.893 that
barely moves with L. A peak near 0.9 that sharpens with L would be the surprise.
Where the peaks actually land is the result.

## Known answers

| Quantity | Value | Source |
|---|---|---|
| Ising T_c | 2 / ln(1 + √2) = 2.2692 | Onsager, exact |
| XY T_BKT | 0.8929 | Hasenbusch 2005 |
| Our XY T_BKT | 0.9051 ± 0.0028 (all L), 0.9004 ± 0.0043 (L ≥ 16) | XY_BKT Step 2 |
| XY specific-heat peak | T ≈ 1.05–1.10, does not grow with L | XY_BKT Step 4 |
| Ising finite-size shift | T_c(L) − T_c ∝ 1/L (ν = 1) | standard |
| BKT finite-size shift | T(L) − T_BKT ∝ 1/(ln L)^2 | standard |

## Words used here

- **Snapshot**: one saved spin configuration, an L × L image.
- **Channel**: one layer of the image. Ising has 1 (the spin). XY uses 2, cos θ and
  sin θ, because θ itself jumps from 2π to 0 and would show a fake edge.
- **CNN**: a stack of small filters slid over the image. Each layer sees a slightly
  bigger patch than the one before.
- **Receptive field**: the size of the patch one output can see. Two 3×3 layers see 5×5.
- **Circular padding**: wrap the image edges around. That is the lattice's periodic
  boundary condition.
- **Global average pooling**: average each filter's output over the whole lattice,
  so the network ignores *where* a feature is.
- **Told (supervised)**: labels come from the known transition temperature.
- **Learning by confusion**: guess a cut T′, label every snapshot by it, train,
  record test accuracy; repeat for every T′ on the grid. Accuracy vs T′ is
  W-shaped, and the middle peak sits where the data really changes
  (van Nieuwenburg, Liu and Huber 2017).
- **Vortex**: a plaquette where θ winds by +2π (antivortex: −2π). From `xy.vortices()`.
- **Crossing temperature T*(L)**: where the network's average output passes 0.5.
- **Leak**: test data that is secretly almost the same as training data. Snapshots a
  few sweeps apart in one chain are near-copies, so train and test must come from
  different chains.

## Ground rules

- Train/test split by chain, never by shuffling snapshots.
- One small CNN definition (`model.py`) and one training loop (`train.py`), used by
  every step.
- Short, readable code. No try/except fallbacks.
- Fixed seeds. Every number in RESULTS.md comes from a script in this folder.
- Time one unit (one chain, one training) before any run. Anything over ~10 minutes
  gets a plan first and saves partial results.
- Simulated spins only, nothing from the lab, so this is safe for the public fork.
- Snapshot files are big and can be regenerated, so `data/` is git-ignored.

## The network (shared by all steps)

    input: C channels, L × L
    conv 3×3, C → 8, circular padding, ReLU
    conv 3×3, 8 → 16, circular padding, ReLU
    global average pool  → 16 numbers
    linear 16 → 1, sigmoid  → p(ordered)

1,265 weights, trained with binary cross-entropy and Adam. Small on purpose:
a network this size can be read, and it trains in seconds. Steps 4–5 change the
depth on purpose to change the receptive field.

---

## Step 0 — Snapshots

Make snapshots with the week-7 samplers, imported from XY_BKT, not copied.

| | Ising | XY |
|---|---|---|
| sampler | Swendsen-Wang (`ising.sw_sweep`) | cluster step (`xy_cluster.py`) + Metropolis (`xy.sweep`) |
| sizes | L = 16, 32, 64 | L = 16, 32, 64 |
| temperatures | 1.6 to 3.0, step 0.05 (29) | 0.5 to 1.5, step 0.05 (21) |
| chains per T | 6: chains 0–3 train, 4–5 test | same |
| snapshots per chain | 150, one every 50 sweeps | 150, one every 20 sweeps |

Measured cost on this Mac (2026-11-13): one sweep takes 0.07 / 0.15 / 0.53 ms for
XY at L = 16 / 32 / 64, and 0.06 / 0.09 / 0.24 ms for Ising Swendsen-Wang. With
gap = 50 that is under a minute per size on 16 cores. Start with L = 16 and 32;
L = 64 waits for Step 3.

**Check 0** (all must pass):
1. Ising energy per site matches Onsager's `e_exact(T)` within 0.01 for
   |T − T_c| > 0.2 (L = 32).
2. Ising ⟨|m|⟩ matches `m_exact(T)` within 0.02 for T ≤ 2.1 (L = 32).
3. XY vortex density matches `n_vort` in XY_BKT's `results_step2.npz` at the shared
   (L, T) points, within error bars.
4. Snapshots are nearly independent: the correlation of |m| between consecutive
   saved snapshots is below 0.2 at every T. Metropolis slows down near T_BKT at
   large L. If XY fails there, raise `gap`, or add a cluster update for XY (Wolff's
   reflection trick, vectorized like `ising.sw_sweep`).

Done 11/13. Metropolis alone failed check 4 (lag-1 of |m| up to 0.78 at L = 32,
and above 0.4 even at T = 0.5), so XY now uses the cluster step. Details and
numbers in RESULTS.md.

## Step 1 — Ising, told the answer

Label = (T < T_c). Train on L = 32 train chains, test on test chains. Augment with a
global spin flip s → −s, which leaves the energy unchanged.

- Plot: average output vs T, for L = 16 and 32 (64 later).
- Baseline: logistic regression on |m| alone, one number per snapshot.

**Check 1:**
1. Crossing T*(32) within 0.05 of 2.269.
2. T*(16) is farther from T_c than T*(32) (finite size, shrinking like 1/L).
3. What did it learn: network output vs |m| for every test snapshot falls on one
   curve, and test accuracy is within 1% of the |m| baseline. For Ising the CNN
   should be a magnetization meter. That is fine here, because |m| is the true
   order parameter.

## Step 2 — Ising, not told (learning by confusion)

For every trial cut T′ on the grid: relabel by (T < T′), train from scratch, record
test accuracy. 29 trainings; time one first.

**Check 2:**
1. W shape: accuracy near 1 at both ends of the grid (one label wins trivially), a
   dip on each side, a peak in the middle.
2. The middle peak within one grid step (0.05) of 2.269 for L = 32.
3. Same peak (within one grid step) from 3 different training seeds.

## Step 3 — XY from raw spins

Input: 2 channels, cos θ and sin θ. Augment with a random global rotation
θ → θ + α per snapshot (energy unchanged), so the network cannot use which way the
spins point, only how well they line up.

- 3a, told: label = (T < 0.893), for L = 16, 32, 64. The crossing T*(L) will sit
  near 0.893 at every L, because the labels force it (as with Ising: 2.274 vs
  2.269). The informative numbers are the test accuracy, which should be lower
  than Ising's since nothing local jumps at T_BKT, and what the network reads.
- 3b, not told: confusion, as in Step 2. This is the real test.
- 3c, wider window: repeat 3b with the four-layer network (9×9) from Step 4.
  If the network is reading alignment within its window, the peak should move
  down toward 0.893 as the window grows, while changing L should barely move it.
- Diagnostics: network output vs |m|, vs energy per site, and vs vortex density
  (R² of output against its mean in narrow bins, as in Step 1). Baselines:
  logistic regression on each of the three alone. Energy is the most likely
  thing a local meter reads.
- Before this step, read Beach, Golubeva and Melko (2018), who studied this exact
  question. A sub-agent summary is fine (week-1 lesson). Compare with what they found.

**Check 3:**
1. Same confusion peak (within one grid step) from 3 training seeds.
2. Place the 5×5 peak for each L against the prediction table. Expected: above
   0.893, near 1.0–1.1, and barely moving with L.
3. Which baseline the network matches best (|m|, energy or vortex density), from
   test accuracy and the R² diagnostics.
4. 3c: does the 9×9 peak sit lower than the 5×5 peak? Plot peak T against window
   size and against L on the same axes.

## Step 4 — XY from vortices only

Input: the vortex map from `xy.vortices()` (−1, 0 or +1 per plaquette), one channel.
Magnetization and smooth spin waves are gone; only vortices are left.

Below T_BKT, vortices come in tight +/− pairs. Above it, free ones wander. Telling
the two apart means seeing pairs, so the receptive field should matter. Run
confusion with three networks:

| Network | Receptive field | Can it see pairs? |
|---|---|---|
| one 1×1 conv, then pool | 1 plaquette | no, it can only count |
| standard (two 3×3) | 5×5 | close pairs |
| four 3×3 layers | 9×9 | yes |

**Check 4:**
1. The 1×1 counter's peak lands near the specific-heat bump (T ≈ 1.05–1.1), since
   counting is all it can do.
2. Does a wider receptive field move the peak toward 0.893? Report all three.
3. Same answers from 3 training seeds.

## Step 5 — Can a small CNN compute vorticity? (stretch)

Train a tiny CNN to output the vortex map from (cos θ, sin θ), with exact
per-plaquette labels from `xy.vortices()`. One plaquette needs a 2×2 view plus a
nonlinearity (the wrap of each angle difference).

**Check 5:** at least 99% per-plaquette accuracy on test chains. If vorticity is easy
to compute locally but the Step 3 network did not use it, the classifier took the
easier signal. Look at Step 3's first-layer filters for anything curl-like.

## Step 6 — Figures and write-up

1. Network output vs T: Ising and XY, all L.
2. Confusion W curves: Ising; XY raw; XY vortices with three receptive fields.
3. Output vs |m| and vs vortex count.
4. T*(L) vs 1/L (Ising) and vs 1/(ln L)^2 (XY).

Portfolio: this could merge with XY_BKT into one project on 2D transitions.

## Step 7 — Disorder: random-bond XY

A dirty or granular film has bonds of different strength. Model: the XY model with
each bond strength drawn at random, J_ij uniform in [1 − w, 1 + w]. One knob, w.
The average bond stays 1, so any change comes from the disorder itself. w = 0 is
the clean model. Grid: w = 0, 0.3, 0.6, 0.9.

- **True T_BKT(w):** from the helicity-modulus jump, Y(T_BKT) = 2T/π, as in XY_BKT
  Step 2, with J_ij inside the sums. Average over several bond realizations (say 8).
- **7a, transfer:** take the network trained on clean snapshots (Step 3) and apply
  it to disordered ones. Does its crossing follow the true T_BKT(w), or stay at
  the clean value?
- **7b, confusion on disordered data:** run learning by confusion directly at each w.
- **Code:** a bond-dependent copy of the XY sweep in this folder (J arrays for the x
  and y bonds). The week-7 code stays untouched.

**Check 7:**
1. w = 0 gives back the clean T_BKT (0.893, or XY_BKT's measured value).
2. The helicity T_BKT(w) falls as w grows.
3. Same CNN answers from 3 training seeds.

**Why:** in a real film, the gap between pairing and coherence grows with disorder
(Beasley, Mooij and Orlando 1979; the sheet-resistance knob in XY_BKT Step 3). This
step asks whether a network trained on a clean film can be trusted on a dirty one.

---

## Files (planned)

| File | Does |
|---|---|
| `data.py` | makes and loads snapshots (imports the XY_BKT samplers) |
| `xy_cluster.py` | XY cluster step (Wolff reflection, Swendsen-Wang style), added in Step 0 |
| `model.py` | the CNN |
| `train.py` | one training run: data in, accuracy and outputs out |
| `step0_snapshots.py` … `step5_vorticity.py` | one script per step; prints its check |
| `figures/` | one PNG per check |
| `data/` | snapshots (git-ignored) |
| `results_stepN.npz` | numbers behind RESULTS.md |

## References

Check the exact citations when reading.

- J. Carrasquilla and R. G. Melko, "Machine learning phases of matter,"
  Nature Physics 13, 431 (2017).
- E. P. L. van Nieuwenburg, Y.-H. Liu and S. D. Huber, "Learning phase transitions
  by confusion," Nature Physics 13, 435 (2017).
- M. J. S. Beach, A. Golubeva and R. G. Melko, "Machine learning vortices at the
  Kosterlitz-Thouless transition," Phys. Rev. B 97, 045207 (2018).
- M. Hasenbusch, "The two-dimensional XY model at the transition temperature: a
  high-precision Monte Carlo study," J. Phys. A 38, 5869 (2005).
- M. R. Beasley, J. E. Mooij and T. P. Orlando, "Possibility of vortex-antivortex
  pair dissociation in two-dimensional superconductors," Phys. Rev. Lett. 42, 1165
  (1979).
