# Results: Can a CNN See the BKT Transition?

Numbers and figures, one section per step, filled in as each step's check passes.
The plan and the reasoning are in [PLAN.md](PLAN.md).

## Headline numbers

| Quantity | Measured | Expected |
|---|---|---|
| Ising crossing T*(32), told | 2.274 (T*(16) = 2.289) | 2.269 |
| Ising confusion peak, L = 32 | 2.325 on the grid (flat top 2.275–2.325; parabola 2.303) | 2.269 |
| XY confusion peak, raw spins, L = 16 / 32 / 64 | 1.125 / 1.075 / 1.025 (median of 3 seeds, 5×5) | see PLAN prediction table |
| XY confusion peak, vortices only (1×1 / 5×5 / 9×9) | – | see PLAN prediction table |

## Step 0: Snapshots

Run 2026-11-13, `python step0_snapshots.py`, about 30 s on 16 cores. Figure:
[figures/step0_check.png](figures/step0_check.png).

Data: L = 16 and 32, 6 chains per T (0-3 train, 4-5 test), 150 snapshots per
chain. Ising: Swendsen-Wang, a snapshot every 50 sweeps, 29 T from 1.6 to 3.0.
XY: a cluster step plus a Metropolis sweep per "sweep", a snapshot every 20,
21 T from 0.5 to 1.5. Everything starts ordered and thermalizes 2000 sweeps.
L = 64 not made yet (about 3 s per XY chain, so under a minute).

**Check 0: pass.**

| Check | Result | Needed |
|---|---|---|
| 1. Ising e vs Onsager, \|T − T_c\| > 0.2, L = 32 | max diff 0.0054 | < 0.01 |
| 2. Ising ⟨\|m\|⟩ vs Onsager, T ≤ 2.1, L = 32 | max diff 0.0012 | < 0.02 |
| 3. XY vortex density vs week 7, 14 shared (L, T) points | 13 within 2.2σ; L = 16, T = 0.90 at −3.5σ; rerun with 48 new chains: z = −0.1 | within error bars |
| 4. lag-1 correlation of \|m\|, worst T | Ising 0.04 / 0.05; XY 0.07 / 0.11 (L = 16 / 32) | < 0.2 |

What happened on the way:

- **Metropolis alone was too slow for XY.** On the first try (Metropolis only, a
  snapshot every 50 sweeps), check 4 failed. At L = 32 the lag-1 correlation of |m| was
  0.45-0.55 at *every* T below 1.0 and peaked at 0.78 at T = 1.05. Local moves
  are slow for long spin waves at all low T, not only near T_BKT, and it gets
  worse as L² with size. Raising the gap would have cost about 4× more at L = 64.
- **Fix: a cluster step** (`xy_cluster.py`: Wolff's reflection trick, built as
  Swendsen-Wang clusters with the same bond list as `ising.sw_sweep`). It costs
  0.28 / 0.84 ms per step at L = 32 / 64. Correctness test at L = 16, T = 0.90,
  16 chains × 40,000 steps each: vortex density 0.00849 ± 0.00002 (cluster)
  vs 0.00846 ± 0.00004 (Metropolis only) vs 0.00850 ± 0.00008 (week 7).
  Energy −1.4434 vs −1.4437. The cluster step gives the same physics.
- **The one 3.5σ point is a fluctuation, not a bug.** The six chains at L = 16,
  T = 0.90 give 0.00778 (all six happen to be low; one chain in three is below
  0.0080). Six chains also make a rough error bar. 48 new chains from the same
  code (indices 6-53, never used for training) give 0.00848 ± 0.00012. I kept
  the data as made instead of choosing new seeds until it passed. It is a 9% low
  vortex count at one temperature of the small lattice, which should not matter
  for a classifier.

## Step 1: Ising, told the answer

Run 2026-11-13, `python step1_ising_told.py`, about 10 s (Mac GPU). Figure:
[figures/step1_check.png](figures/step1_check.png). One network per L, seed 0,
15 epochs, global-flip augmentation. The CNN has 1,265 weights.

**Check 1: pass.**

| L | CNN test acc | \|m\| logistic baseline | T*(L) | T* − T_c | output vs \|m\| R² |
|---|---|---|---|---|---|
| 16 | 0.9363 | 0.9278 | 2.2894 | +0.020 | 0.983 |
| 32 | 0.9702 | 0.9648 | 2.2744 | +0.005 | 0.984 |

1. T*(32) is 0.005 from T_c (needed < 0.05).
2. T*(16) is 4× farther off than T*(32). For ν = 1 the shift should halve when L
   doubles, so 4× is more than expected. With one seed and a 0.05 grid, T* is
   only good to about ±0.005, so this needs L = 64 and more seeds before reading
   anything into it.
3. The CNN beats the |m|-only baseline by 0.5% (within 1%). Its output is 98%
   explained by |m| (R² of output against its mean in narrow |m| bins). As
   expected, the Ising CNN is mostly a magnetization meter. The last 2% shows
   in the scatter plot: a few snapshots with |m| ≈ 0.2-0.5 get high outputs.
   My guess (not checked): these are two-domain configurations, where |m| is small but the
   spins are still ordered locally.

Code notes: training runs on MPS (about 5 s per L = 32 run, vs 56 s on one CPU
core). The |m| baseline is a small torch logistic regression in `train.py`,
not sklearn: importing sklearn before torch segfaults here (two OpenMP runtimes).

## Step 2: Ising, not told

Run 2026-11-13, `python step2_ising_confusion.py`, 87 trainings (29 cuts × 3
seeds, L = 32) in 7 min on 3 GPU + 12 CPU workers (`train.run_parallel`).
Figure: [figures/step2_check.png](figures/step2_check.png).

**Check 2: FAIL on 2.2, by 0.006.**

1. W shape: yes, for all 3 seeds. Accuracy is 1.000 / 0.966 at the two ends,
   dips to 0.93–0.95 near T' = 1.75–1.80 and 2.75, and peaks in the middle.
2. Peak position: boundary at 2.325 for every seed, |2.325 − T_c| = 0.056
   (needed ≤ 0.05).
3. Seeds agree exactly (spread 0).

Notes on 2.2:

- **Labeling convention.** Label = (T < T′) with T′ on the grid puts the
  label boundary halfway between T′ − 0.05 and T′. So the peak is reported at
  T′ − 0.025. Before this fix, the script printed the cut itself (2.35, 0.081 off).
  The fix only moves the reported number. The criterion is unchanged.
- **The top is flat.** Mean accuracy at the boundaries near T_c:

  | boundary T | 2.175 | 2.225 | 2.275 | 2.325 | 2.375 | 2.425 |
  |---|---|---|---|---|---|---|
  | accuracy (3-seed mean) | 0.9669 | 0.9669 | 0.9706 | 0.9709 | 0.9670 | 0.9620 |

  2.275 (0.006 from T_c) and 2.325 differ by 0.0003, about 3 of the 8,700 test
  snapshots. All three seeds put 2.325 on top, but they share the same test set,
  so that is not three independent votes. A parabola through the top gives
  2.303. The honest reading: the peak spans 2.275–2.325, T_c sits at its low
  edge, and the 0.05 grid cannot split it.
- A peak slightly above T_c at L = 32 is plausible. Finite-size pseudo-critical
  points (e.g. the susceptibility peak) sit above T_c. This step did not test that.

## Step 3: XY from raw spins

Input (cos θ, sin θ), random global rotation per snapshot. L = 64 XY data made
for this step: 48 s, worst lag-1 correlation of |m| 0.06. Figures:
[figures/step3_told.png](figures/step3_told.png),
[figures/step3_confusion.png](figures/step3_confusion.png).

Read first: Beach, Golubeva and Melko 2018 (arXiv 1710.09842v1, read in full by
a sub-agent; the numbers quoted below were checked against the text).
Raw angles, CNN without global pooling, L = 8 to 64. Supervised at T_KT:
the extrapolated CNN transition is 0.92 ± 0.03, but with "no clear finite-size
scaling trend", which they read as a finite-size magnetization threshold.
Confusion on raw spins gave no usable W: "all T* < T_KT have accuracy
greater than 98.5% for L = 64". On vortex maps, the confusion peak "always stays
above [T_KT], similar to the specific heat peak", with best accuracy at T* ≈ 1.

### 3a, told (label T < 0.893), `python step3_xy_raw.py told`, 25 s

| L | CNN acc | T* | logistic on \|m\| | on energy | on vortex density | R² vs \|m\| | vs energy | vs vortices |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.9460 | 0.867 | 0.9275 | **0.9549** | 0.9021 | 0.918 | **0.962** | 0.817 |
| 32 | 0.9670 | 0.864 | 0.9495 | **0.9795** | 0.9613 | 0.924 | **0.973** | 0.931 |
| 64 | 0.9889 | 0.869 | 0.9662 | **0.9968** | 0.9840 | 0.938 | **0.990** | 0.977 |

- **Check 3.3: the network is an energy meter.** Energy per site explains its
  output best at every L (R² 0.96 to 0.99), better than |m| does. A logistic
  regression on energy alone *beats* the CNN at every L (0.997 vs 0.989 at
  L = 64). Accuracy rises with L only because energy fluctuations per site
  shrink with size, so a threshold on energy gets sharper.
- T* is 0.864–0.869 at every L. As predicted, this is forced by the labels, not
  found. It sits about 0.025 below 0.893 because no grid points lie between
  0.85 and 0.90.
- Different from Beach et al.: they concluded magnetization. With global
  average pooling and cos/sin input, our net prefers local alignment (bond
  energy), which a 5×5 window can measure directly. |m| = √(⟨cos θ⟩² + ⟨sin θ⟩²)
  needs a nonlinearity *after* the average over the lattice. Our net has only a
  linear layer there, and the random rotation makes ⟨cos θ⟩ and ⟨sin θ⟩ useless
  on their own. So by design, this architecture cannot compute |m| exactly. It can
  only use what correlates with it inside a 5×5 window.

### 3b, confusion, 5×5 window, `python step3_xy_raw.py confusion 5x5 16 32 64`

63 trainings per L (21 cuts × 3 seeds), 26 min in total on 3 GPU + 12
CPU workers. That was slower than planned: XY trainings on CPU are much slower
than Ising, so the 35% CPU share measured on Ising left the GPU idle while CPU
jobs finished. **Next time, run XY on the GPU only.**

| L | peak, seeds 0 / 1 / 2 | parabola | prominence | 3.1 (spread ≤ 0.05) |
|---|---|---|---|---|
| 16 | 0.775 / 1.125 / 1.125 | 0.76 / 1.13 / 1.13 | 0.01 | FAIL |
| 32 | 1.075 / 1.075 / 1.075 | 1.08 / 1.08 / 1.06 | 0.02 | ok |
| 64 | 1.025 / 0.575 / 1.025 | 1.03 / 0.59 / 1.02 | 0.04 | FAIL |

- **Check 3.1: FAIL at L = 16 and 64.** In each case, one seed picks a different
  bump. The curves are nearly flat, not a clean W. At L = 64, accuracy is
  0.98–0.99 for every cut from 0.6 to 1.2, which is the "−∨" shape Beach et
  al. describe. The prominences are only 0.01–0.04. Two of three seeds agree at
  every L, so the medians below are usable, but treat them as rough.
- **Check 3.2:** median peak 1.125 / 1.075 / 1.025 at L = 16 / 32 / 64. That is
  above T_BKT (by 0.23 / 0.18 / 0.13) and in or near the specific-heat bump
  (1.05–1.1), as predicted for a local meter. But the peak *does* move with L,
  down by 0.05 each time L doubles. The prediction table said "barely moves".
  Whether it keeps falling toward 0.893 or stops at about 1.0 can't be told
  from three sizes and one-step (0.05) moves.
- Matches Beach et al. (vortex-input peak stays above T_KT at T* ≈ 1). Our
  raw-spin curves at least have a middle peak, which theirs did not. Our
  window is 0.5–1.5; theirs was 0.1–3.0.

### 3c (9×9 window): not run

About 33 min at L = 64 on the GPU (L = 16 and 32: about 11 min). Check 3.4 waits for this.

## Step 4: XY from vortices only

Not run yet.

## Step 5: Can a small CNN compute vorticity?

Not run yet.

## Step 7: Disorder, random-bond XY

Not run yet.
