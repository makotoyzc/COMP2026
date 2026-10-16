# Hard Disks — Metropolis Monte Carlo

Self-chosen week-6 agentic project (Statistics and Monte Carlo). This is the
problem the Metropolis algorithm was invented for in 1953: N identical disks in a
periodic box, with one rule — no two disks may overlap.

## The physics in one paragraph

There is no energy, only "allowed" (weight 1) or "overlapping" (weight 0). So a
Metropolis move is accepted if it makes no overlap and rejected if it does, and
temperature drops out entirely. The only control knob is the packing fraction
eta, the fraction of the box area covered by disks. At low eta the disks behave
like a gas. Near eta = 0.70 they order into a hexagonal crystal with no
attraction at all, because an ordered arrangement leaves each disk more room to
rattle (more entropy). In 2D the melting goes liquid -> hexatic -> solid
(Bernard & Krauth, 2011).

## Conventions

- Disk diameter = 1. Box is a square of side L with periodic edges.
- eta = N * pi * (1/2)^2 / L^2, so L = sqrt(N * pi / (4 * eta)).
- One sweep = N attempted single-disk moves.
- Short readable numpy code. No try/except, no silent defaults.
- Python: `~/anaconda3/bin/python3`.

## Steps, each with a check

### Step 1 — the sampler  (today)
Start disks on a square grid (no overlaps by construction). Pick a disk, move it
by a uniform random step in [-delta, delta] in x and y, reject if it overlaps
anyone, else accept.

Checks:
1. The overlap counter catches a planted overlap (test the test).
2. Zero overlaps after every sweep, counted by a separate all-pairs routine
   that shares no code with the move.
3. Acceptance falls as delta grows; pick delta for roughly 30-50% acceptance.
4. Two disks in a periodic box: every allowed placement is equally likely, so
   the separation r (for 1 < r < L/2) must have density proportional to r.
   This tests that the sampler is *fair*, not just that it avoids overlaps.

### Step 2 — pressure from contact
Histogram all pair distances to get g(r). Pressure comes from g at contact:

    Z = P A / (N k T) = 1 + 2 eta g(1+)

Checks: g(r) = 0 for r < 1 and g -> 1 at large r; the first and second halves
of the run agree (equilibrated); Z matches the virial series at eta <= 0.3.
The two-term series 1 + 2 eta + 3.128 eta^2 is already 8% low at eta = 0.3, so
the check uses seven known terms (the dropped terms add < 0.001 there).

Result (N = 256, 8000 sweeps per eta): all four pass. Agreement is within 1.5
error bars up to eta = 0.3 (Z = 2.064 +/- 0.006 vs 2.062).

### Step 3 — pictures near melting
Snapshots at eta = 0.65, 0.70, 0.72, each disk colored by its local hexagonal
order |psi_6|. Check: |psi_6| near 1 everywhere for a perfect triangular lattice.

Step 3 as run (N = 288, hexagonal box, 6000 sweeps per eta, each eta started
both from a crystal and from a squeezed liquid): Psi6 rises from ~0.1 at
eta = 0.66 to ~0.8 at 0.74, with the rise between 0.68 and 0.72. The two starts
agree at 0.66 and 0.70 but not at 0.68 or 0.71, so those runs have not
equilibrated. Needs longer runs (in parallel, one process per eta).

Step 3, second pass: 14 independent runs in a process pool (one per eta and
start, seeds from SeedSequence.spawn), 60000 sweeps for 0.68-0.72 and 20000 at
0.66 and 0.74; 4 min on 11 cores. Squeeze rewritten to shrink until the closest
pair touches (the fixed 0.2% steps were almost always rejected at high eta).
Now both starts agree within their fluctuations at every eta. The steep rise
sits between eta = 0.69 and 0.71, where Bernard & Krauth put liquid-hexatic
coexistence (0.700-0.716). There Psi6 swings between 0.3 and 0.75 over
~10000-sweep stretches, the signature of a system flipping between ordered and
disordered. N = 288 cannot tell hexatic from solid. Final configurations are in
results_step3.npz; plot_step3.py redraws the figures without rerunning.

## Open decisions (yours)

- N for production runs (step 1 uses N = 256, a 16 x 16 grid).
- Square grid start vs triangular start for step 3. A square start must melt and
  re-form into hexagons, which is slow at high eta; a triangular lattice needs a
  box with aspect ratio sqrt(3)/2 to fit the periodic edges.
- How many sweeps to discard before measuring (equilibration).
