# Finding C₆₀ from random atoms: global optimization with tight binding

Week 8 agentic, Optimization & the Cosmological Constant. Started Friday 2026-10-30.
Results and numbers for each step are in [RESULTS.md](RESULTS.md).

| Step | What | Status |
|---|---|---|
| 1 | Energy of any arrangement of carbon atoms | **done**, Gate 1 passed |
| 2 | Forces on the atoms | **done**, Gate 2 passed |
| 3 | Relax one cluster downhill; build the C₆₀ target | **done**, Gate 3 passed |
| — | Pictures: snapshots, GIF, rotatable browser viewer | **done** |
| 4 | Random multistart on small clusters, C₁₀ to C₂₄, and on C₆₀ | **done**: fails above ~11 atoms (that is the result) |
| 5 | Simulated annealing (MD) on C₂₀ and C₆₀ | code ready and smoke-tested; full study stopped (too long, ~45 min); needs a shorter plan |
| 6 | Genetic algorithm on C₆₀ (and C₂₀) | **C₆₀ found** in 1 ecology (seed 1, 4536 matings, 2 min); more ecologies and C₂₀ still to run |
| 7 | Compare the three searches at equal cost | script ready (`step7_compare.py`); needs Step 5 results and more GA ecologies |
| 8 | Portfolio writeup | to do |

## The idea in one paragraph

Put 60 carbon atoms at random positions. Ask a computer for the arrangement with the
lowest energy, without telling it what the answer is. The answer should be the
soccer-ball molecule C₆₀ (buckminsterfullerene). Deaven and Ho did this in 1995
(Phys. Rev. Lett. **75**, 288), and showed that the *way* you search matters: a
genetic algorithm with a clever "cut and splice" move found C₆₀, while simpler
searches got stuck. We redo their experiment.

## Words used here

- **Configuration**: the positions of all 60 atoms, i.e. 180 numbers.
- **Energy landscape**: the energy E as a function of those 180 numbers. Low = stable.
- **Local minimum**: a configuration where every small move raises the energy. A
  tangle of carbon chains is one. There are a huge number of them.
- **Global minimum**: the lowest of all local minima. For 60 carbons, that is C₆₀.
- **Force**: minus the slope of E with respect to an atom's position. It points downhill.
- **Relaxation**: following the forces downhill until they vanish. It always ends in
  the *nearest* local minimum, not the global one. We use the L-BFGS method for this.
- **Binding energy E_b**: energy per atom compared with 60 separate atoms. More
  negative = more stable. Graphite is −7.26 eV/atom in this model, C₆₀ is −6.86.
- **E_ref**: the binding energy of the relaxed perfect C₆₀ (−6.8575 eV/atom). A
  search succeeds when it reaches this value.

## Where the optimization is

There are two levels, one inside the other.

**Inner level: relaxation (Steps 2–3).** From any configuration, slide downhill to the
nearest local minimum using the forces. This is the gradient descent from the Tuesday
notebook. On its own it gets stuck: the pictures in RESULTS.md show 60 random atoms
relaxing into a tangle of chains, 0.85 eV/atom above C₆₀.

**Outer level: global search (Steps 4–7).** Choose *where to start* the relaxations so
that one of them lands in C₆₀. Three methods compete, all from the Tuesday notebook:

| Method | Tuesday notebook | What it does here | Expected |
|---|---|---|---|
| Random multistart | Problem 1 | Many random starts, relax each, keep the best | Never finds C₆₀ |
| Simulated annealing | Problems 2 & 4 | Random moves of size step·√T; accept uphill moves sometimes; cool slowly | Stuck in defective cages |
| Genetic algorithm | Problem 3 | A population of relaxed clusters; children made by cut-and-splice | Finds C₆₀ |

**What makes Deaven and Ho's genetic algorithm work.**
1. *Crossover in space, not by atom index.* Atom labels mean nothing, so gluing two
   coordinate lists by index (as in Problem 3) makes nonsense. Instead, cut both parent
   clusters with the same random plane through their centers. Take the top half of one
   and the bottom half of the other, and slide the plane until the child has exactly
   60 atoms.
2. *Relax every child before scoring it.* The search then moves only between local
   minima, which is a far smaller space than all configurations.

**The fair comparison (Step 7).** Give every method the same number of energy+force
calls. Plot the fraction of independent runs that reach E_ref against that number.

## The energy model

Tight binding for carbon from C. H. Xu, C. Z. Wang, C. T. Chan, K. M. Ho,
J. Phys.: Condens. Matter **4**, 6047 (1992). It is the Slater-Koster model of the
PHYS 7324 exercise (`tb.py`, whose `sk()` is copied here). The difference is that the
hoppings shrink smoothly with bond length, so the energy is a smooth function of the
atom positions.

$$E_{\rm tot} = \underbrace{2\sum_{n\ \rm occ}\varepsilon_n}_{\text{electrons: bonding}}
 + \underbrace{\sum_i f\Big(\sum_{j\neq i}\phi(r_{ij})\Big)}_{\text{cores: repulsion}}$$

- Four orbitals per atom (s, px, py, pz), four electrons per atom, two per level. The
  levels εₙ come from diagonalizing a 240×240 Hamiltonian for C₆₀.
- Each hopping is its value at r₀ times $s(r) = (r_0/r)^n \exp\{n[-(r/r_c)^{n_c} + (r_0/r_c)^{n_c}]\}$.
- Pair repulsion: $\phi(r) = \phi_0 (d_0/r)^m \exp\{m[-(r/d_c)^{m_c} + (d_0/d_c)^{m_c}]\}$,
  then passed through a polynomial $f(x) = \sum_{k=0}^{4} C_k x^k$.
- Near the cutoff each function switches over to a cubic that reaches zero smoothly at
  2.6 Å: from 2.45 Å for s(r), from 2.57 Å for φ(r). Beyond 2.6 Å, atoms do not interact.

| Parameter | Value |
|---|---|
| E_s, E_p | −2.99, 3.71 eV |
| V_ssσ, V_spσ, V_ppσ, V_ppπ (at r₀) | −5.0, 4.7, 5.5, −1.55 eV |
| r₀, n, n_c, r_c | 1.536329 Å, 2, 6.5, 2.18 Å |
| φ₀, m, m_c, d_c, d₀ | 8.18555 eV, 3.30304, 8.6655, 2.1052 Å, 1.64 Å |
| C₀ … C₄ | −2.5909765118191, 0.5721151498619, −1.7896349903996e−3, 2.3539221516757e−5, −1.24251169551587e−7 |

All of these numbers were **checked against the paper** (text and Tables 1–2) on
2026-10-30. The PDF stays outside this public repo.

**Three things worth knowing about the code** (details in RESULTS.md):
1. The cubic cutoff tails are *computed* by the paper's rule (value and slope match at
   the start, value and slope zero at 2.6 Å). This reproduces Table 2 of the paper.
2. A free atom has energy 2E_s + 2E_p + C₀, because f(0) = C₀. Binding energies are
   measured from that. The paper's Fig. 1 plots E_tot/N instead, which is 1.151 eV/atom
   lower. Deaven & Ho's energies are lower again, by a constant 1.38 eV/atom; only energy
   differences matter.
3. **Our own addition:** f(x) is a quartic that peaks at x = 146.5 eV and then falls,
   so atoms squeezed together would attract without limit. Beyond x = 100 eV we
   continue f along its tangent line. Real structures have x ≈ 50 eV, so they are
   unaffected. Only overlapping atoms (closer than about 1.2 Å) reach x > 100, which
   random starts and crossover children do all the time.

## Steps

Each step ends with a gate: checks that must pass before the next step starts.

### Step 1: the energy (done)

`carbon_tb.py`: `energy(R)` for a cluster, and `energy_periodic(lattice, basis, kmesh)`
for a crystal (band energy averaged over a k-mesh, as in `tb.py`).

Gate 1: free atoms give binding 0; diamond minimum at 1.54 Å; graphite minimum at
1.42 Å (graphite = one graphene sheet here, since the 3.35 Å layer spacing is beyond
the cutoff); graphite and diamond within 0.1 eV/atom; energy unchanged by rotation.

### Step 2: the forces (done)

`energy_forces(R)` in `carbon_tb.py`. Band part by Hellmann-Feynman,
$\mathbf F_a = -2\sum_{\rm occ}\langle\psi|\partial H/\partial\mathbf R_a|\psi\rangle$; repulsive
part by the chain rule.

Gate 2: forces match finite differences to 1e−6 eV/Å; they sum to zero; there is no
net torque; time one call on 60 atoms.

Caveat: these forces assume the HOMO-LUMO gap is open. If the highest filled and
lowest empty levels cross, the energy has a kink there. This has not caused trouble so
far. If it does in Steps 5–6, the standard fix is a small electronic temperature
(Fermi smearing).

### Step 3: relaxation and the C₆₀ target (done)

`structures.py`: `relax()` (scipy L-BFGS-B), the perfect truncated icosahedron
`c60_ideal()`, and the ring and coordination counters used to recognize C₆₀.

Gate 3: after relaxing, every atom has 3 neighbors (90 bonds); there are exactly two
bond lengths, pentagon edges longer than hexagon-hexagon edges; 12 pentagons and 20
hexagons; a few tenths of an eV/atom above graphite; HOMO-LUMO gap open. The relaxed
cage is saved as `c60_ref.npy` and its energy is **E_ref**.

### Pictures (done)

`viz.py` and `demo_relax.py`: three relaxations shown as snapshots, a rotating GIF,
and `viewer.html`, a rotatable viewer for any browser with a step slider (no installs,
works on the github.io portfolio). Atoms are colored by bond count. Pentagon bonds are red.

### Step 4: random multistart (done)

`step4_multistart.py`. First, the three C₂₀ isomers that Deaven & Ho name (ring, bowl,
cage) are built and relaxed, and their energy differences compared with the paper's.
Then 1000 random starts are run for each N = 10…24, and 720 for C₆₀.

Gate 4: (a) the C₂₀ isomer energies agree with D&H: **passed**, to 0.005 eV/atom.
(b) The lowest structure found is stable when the number of starts doubles:
**failed** for most N. Multistart does not converge above about 11 atoms, never finds
any of the three C₂₀ isomers, and gets no closer than 0.76 eV/atom to C₆₀. That is the
baseline the other two methods must beat.

### Step 5: simulated annealing (code ready, not yet run)

`anneal.py`, `step5_anneal.py`. Molecular dynamics with a thermostat: heat the cluster
to 4000 K, lower the temperature linearly to 0, relax. This is the standard way
annealing is done for carbon clusters. The Metropolis moves of Problem 4 are a weaker
version: in 180 dimensions a random move of every atom must be tiny to be accepted,
while MD moves all atoms together along the forces. A soft spherical wall keeps hot
atoms from flying off. Schedules: 2k to 60k MD steps of 1 fs for C₂₀, 10k to 300k for
C₆₀; 36 runs per schedule for C₂₀, 18 for C₆₀.

Gate 5: on C₂₀, annealing finds the bowl at a lower cost per success than multistart
(which never found it). On C₆₀, report how close it gets and what it makes instead.

Cost: one C₆₀ MD step is one energy+force call (~3 ms), so a 300k-step anneal takes about
15–20 min per core. The full set of schedules was about 45 min on 18 cores and was
stopped on 2026-10-30. A shorter plan: drop the 300k-step schedule (≈ 12 min total), or
run C₂₀ only first (≈ 2 min).

### Step 6: the genetic algorithm (C₆₀ found once)

`ga.py`, `step6_ga.py`, `step6_plots.py`. As in the paper: population p = 4, parents
chosen with weight exp(−E/T) at T = 0.2 eV/atom, cut-and-splice, every child relaxed, no
two members within δE = 0.01 eV/atom, no mutation for C₆₀. Two differences: 18 children
relax at once (one per core), and a child may replace its nearest-energy twin if it is
lower. Success = within 1e−3 eV/atom of E_ref.

Gate 6: it finds C₆₀ from random starts in at least some ecologies (independent runs).

### Step 7: the comparison plot

Success fraction against number of energy+force calls, for all three methods.

### Step 8: portfolio writeup

The GA's best cluster per generation as it folds into a cage (viewer + GIF); the
comparison plot; ring statistics.

## Files

| File | What it does | Run |
|---|---|---|
| `carbon_tb.py` | The model: energy, forces, levels, crystal energy | (library) |
| `structures.py` | C₆₀ builder, random clusters, `relax()`, bonds, rings | (library) |
| `viz.py`, `viewer_template.html` | 3D plots, GIF, browser viewer | (library) |
| `gate1.py` | Step 1 checks, `gate1_bulk.png` | 2 s |
| `gate2.py` | Step 2 checks and timing | 3 s |
| `gate3.py` | Step 3 checks, writes `c60_ref.npy` | 1 s |
| `demo_relax.py` | `relax_snapshots.png`, `relax_random.gif`, `viewer.html` | 15 s |
| `parallel.py` | runs jobs on all 18 cores, one BLAS thread each | (library) |
| `step4_multistart.py` | Step 4, `step4_*.png`, `viewer_step4.html` | 6.5 min |
| `anneal.py`, `step5_anneal.py` | Step 5, `step5_*`, `viewer_step5.html` | ~45 min as written; ~12 min without the 300k schedule |
| `ga.py`, `step6_ga.py`, `step6_plots.py` | Step 6, `ga_runs/`, `step6_*`, `viewer_step6_*.html` | 2–4 min per C₆₀ ecology |
| `step7_compare.py` | Step 7 comparison plot | seconds |

Python: `~/anaconda3/bin/python3`, run from this folder. Everything parallel runs as 18
single-threaded processes (6 performance + 12 efficiency cores). The GPU is not used: numpy
has no GPU path, and Apple's GPU has no float64 eigensolver.

## Rules for the agent

- Short, readable code: numpy, scipy, and matplotlib only (the viewer is plain
  HTML + JavaScript, no libraries).
- No `try/except` fallbacks and no silent defaults.
- Work in small pieces. Stop at each gate and show me the check output before going on.
- Update this file's status table and RESULTS.md at the end of every step.
- Do not commit. I commit, under my name only, with no Co-Authored-By line.
- This fork is public: no unpublished research data in this folder.
