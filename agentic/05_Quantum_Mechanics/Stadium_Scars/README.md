# Quantum Billiards and Scars

A free particle in a hard-walled 2D table. With `hbar = 1` and `2m = 1`,

```
-lap psi = E psi   inside,      psi = 0   on the wall
```

Everything here follows the Natural Steps in `../Quantum_Scars.md`.

## How it works

A square grid of spacing `h` goes over the table, and the Laplacian becomes the
five-point second difference, so the problem turns into the eigenvalue problem
of a sparse matrix. At `h = 1/200` the stadium matrix is 71,003 x 71,003 with
under 5 nonzeros per row; `scipy.sparse.linalg.eigsh` in shift-invert mode
returns the lowest 400 eigenpairs in about 15 seconds.

**The mask is the boundary condition.** One boolean per grid node:

| mask | meaning |
|---|---|
| `True` | `psi` here is an unknown, the solver finds it |
| `False` | `psi` here is known, and it is `0` |

There is no third category, so the matrix has exactly `mask.sum()` rows. The
stencil needs four neighbour values, and at a node against the wall one of them
is off the table, so you have to say what it is:

- `psi_outside = 0` — the term vanishes. Drop that off-diagonal entry and **keep
  the 4** on the diagonal. Hard wall.
- `psi_outside = psi_here` — the term cancels one unit of the diagonal, so the
  **4 becomes a 3**. Reflecting wall.

That is the whole difference, and getting it wrong is silent: the code runs and
returns a clean spectrum of the wrong problem. So `laplacian()` takes
`wall="hard"` or `wall="mirror"`, and check 2 builds both from the same mask.
Their off-diagonal entries are identical; the hard spectrum starts at
`E = 14.8018` and the reflecting one at `E = 0`, a flat wavefunction that
`psi = 0` forbids. Both match their own closed form to better than 1e-12.

## Files

| file | what it does |
|---|---|
| `billiard.py` | grids, masks, the masked Laplacian, the solver, IPR, closed forms |
| `checks_eigenvalues.py` | Step 2. Five checks on a rectangle, where the answer is known |
| `checks_stadium.py` | Step 3. Stadium mask, and what the staircased arc costs |
| `scars.py` | Steps 4 and 5. Rank 400 states by IPR, then look at them |
| `level_statistics.py` | Push Harder. Poisson vs Wigner level spacings |
| `mask_figure.py` | draws the mask itself, so the zeros are visible |

Run any of them directly; `figures/` is written in place. Total runtime is
about two minutes.

## Results

**Step 2, the rectangle (99 x 70 nodes, so `Lx/Ly = 1.41428`, which is `sqrt(2)`
to one part in 10^4).**

| check | result |
|---|---|
| 200 levels vs the exact *discrete* spectrum | agree to 3.6e-14 |
| max error vs the continuum, halving `h` twice | falls **4.00x**, then **4.00x** |
| hard vs reflecting walls, same mask | `E_1 = 14.8018` vs `E_1 = 0` |
| near-degenerate gaps, 99:70 vs 70:70 square | **0** of 199 vs **94** of 199 |

**Step 3, what the curved wall costs.** The mask's area converges as
`2.23 h^1.015`, clean O(h). Its counted perimeter stays ~30% too long at every
`h` and never improves, because a staircase on a 45-degree slope is always
`sqrt(2)` times longer — so the counted perimeter is not usable for anything.
Eigenvalue error has **no single exponent**: `p = 0.97` for the ground state
rising to ~1.6 by level 20, and forcing `p = 1` fits the high levels 3x worse.
Against the rectangle at matched `h` the stadium is only 2-3x worse, because the
strip the staircase removes lies where `|psi|^2` is already near zero. The real
loss is the reference: the rectangle's error is measured against a formula, the
stadium's against an `E(0)` fitted from its own data. Weyl's law is followed
with both the area and perimeter terms to within 2 levels over the first 100,
drifting once the wavelength falls below ~10 grid spacings.

**Steps 4 and 5, and the result that was not expected.** Every rectangle
eigenstate gives `I*A = 2.250000000000` with spread 1.4e-14 — exactly 9/4,
because they are all separable. The stadium spreads around 2.89, near the
random-speckle value of 3.

But ranking by IPR does **not** rank scars. Of the 20 most localised states, 11
bounce straight up and down between the flat walls and 7 pool in the curved cap,
sorted by the momentum anisotropy `alpha_y = <p_y^2>/E` (the sum rule
`<p_x^2> + <p_y^2> = E` holds to 4e-13). Both families sit on **neutrally
stable** orbits; a Heller scar needs an **unstable** one. Remove both and the
most localised state left reaches `I*A = 3.14` against a median of 2.89 — a few
percent. Finding real scars needs a measure tied to one specific unstable orbit,
which means finding that orbit first.

**Push Harder, the level spacings.** Unfold with Weyl's law so the mean gap is 1,
then histogram the gaps. Levels 21-400.

| | gaps below 0.1 | KS to Poisson | KS to Wigner | verdict |
|---|---|---|---|---|
| rectangle 99:70 | 15.3% | **0.077** | 0.284 | Poisson-like |
| quarter stadium, `h = 1/200` | 0.5% | 0.207 | **0.036** | Wigner |
| quarter stadium, `h = 1/150` | 1.1% | 0.205 | **0.032** | Wigner |

Poisson wants 9.5% of gaps below 0.1, Wigner wants 0.8%. The KS 5% cutoff for
379 spacings is 0.070, so the stadium is statistically consistent with the
random-matrix result, at two grid spacings and under two different unfoldings
(Weyl, and a cubic fit to the counting staircase) — so it is a property of the
spectrum, not of the rescaling or the grid.

The rectangle is Poisson-*like* but fails at 5%: 15.3% of gaps sit below 0.1
where Poisson wants 9.5%. That excess is real. 99/70 is a ratio of whole
numbers, so `n^2/Lx^2 + m^2/Ly^2` still throws up systematic near-coincidences;
they are pushed out of exact degeneracy but not out of the spectrum. A truly
irrational aspect ratio would fix it, at the price of walls no longer landing on
grid lines. The contrast does not depend on it: 15.3% against 0.5% is not close.

## Figures

- `figures/eigenvalues.png` — the matrix, hard vs reflecting walls, solver vs
  exact spectrum, and h^2 convergence
- `figures/mask.png` — which nodes are unknowns, which stencil links are lost to
  the wall, and the exact zeros in the ground state
- `figures/staircase_cost.png` — curved wall vs grid-aligned wall
- `figures/scars_top20.png` — the 20 most localised states, labelled by family
- `figures/balanced_top12.png` — the same ranking with both families removed
- `figures/ipr_distribution.png` — one value vs a spread, and where the top of
  the list lives
- `figures/level_statistics.png` — Poisson becoming Wigner

## Not done

A real periodic orbit. The remaining piece is to root-find the bounce map for an
unstable orbit, measure each state's weight in a thin band around it, and
overlay the orbit on the states that score highest. That is the measurement that
would actually identify scars, and the IPR result above is the reason to do it
rather than an optional extra.
