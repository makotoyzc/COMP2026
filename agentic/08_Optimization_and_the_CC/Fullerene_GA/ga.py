"""Genetic algorithm of Deaven & Ho, PRL 75, 288 (1995), for carbon clusters.

Population: p relaxed clusters. Repeat:
  1. pick two parents, each with probability ~ exp(-E/T)   (E in eV/atom, T = 0.2)
  2. cut-and-splice them into a child                       (cut_splice below)
  3. relax the child to its nearest local minimum
  4. put it in the population if it beats the worst member and no member is within
     dE = 0.01 eV/atom of it (we let it replace its nearest-energy twin if lower)
A fraction mu of children is mutated before relaxing (first mutation type of the paper).

Parallel: WORKERS children are relaxed at the same time. Whenever one finishes it is
inserted, and a new child is made from the current population and sent off. This is the
same steady-state algorithm as in the paper, run 18 matings at a time.
"""
import parallel  # first: one BLAS thread per process

import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

import carbon_tb as tb
import structures as st


def random_rotation(rng):
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.diag(Rm))
    return Q if np.linalg.det(Q) > 0 else -Q


def cut_splice(A, B, rng):
    """Child of parents A and B (both (N, 3)). Turn each parent to its own random
    orientation (the paper's random plane through *each* parent), cut both with the plane
    z = 0 through their centers, take A above it and B below. If that is not N atoms, slide
    A up and B down by the same distance d until it is (Deaven & Ho, p. 289).
    Without the independent rotations, children inherit their parents' orientation and
    related members stay aligned, so two half-bowls are always glued back into a bowl."""
    N = len(A)
    n = np.array([0.0, 0.0, 1.0])
    A = (A - A.mean(0)) @ random_rotation(rng).T
    B = (B - B.mean(0)) @ random_rotation(rng).T
    a, b = A @ n, B @ n
    # A atom i is in the child once d > -a_i, B atom j once d > b_j: the child has
    # N atoms for d between the N-th and (N+1)-th smallest of these thresholds.
    t = np.sort(np.concatenate([-a, b]))
    d = 0.5 * (t[N - 1] + t[N])
    child = np.vstack([A[a > -d] + d * n, B[b < d] - d * n])
    assert len(child) == N
    return child


def mutate(R, rng):
    """Move a random atom one bond length (1.4 A) in a random direction, 5-50 times;
    after each move push apart any atoms closer than 1.0 A."""
    R = R.copy()
    for _ in range(rng.integers(5, 51)):
        k = rng.integers(len(R))
        u = rng.normal(size=3)
        R[k] += 1.4 * u / np.linalg.norm(u)
        d = R - R[k]
        r = np.linalg.norm(d, axis=1)
        close = (r < 1.0) & (r > 0)
        R[close] += (1.0 - r[close])[:, None] * d[close] / r[close][:, None]
    return R


def relax_child(R, fmax=1e-3, maxiter=3000):
    """Runs in a worker process."""
    R, E, info = st.relax(R, fmax=fmax, maxiter=maxiter)
    return R, tb.binding(E, len(R)), info["nfev"]


class Population:
    def __init__(self, members, T=0.2, dE=0.01):
        self.R = [m[0] for m in members]
        self.E = [m[1] for m in members]
        self.T, self.dE = T, dE

    def parents(self, rng):
        E = np.array(self.E)
        w = np.exp(-(E - E.min()) / self.T)
        i, j = rng.choice(len(E), size=2, replace=False, p=w / w.sum())
        return self.R[i], self.R[j]

    def offer(self, R, E):
        """Insert the child if it earns a place. Returns True if inserted."""
        E_arr = np.array(self.E)
        twin = np.flatnonzero(np.abs(E_arr - E) < self.dE)
        if len(twin):
            # Only the nearest-energy twin may be replaced: a copy of a member then only
            # replaces its original, so copies of the best cannot crowd out the others.
            k = twin[np.argmin(np.abs(E_arr[twin] - E))]
            if E >= E_arr[k]:
                return False
        else:
            k = int(np.argmax(E_arr))
            if E >= E_arr[k]:
                return False
        self.R[k], self.E[k] = R, E
        return True


def run(N, seed, mu, p=4, matings=6000, T=0.2, dE=0.01, workers=parallel.WORKERS,
        E_target=None, start_radius=None, log_every=500):
    """One GA ecology from random starts. Stops after `matings` children, or as soon as
    the best member is within 1e-3 eV/atom of E_target (if given).
    Returns a history dict: per finished mating the calls so far and the population
    min/max, plus every new best structure."""
    rng = np.random.default_rng(seed)
    radius = start_radius or 1.25 * N ** (1 / 3)
    hist = dict(calls=[], Emin=[], Emax=[], best=[], best_E=[], best_at=[], accepted=0)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        starts = [st.random_cluster(N, rng, radius) for _ in range(p)]
        first = list(ex.map(relax_child, starts))
        calls = sum(f[2] for f in first)
        pop = Population([(f[0], f[1]) for f in first], T, dE)
        best = min(pop.E)

        def submit():
            A, B = pop.parents(rng)
            child = cut_splice(A, B, rng)
            if rng.random() < mu:
                child = mutate(child, rng)
            return ex.submit(relax_child, child)

        running = {submit() for _ in range(min(workers, matings))}
        done_count = 0
        while running:
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                R, E, nfev = fut.result()
                calls += nfev
                done_count += 1
                hist["accepted"] += pop.offer(R, E)
                hist["calls"].append(calls)
                hist["Emin"].append(min(pop.E))
                hist["Emax"].append(max(pop.E))
                if min(pop.E) < best - 1e-6:
                    best = min(pop.E)
                    hist["best"].append(pop.R[int(np.argmin(pop.E))].copy())
                    hist["best_E"].append(best)
                    hist["best_at"].append(done_count)
                if log_every and done_count % log_every == 0:
                    print(f"  seed {seed}: {done_count} matings, {calls} calls, best {best:+.4f},"
                          f" worst {max(pop.E):+.4f} eV/atom, {time.perf_counter() - t0:.0f} s",
                          flush=True)
            if E_target is not None and best < E_target + 1e-3:
                ex.shutdown(wait=True, cancel_futures=True)   # children still relaxing are dropped
                break
            while len(running) < workers and done_count + len(running) < matings:
                running.add(submit())
    hist.update(pop_R=pop.R, pop_E=pop.E, matings=done_count, seconds=time.perf_counter() - t0,
                init_calls=sum(f[2] for f in first))
    return hist
