"""
Set Cover: modelo exacto (PuLP + CBC) y Algoritmo Genetico.
Problema: dado un universo U de m elementos y n subconjuntos S_j con costo c_j,
elegir la coleccion de menor costo que cubra todos los elementos.
"""
import time
from dataclasses import dataclass
import numpy as np
import pulp


# ----------------------------------------------------------------- Instancias
@dataclass
class Instance:
    A: np.ndarray          # matriz de cobertura (m x n), A[i, j] = 1 si S_j cubre i
    c: np.ndarray          # costos (n,)
    seed: int = 0

    @property
    def m(self):
        return self.A.shape[0]

    @property
    def n(self):
        return self.A.shape[1]


def generate_instance(m=50, n=30, density=0.15, cost_range=(1, 10), seed=0):
    """Instancia aleatoria factible (cada elemento queda cubierto al menos una vez)."""
    rng = np.random.default_rng(seed)
    A = (rng.random((m, n)) < density).astype(np.int8)
    for i in range(m):                      # garantiza factibilidad
        if A[i].sum() == 0:
            A[i, rng.integers(n)] = 1
    for j in range(n):                      # evita subconjuntos vacios
        if A[:, j].sum() == 0:
            A[rng.integers(m), j] = 1
    c = rng.integers(cost_range[0], cost_range[1] + 1, size=n).astype(float)
    return Instance(A=A, c=c, seed=seed)


# --------------------------------------------------------------- Metodo exacto
def solve_exact(inst: Instance, time_limit=60, msg=False):
    """
    min  sum_j c_j x_j
    s.a. sum_{j : i in S_j} x_j >= 1   para todo i in U
         x_j in {0,1}
    """
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(inst.n)]
    prob += pulp.lpSum(inst.c[j] * x[j] for j in range(inst.n))
    for i in range(inst.m):
        prob += pulp.lpSum(x[j] for j in range(inst.n) if inst.A[i, j]) >= 1, f"cover_{i}"

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    elapsed = time.perf_counter() - t0

    sol = np.array([int(round(v.value() or 0)) for v in x])
    return {
        "status": pulp.LpStatus[prob.status],
        "objective": float(inst.c @ sol),
        "solution": sol,
        "time": elapsed,
        "selected": [int(j) for j in np.where(sol == 1)[0]],
    }


# ------------------------------------------------------------ Algoritmo Genetico
@dataclass
class GAParams:
    pop_size: int = 60
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02      # probabilidad por gen
    tournament_k: int = 3
    elite: int = 2
    patience: int = 50               # generaciones sin mejora para parar
    init_prob: float = 0.2           # prob. de que un gen inicie en 1
    seed: int = 0


def is_feasible(inst, x):
    return bool(np.all(inst.A[:, x == 1].sum(axis=1) >= 1)) if x.any() else False


def repair(inst, x, rng):
    """
    1) Construccion: mientras haya elementos sin cubrir, agrega el subconjunto
       de menor costo por elemento nuevo cubierto.
    2) Poda: elimina (en orden aleatorio) subconjuntos redundantes.
    """
    x = x.copy()
    A, c = inst.A, inst.c
    cover = A[:, x == 1].sum(axis=1).astype(int) if x.any() else np.zeros(inst.m, dtype=int)
    unc = cover == 0
    while unc.any():
        gains = A[unc].sum(axis=0).astype(float)
        gains[x == 1] = 0
        cand = gains > 0
        ratio = np.where(cand, c / np.where(cand, gains, 1), np.inf)
        j = int(np.argmin(ratio))
        x[j] = 1
        cover += A[:, j]
        unc = cover == 0
    for j in rng.permutation(np.where(x == 1)[0]):
        rows = A[:, j] == 1
        if np.all(cover[rows] >= 2):
            x[j] = 0
            cover -= A[:, j]
    return x


def run_ga(inst: Instance, p: GAParams = GAParams()):
    rng = np.random.default_rng(p.seed)
    t0 = time.perf_counter()
    n = inst.n

    pop = np.array([repair(inst, (rng.random(n) < p.init_prob).astype(np.int8), rng)
                    for _ in range(p.pop_size)])
    fit = pop @ inst.c

    i0 = int(np.argmin(fit))
    best_x, best_f = pop[i0].copy(), float(fit[i0])
    hist_best, hist_mean = [best_f], [float(fit.mean())]
    stall, gen = 0, 0

    for gen in range(1, p.generations + 1):
        order = np.argsort(fit)
        new = [pop[i].copy() for i in order[:p.elite]]          # elitismo

        def tournament():
            idx = rng.integers(0, p.pop_size, p.tournament_k)
            return pop[idx[np.argmin(fit[idx])]]

        while len(new) < p.pop_size:
            p1, p2 = tournament(), tournament()
            if rng.random() < p.crossover_rate:                  # cruce uniforme
                mask = rng.random(n) < 0.5
                child = np.where(mask, p1, p2).astype(np.int8)
            else:
                child = p1.copy()
            flip = rng.random(n) < p.mutation_rate               # mutacion bit-flip
            child[flip] = 1 - child[flip]
            new.append(repair(inst, child, rng))

        pop = np.array(new)
        fit = pop @ inst.c
        i = int(np.argmin(fit))
        if fit[i] < best_f - 1e-9:
            best_f, best_x, stall = float(fit[i]), pop[i].copy(), 0
        else:
            stall += 1
        hist_best.append(best_f)
        hist_mean.append(float(fit.mean()))
        if stall >= p.patience:                                  # criterio de parada
            break

    return {
        "objective": best_f,
        "solution": best_x,
        "selected": [int(j) for j in np.where(best_x == 1)[0]],
        "time": time.perf_counter() - t0,
        "generations": gen,
        "hist_best": hist_best,
        "hist_mean": hist_mean,
        "feasible": is_feasible(inst, best_x),
    }
