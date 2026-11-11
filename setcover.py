"""
setcover.py
===========
Núcleo del proyecto de clase: problema de COBERTURA DE CONJUNTOS (SET COVER).

Este módulo contiene todo lo necesario para resolver el problema de dos maneras
y poder compararlas sobre las MISMAS instancias:

    1. Método EXACTO  -> modelo de programación entera binaria resuelto con
                         el solver CBC a través de PuLP        (solve_exact)
    2. METAHEURÍSTICA -> Algoritmo Genético con operador de
                         reparación                            (run_ga)

Definición del problema
-----------------------
Se tiene un universo U = {1, ..., m} de elementos y n subconjuntos S_1, ..., S_n
de U, cada uno con un costo c_j > 0. Hay que elegir la colección de subconjuntos
de MENOR COSTO TOTAL cuya unión cubra todos los elementos de U.

Formulación matemática (la que implementa solve_exact)
------------------------------------------------------
    Datos:        a[i][j] = 1 si el elemento i pertenece al subconjunto S_j, 0 si no
                  c[j]    = costo del subconjunto S_j

    Variables:    x[j] ∈ {0, 1}   (x[j] = 1  <=>  se elige el subconjunto S_j)

    Minimizar     Z = sum_j  c[j] * x[j]                    (costo total)

    Sujeto a      sum_j a[i][j] * x[j] >= 1   para todo i   (cada elemento
                                                             cubierto >= 1 vez)
                  x[j] ∈ {0, 1}               para todo j

Convenciones usadas en todo el archivo
--------------------------------------
    m    número de elementos del universo (filas de la matriz A)
    n    número de subconjuntos (columnas de la matriz A)
    A    matriz de cobertura m x n con valores 0/1
    c    vector de costos de longitud n
    x    solución: vector de longitud n con valores 0/1.
         Es el "cromosoma" del GA y las variables de decisión del modelo exacto.
    Costo de una solución x = c @ x  (producto punto = sum_j c[j]*x[j]).

Estructura del archivo
----------------------
    Instance          estructura de datos que representa un problema
    generate_instance crea instancias aleatorias factibles
    solve_exact       método exacto (PuLP + CBC)
    GAParams          parámetros del Algoritmo Genético
    is_feasible       verifica si una solución cubre todo
    repair            operador de reparación del GA (construcción voraz + poda)
    run_ga            el Algoritmo Genético completo

Dependencias: numpy, pulp (se recomienda pulp==2.9.0; la serie 4.x cambió la API).
"""
import time
from dataclasses import dataclass

import numpy as np
import pulp


# =============================================================================
# 1. REPRESENTACIÓN Y GENERACIÓN DE INSTANCIAS
# =============================================================================
@dataclass
class Instance:
    """
    Un problema Set Cover concreto (una "instancia").

    Atributos
    ---------
    A : np.ndarray de forma (m, n)
        Matriz de cobertura. A[i, j] = 1 si el subconjunto j cubre al elemento i.
        Filas = elementos del universo; columnas = subconjuntos.
    c : np.ndarray de forma (n,)
        Costo de cada subconjunto. c[j] es el costo de elegir el subconjunto j.
    seed : int
        Semilla con la que se generó la instancia (solo informativa).
    """
    A: np.ndarray
    c: np.ndarray
    seed: int = 0

    @property
    def m(self):
        """Número de elementos del universo (filas de A)."""
        return self.A.shape[0]

    @property
    def n(self):
        """Número de subconjuntos disponibles (columnas de A)."""
        return self.A.shape[1]


def generate_instance(m=50, n=30, density=0.15, cost_range=(1, 10), seed=0):
    """
    Genera una instancia ALEATORIA y FACTIBLE de Set Cover.

    Parámetros
    ----------
    m : int
        Número de elementos del universo.
    n : int
        Número de subconjuntos disponibles.
    density : float en (0, 1)
        Probabilidad de que A[i, j] valga 1. Es la proporción esperada de unos
        en la matriz: con density=0.10, cada subconjunto cubre en promedio el
        10 % de los elementos.
    cost_range : (int, int)
        Rango [mínimo, máximo] de los costos enteros (ambos extremos incluidos).
    seed : int
        Semilla del generador aleatorio. La misma semilla produce SIEMPRE la
        misma instancia, lo que hace los experimentos reproducibles.

    Retorna
    -------
    Instance
        Instancia con A (int8, 0/1) y c (float).

    Garantías
    ---------
    - Todo elemento queda cubierto por al menos un subconjunto (si no, el
      problema no tendría solución).
    - Ningún subconjunto queda vacío (un subconjunto vacío no sirve de nada).
    """
    # Generador aleatorio moderno de NumPy; con semilla fija es determinista.
    rng = np.random.default_rng(seed)

    # Cada casilla vale 1 con probabilidad `density` (comparar un número
    # uniforme en [0,1) contra la densidad produce un Bernoulli).
    A = (rng.random((m, n)) < density).astype(np.int8)

    # Factibilidad: si un elemento (fila) quedó sin ningún 1, se le asigna un
    # subconjunto elegido al azar que lo cubra.
    for i in range(m):
        if A[i].sum() == 0:
            A[i, rng.integers(n)] = 1

    # Evita columnas vacías: si un subconjunto no cubre nada, se le agrega un
    # elemento al azar.
    for j in range(n):
        if A[:, j].sum() == 0:
            A[rng.integers(m), j] = 1

    # Costos enteros uniformes en [cost_range[0], cost_range[1]]; se guardan
    # como float para que el producto c @ x sea siempre de tipo real.
    # (+1 porque el límite superior de rng.integers es exclusivo)
    c = rng.integers(cost_range[0], cost_range[1] + 1, size=n).astype(float)
    return Instance(A=A, c=c, seed=seed)


# =============================================================================
# 2. MÉTODO EXACTO: PROGRAMACIÓN ENTERA BINARIA (PuLP + CBC)
# =============================================================================
def solve_exact(inst: Instance, time_limit=60, msg=False):
    """
    Resuelve la instancia de forma EXACTA con el solver CBC (vía PuLP).

    Aquí NO se programa cómo buscar la solución: solo se DESCRIBE el problema
    (variables, objetivo, restricciones) y CBC aplica internamente su algoritmo
    de ramificación y corte (branch-and-cut), que garantiza el óptimo si
    termina dentro del límite de tiempo.

    Modelo:
        min   sum_j c[j] * x[j]
        s.a.  sum_{j : i in S_j} x[j] >= 1     para cada elemento i
              x[j] in {0, 1}

    Parámetros
    ----------
    inst : Instance
        Problema a resolver.
    time_limit : int
        Tiempo máximo del solver en segundos. Si se agota, CBC devuelve la
        mejor solución encontrada, que podría no ser óptima (ver 'status').
    msg : bool
        Si es True, muestra en consola el registro interno de CBC.

    Retorna
    -------
    dict con las claves:
        'status'    : estado del solver ('Optimal' = óptimo demostrado)
        'objective' : costo de la solución encontrada (float)
        'solution'  : vector x de 0/1 (np.ndarray de longitud n)
        'time'      : tiempo de resolución en segundos (incluye el arranque
                      del programa CBC, por eso varía entre ejecuciones)
        'selected'  : lista de índices j con x[j] = 1 (subconjuntos elegidos,
                      numerados desde 0)
    """
    # Crea el problema indicando que es de MINIMIZACIÓN.
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)

    # Variables de decisión x_0 ... x_{n-1}, BINARIAS (solo pueden valer 0 o 1).
    x = [pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(inst.n)]

    # FUNCIÓN OBJETIVO: al sumar una expresión (sin comparador) al problema,
    # PuLP la interpreta como el objetivo: sum_j c[j] * x[j].
    prob += pulp.lpSum(inst.c[j] * x[j] for j in range(inst.n))

    # RESTRICCIONES DE COBERTURA: una por elemento i. Se suman únicamente las
    # variables de los subconjuntos que contienen a i (A[i, j] = 1) y se exige
    # que la suma sea al menos 1, es decir, que i quede cubierto.
    for i in range(inst.m):
        prob += (pulp.lpSum(x[j] for j in range(inst.n) if inst.A[i, j]) >= 1,
                 f"cover_{i}")

    # Se mide el tiempo de pared alrededor de la llamada al solver.
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    elapsed = time.perf_counter() - t0

    # Lectura de la solución. CBC puede devolver 0.9999999 en lugar de 1 por
    # redondeo numérico, por eso se redondea; `or 0` cubre el caso de valor None.
    sol = np.array([int(round(v.value() or 0)) for v in x])
    return {
        "status": pulp.LpStatus[prob.status],      # p. ej. 'Optimal'
        "objective": float(inst.c @ sol),          # costo = sum c[j]*x[j]
        "solution": sol,
        "time": elapsed,
        "selected": [int(j) for j in np.where(sol == 1)[0]],
    }


# =============================================================================
# 3. ALGORITMO GENÉTICO
# =============================================================================
@dataclass
class GAParams:
    """
    Parámetros del Algoritmo Genético (con sus valores por defecto).

    Atributos
    ---------
    pop_size : int
        Tamaño de la población (soluciones por generación).
    generations : int
        Número máximo de generaciones (tope de cómputo).
    crossover_rate : float
        Probabilidad de aplicar cruce a un par de padres. Si no se cruza, el
        hijo es una copia del primer padre.
    mutation_rate : float
        Probabilidad de que cada gen (bit) del hijo se invierta.
    tournament_k : int
        Tamaño del torneo de selección: cuántos individuos compiten por ser
        padre. Más grande = mayor presión selectiva.
    elite : int
        Cantidad de mejores individuos que pasan sin cambios a la siguiente
        generación (elitismo).
    patience : int
        Criterio de parada por estancamiento: el algoritmo termina si el mejor
        costo no mejora durante este número de generaciones seguidas.
    init_prob : float
        Probabilidad de que cada gen valga 1 al crear la población inicial.
    seed : int
        Semilla del generador aleatorio (el GA es estocástico; la semilla
        permite repetir exactamente una corrida).
    """
    pop_size: int = 60
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02      # probabilidad POR GEN
    tournament_k: int = 3
    elite: int = 2
    patience: int = 50               # generaciones sin mejora para parar
    init_prob: float = 0.2           # prob. de que un gen inicie en 1
    seed: int = 0


def is_feasible(inst, x):
    """
    Indica si la solución x CUBRE todos los elementos del universo.

    Para cada elemento (fila de A) suma cuántos subconjuntos elegidos lo
    contienen; la solución es factible si todas esas sumas son >= 1.
    Una solución vacía (ningún subconjunto elegido) nunca es factible.

    Parámetros
    ----------
    inst : Instance
    x : np.ndarray de 0/1, longitud n

    Retorna
    -------
    bool
    """
    if not x.any():                       # nada elegido => no cubre nada
        return False
    # A[:, x == 1] conserva solo las columnas elegidas; sum(axis=1) cuenta, por
    # fila, cuántos subconjuntos elegidos cubren ese elemento.
    return bool(np.all(inst.A[:, x == 1].sum(axis=1) >= 1))


def repair(inst, x, rng):
    """
    OPERADOR DE REPARACIÓN: convierte cualquier vector x en una solución
    FACTIBLE y SIN subconjuntos redundantes.

    Motivo: el cruce y la mutación pueden producir hijos que dejan elementos
    sin cubrir o que incluyen subconjuntos innecesarios. En lugar de penalizar
    esas soluciones, se arreglan, de modo que todo individuo de la población
    siempre es válido y la función de aptitud es simplemente el costo.

    Fase 1 - CONSTRUCCIÓN VORAZ (agrega lo que falta):
        Mientras haya elementos sin cubrir, agrega el subconjunto con menor
        razón  costo / (número de elementos descubiertos que cubriría),
        siguiendo la heurística de Chvátal.
    Fase 2 - PODA (quita lo que sobra):
        Recorre los subconjuntos elegidos en orden aleatorio y elimina los
        redundantes, es decir, aquellos cuyos elementos están todos cubiertos
        también por otros subconjuntos elegidos.

    Parámetros
    ----------
    inst : Instance
    x : np.ndarray de 0/1, longitud n (NO se modifica; se trabaja sobre una copia)
    rng : np.random.Generator
        Generador aleatorio (se usa para el orden aleatorio de la poda).

    Retorna
    -------
    np.ndarray de 0/1 (solución factible, sin redundancias).
    """
    x = x.copy()                       # no alterar el vector que recibimos
    A, c = inst.A, inst.c

    # cover[i] = cuántos subconjuntos elegidos cubren al elemento i.
    if x.any():
        cover = A[:, x == 1].sum(axis=1).astype(int)
    else:
        cover = np.zeros(inst.m, dtype=int)
    unc = cover == 0                   # unc[i] = True si el elemento i está SIN cubrir

    # ---- Fase 1: construcción voraz ----------------------------------------
    # Termina siempre: todo elemento descubierto tiene algún subconjunto que lo
    # contiene y que aún no está elegido, así que cada vuelta cubre >= 1 elemento.
    while unc.any():
        # gains[j] = cuántos elementos AÚN descubiertos cubriría el subconjunto j
        gains = A[unc].sum(axis=0).astype(float)
        gains[x == 1] = 0              # los ya elegidos no aportan nada nuevo
        cand = gains > 0               # solo se consideran los que aportan algo
        # Costo por elemento nuevo cubierto; np.where evita dividir por cero
        # y deja en infinito a los subconjuntos que no son candidatos.
        ratio = np.where(cand, c / np.where(cand, gains, 1), np.inf)
        j = int(np.argmin(ratio))      # el más barato por elemento cubierto
        x[j] = 1                       # se agrega a la solución
        cover += A[:, j]               # se actualiza la cobertura
        unc = cover == 0               # se recalculan los elementos pendientes

    # ---- Fase 2: poda de redundancia ---------------------------------------
    # El orden aleatorio evita quitar siempre los mismos subconjuntos primero.
    for j in rng.permutation(np.where(x == 1)[0]):
        rows = A[:, j] == 1            # elementos que cubre el subconjunto j
        # Si todos esos elementos están cubiertos por al menos 2 subconjuntos
        # elegidos, j es redundante: quitarlo no deja nada descubierto.
        if np.all(cover[rows] >= 2):
            x[j] = 0
            cover -= A[:, j]
    return x


def run_ga(inst: Instance, p: GAParams = GAParams()):
    """
    Ejecuta UNA corrida del Algoritmo Genético sobre la instancia.

    Diseño
    ------
    Codificación : cromosoma binario x de longitud n (un gen por subconjunto).
    Aptitud      : costo c @ x (se MINIMIZA). Como todo individuo se repara,
                   siempre es factible y no se necesitan penalizaciones.
    Inicialización: genes aleatorios (1 con prob. init_prob) + reparación.
    Selección    : torneo de tamaño tournament_k.
    Cruce        : uniforme, con probabilidad crossover_rate.
    Mutación     : inversión de bits, con probabilidad mutation_rate por gen.
    Reparación   : repair() aplicada a cada hijo.
    Reemplazo    : generacional con elitismo (pasan los `elite` mejores).
    Parada       : se alcanzan `generations` generaciones, o el mejor costo no
                   mejora durante `patience` generaciones seguidas.

    Parámetros
    ----------
    inst : Instance
    p : GAParams
        Parámetros del algoritmo.

    Retorna
    -------
    dict con las claves:
        'objective'   : costo de la mejor solución hallada en TODA la corrida
        'solution'    : vector x de esa solución (np.ndarray de 0/1)
        'selected'    : índices de los subconjuntos elegidos
        'time'        : tiempo de la corrida en segundos
        'generations' : generaciones ejecutadas hasta detenerse
        'hist_best'   : mejor costo hallado hasta cada generación (para graficar
                        la convergencia; incluye la generación 0)
        'hist_mean'   : costo promedio de la población en cada generación
        'feasible'    : True si la solución final cubre todos los elementos
    """
    rng = np.random.default_rng(p.seed)    # azar reproducible mediante la semilla
    t0 = time.perf_counter()
    n = inst.n

    # ---- Población inicial --------------------------------------------------
    # pop es una matriz (pop_size x n): cada fila es un individuo (una solución).
    # Cada individuo: gen = 1 con probabilidad init_prob, luego se repara.
    pop = np.array([repair(inst, (rng.random(n) < p.init_prob).astype(np.int8), rng)
                    for _ in range(p.pop_size)])
    fit = pop @ inst.c                     # aptitud (costo) de TODOS los individuos de una vez

    # Mejor solución encontrada hasta ahora (se conserva aunque la población cambie).
    i0 = int(np.argmin(fit))
    best_x, best_f = pop[i0].copy(), float(fit[i0])
    hist_best, hist_mean = [best_f], [float(fit.mean())]   # historiales (generación 0)
    stall, gen = 0, 0                      # contador de estancamiento y generación actual

    # ---- Bucle evolutivo ----------------------------------------------------
    for gen in range(1, p.generations + 1):
        # ELITISMO: los `elite` individuos de menor costo pasan intactos.
        order = np.argsort(fit)            # índices ordenados de menor a mayor costo
        new = [pop[i].copy() for i in order[:p.elite]]

        def tournament():
            """Selección por torneo: toma tournament_k individuos al azar y
            devuelve el de menor costo (el más apto)."""
            idx = rng.integers(0, p.pop_size, p.tournament_k)
            return pop[idx[np.argmin(fit[idx])]]

        # Se generan hijos hasta completar la nueva población.
        while len(new) < p.pop_size:
            p1, p2 = tournament(), tournament()            # dos padres

            # CRUCE UNIFORME: cada gen del hijo se toma de p1 o p2 con
            # probabilidad 0.5, según una máscara booleana aleatoria.
            if rng.random() < p.crossover_rate:
                mask = rng.random(n) < 0.5
                child = np.where(mask, p1, p2).astype(np.int8)
            else:
                child = p1.copy()                          # sin cruce: copia de p1

            # MUTACIÓN: cada gen se invierte (0 -> 1, 1 -> 0) con prob. mutation_rate.
            flip = rng.random(n) < p.mutation_rate
            child[flip] = 1 - child[flip]

            # REPARACIÓN: garantiza que el hijo sea factible y sin redundancias.
            new.append(repair(inst, child, rng))

        # REEMPLAZO GENERACIONAL: la nueva población sustituye a la anterior.
        pop = np.array(new)
        fit = pop @ inst.c

        # ¿Mejoró el mejor histórico? (1e-9 evita falsas mejoras por redondeo)
        i = int(np.argmin(fit))
        if fit[i] < best_f - 1e-9:
            best_f, best_x, stall = float(fit[i]), pop[i].copy(), 0
        else:
            stall += 1                      # otra generación sin mejora

        # Historiales para las gráficas de convergencia.
        hist_best.append(best_f)
        hist_mean.append(float(fit.mean()))

        # CRITERIO DE PARADA por estancamiento.
        if stall >= p.patience:
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
