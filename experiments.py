"""Experimentos: exacto vs GA en varias instancias. Genera resultados.csv"""
import numpy as np, pandas as pd
from setcover import generate_instance, solve_exact, run_ga, GAParams

import sys
SIZES = [(200, 150)] if "--last" in sys.argv else [(30, 20), (60, 40), (100, 80), (150, 120), (200, 150)]
INSTANCES_PER_SIZE = 3
GA_RUNS = 10          # corridas independientes del GA (es estocastico)

rows = []
for (m, n) in SIZES:
    for s in range(INSTANCES_PER_SIZE):
        inst = generate_instance(m, n, density=0.1, seed=1000 * m + s)
        ex = solve_exact(inst, time_limit=120)
        vals, times = [], []
        for r in range(GA_RUNS):
            g = run_ga(inst, GAParams(seed=r))
            vals.append(g["objective"]); times.append(g["time"])
        best, mean = min(vals), float(np.mean(vals))
        rows.append(dict(m=m, n=n, instancia=s, estado_exacto=ex["status"],
                         z_exacto=ex["objective"], t_exacto=ex["time"],
                         z_ga_mejor=best, z_ga_prom=mean, z_ga_std=float(np.std(vals)),
                         t_ga_prom=float(np.mean(times)),
                         gap_mejor_pct=100 * (best - ex["objective"]) / ex["objective"],
                         gap_prom_pct=100 * (mean - ex["objective"]) / ex["objective"],
                         optimo_hallado=int(abs(best - ex["objective"]) < 1e-6)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("resultados.csv", index=False)
print("Guardado resultados.csv")
