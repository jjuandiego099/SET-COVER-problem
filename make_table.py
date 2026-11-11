"""Convierte resultados.csv en tabla_resultados.tex (para \\input en el informe)."""
import pandas as pd
d = pd.read_csv("resultados.csv")
g = d.groupby(["m", "n"]).agg(z_ex=("z_exacto", "mean"), t_ex=("t_exacto", "mean"),
                              z_best=("z_ga_mejor", "mean"), z_avg=("z_ga_prom", "mean"),
                              t_ga=("t_ga_prom", "mean"), gap_b=("gap_mejor_pct", "mean"),
                              gap_a=("gap_prom_pct", "mean"), opt=("optimo_hallado", "sum"),
                              k=("instancia", "count")).reset_index()
L = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
     r"$m$ & $n$ & $z^*$ & $t_{ex}$ (s) & $z_{GA}^{mejor}$ & $z_{GA}^{prom}$ & $t_{GA}$ (s) & Gap prom (\%) & \'Optimos \\",
     r"\midrule"]
for _, r in g.iterrows():
    L.append(f"{int(r.m)} & {int(r.n)} & {r.z_ex:.1f} & {r.t_ex:.3f} & {r.z_best:.1f} & {r.z_avg:.2f} "
             f"& {r.t_ga:.2f} & {r.gap_a:.2f} & {int(r.opt)}/{int(r.k)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("tabla_resultados.tex", "w").write("\n".join(L))
print("\n".join(L))
