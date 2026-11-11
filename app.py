"""Interfaz visual (Streamlit). Ejecutar con:  streamlit run app.py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from setcover import generate_instance, solve_exact, run_ga, GAParams

st.set_page_config(page_title="Set Cover: Exacto vs GA", layout="wide")
st.title("Set Cover: método exacto vs Algoritmo Genético")

# ------------------------------------------------------------------ Sidebar
st.sidebar.header("Instancia")
m = st.sidebar.slider("Elementos del universo (m)", 10, 300, 60, 10)
n = st.sidebar.slider("Subconjuntos (n)", 10, 300, 40, 10)
dens = st.sidebar.slider("Densidad de la matriz", 0.02, 0.50, 0.10, 0.01)
cmax = st.sidebar.slider("Costo máximo", 1, 50, 10)
seed = st.sidebar.number_input("Semilla de la instancia", 0, 10**6, 1)

st.sidebar.header("Solver exacto")
tlim = st.sidebar.number_input("Límite de tiempo (s)", 1, 3600, 60)

st.sidebar.header("Algoritmo Genético")
pop = st.sidebar.slider("Tamaño de población", 10, 300, 60, 10)
gens = st.sidebar.slider("Generaciones máximas", 10, 1000, 200, 10)
pc = st.sidebar.slider("Prob. de cruce", 0.0, 1.0, 0.9, 0.05)
pm = st.sidebar.slider("Prob. de mutación (por gen)", 0.0, 0.2, 0.02, 0.005)
k = st.sidebar.slider("Tamaño del torneo", 2, 10, 3)
elite = st.sidebar.slider("Élite", 0, 10, 2)
pat = st.sidebar.slider("Paciencia (parada)", 5, 300, 50, 5)
runs = st.sidebar.slider("Corridas independientes del GA", 1, 30, 5)
gseed = st.sidebar.number_input("Semilla base del GA", 0, 10**6, 0)

if st.button("Ejecutar ambos métodos", type="primary"):
    inst = generate_instance(m, n, dens, (1, cmax), int(seed))
    with st.spinner("Resolviendo con el solver exacto..."):
        ex = solve_exact(inst, time_limit=int(tlim))
    with st.spinner("Ejecutando el Algoritmo Genético..."):
        res = [run_ga(inst, GAParams(pop, gens, pc, pm, k, elite, pat, 0.2, int(gseed) + r))
               for r in range(runs)]
    st.session_state["out"] = (inst, ex, res)

if "out" not in st.session_state:
    st.info("Configura los parámetros y pulsa **Ejecutar ambos métodos**.")
    st.stop()

inst, ex, res = st.session_state["out"]
vals = np.array([r["objective"] for r in res])
best_run = res[int(np.argmin(vals))]
gap_best = 100 * (vals.min() - ex["objective"]) / ex["objective"]
gap_mean = 100 * (vals.mean() - ex["objective"]) / ex["objective"]

# ------------------------------------------------------------------ Métricas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Objetivo exacto", f"{ex['objective']:.0f}", ex["status"])
c2.metric("Mejor GA", f"{vals.min():.0f}", f"gap {gap_best:.2f} %", delta_color="inverse")
c3.metric("GA promedio", f"{vals.mean():.2f}", f"gap {gap_mean:.2f} %", delta_color="inverse")
c4.metric("Óptimo hallado (corridas)", f"{int((np.abs(vals - ex['objective']) < 1e-6).sum())}/{len(vals)}")

tab1, tab2, tab3, tab4 = st.tabs(["Comparación", "Convergencia del GA",
                                  "Solución y cobertura", "Distribución del GA"])

with tab1:
    df = pd.DataFrame({
        "Método": ["Exacto (CBC)", "GA (mejor corrida)", "GA (promedio)"],
        "Costo": [ex["objective"], vals.min(), vals.mean()],
        "Tiempo (s)": [ex["time"], best_run["time"], np.mean([r["time"] for r in res])],
        "# subconjuntos": [len(ex["selected"]), len(best_run["selected"]), np.nan],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(df["Método"], df["Costo"], color=["#2a6f97", "#e07a1f", "#f2b366"])
    ax[0].set_title("Costo de la solución"); ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar(df["Método"], df["Tiempo (s)"], color=["#2a6f97", "#e07a1f", "#f2b366"])
    ax[1].set_title("Tiempo de cómputo (s)"); ax[1].tick_params(axis="x", rotation=15)
    plt.tight_layout(); st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for r in res:
        ax.plot(r["hist_best"], color="#e07a1f", alpha=0.25)
    ax.plot(best_run["hist_best"], color="#e07a1f", lw=2, label="Mejor individuo (mejor corrida)")
    ax.plot(best_run["hist_mean"], color="gray", lw=1.5, label="Promedio población (mejor corrida)")
    ax.axhline(ex["objective"], color="#2a6f97", ls="--", label="Óptimo exacto")
    ax.set_xlabel("Generación"); ax.set_ylabel("Costo"); ax.legend()
    st.pyplot(fig)

with tab3:
    sel_ex, sel_ga = set(ex["selected"]), set(best_run["selected"])
    st.write(f"**Exacto** selecciona: {sorted(sel_ex)}")
    st.write(f"**GA (mejor)** selecciona: {sorted(sel_ga)}  | factible: {best_run['feasible']}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, sel, ttl in zip(axs, [sorted(sel_ex), sorted(sel_ga)], ["Exacto", "GA"]):
        a.imshow(inst.A[:, sel] if sel else np.zeros((inst.m, 1)), aspect="auto", cmap="Blues")
        a.set_title(f"{ttl}: columnas seleccionadas"); a.set_xlabel("Subconjunto elegido")
    axs[0].set_ylabel("Elemento")
    plt.tight_layout(); st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(vals, bins=min(15, max(3, len(vals))), color="#e07a1f", alpha=0.8)
    ax.axvline(ex["objective"], color="#2a6f97", ls="--", label="Óptimo exacto")
    ax.set_xlabel("Costo final"); ax.set_ylabel("Corridas"); ax.legend()
    st.pyplot(fig)
    st.caption(f"Media {vals.mean():.2f} · desviación {vals.std():.2f} · "
               f"mín {vals.min():.0f} · máx {vals.max():.0f}")
