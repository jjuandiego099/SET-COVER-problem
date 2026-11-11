"""Interfaz visual (Streamlit). Ejecutar con:  streamlit run app.py"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from setcover import generate_instance, solve_exact, run_ga, GAParams
import storage

BLUE, SKY, NAVY, ICE = "#1d4ed8", "#38bdf8", "#0b1e3f", "#e3ecfc"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9fb3d9",
                     "axes.labelcolor": NAVY, "text.color": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "figure.facecolor": "none", "axes.facecolor": "none", "axes.titleweight": "bold",
                     "font.size": 9, "axes.grid": True, "grid.color": "#d3e0f7", "grid.linewidth": 0.6})

st.set_page_config(page_title="Set Cover Lab", page_icon="🔷", layout="wide")
st.markdown(f"""
<style>
.stApp {{background: #f4f8ff; color: {NAVY};}}
.stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp p {{color: {NAVY};}}
[data-testid="stSidebar"] {{background: {ICE};}}
div[data-baseweb="slider"] [role="slider"] {{background-color: {BLUE} !important;}}
.stTabs [aria-selected="true"] {{color: {BLUE};}}
[data-baseweb="tab-highlight"] {{background-color: {BLUE} !important;}}
.hero {{background: linear-gradient(115deg, {NAVY} 0%, {BLUE} 62%, {SKY} 100%); color: white;
        padding: 1.5rem 2rem; border-radius: 20px; margin-bottom: 1.1rem;
        box-shadow: 0 8px 24px rgba(29,78,216,.25);}}
.hero h1 {{color: white !important; margin: 0; font-size: 2rem; letter-spacing: -.5px;}}
.hero p {{color: #d6e6ff !important; margin: .3rem 0 0 0;}}
[data-testid="stMetric"] {{background: white; border: 1px solid #cddcf7; border-left: 6px solid {BLUE};
        border-radius: 14px; padding: .8rem 1rem; box-shadow: 0 3px 10px rgba(29,78,216,.08);}}
.stButton > button[kind="primary"] {{background: linear-gradient(90deg, {BLUE}, #2563eb); border: 0;
        border-radius: 12px; font-weight: 700; padding: .55rem 1.2rem;}}
.stTabs [data-baseweb="tab"] {{font-weight: 600; border-radius: 10px 10px 0 0;}}
.verdict {{background: white; border: 1px solid #cddcf7; border-left: 8px solid {BLUE}; border-radius: 14px;
        padding: 1rem 1.3rem; margin: .8rem 0; box-shadow: 0 3px 10px rgba(29,78,216,.08);}}
.verdict h3 {{margin: 0 0 .4rem 0; color: {BLUE};}}
.tag {{display:inline-block; background:{ICE}; color:{BLUE}; border-radius:20px; padding:.1rem .7rem;
        font-size:.8rem; font-weight:600; margin-right:.4rem;}}
</style>
<div class="hero"><h1>🔷 Set Cover Lab</h1>
<p>Solver exacto (CBC) frente a Algoritmo Genético · compara, guarda y reabre tus casos</p></div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ Sidebar
mode = st.sidebar.radio("Modo", ["Nuevo caso", "Casos guardados"], horizontal=True)

if mode == "Nuevo caso":
    st.sidebar.subheader("Instancia")
    m = st.sidebar.slider("Elementos del universo (m)", 10, 300, 60, 10)
    n = st.sidebar.slider("Subconjuntos (n)", 10, 300, 40, 10)
    dens = st.sidebar.slider("Densidad de la matriz", 0.02, 0.50, 0.10, 0.01)
    cmax = st.sidebar.slider("Costo máximo", 1, 50, 10)
    seed = st.sidebar.number_input("Semilla de la instancia", 0, 10**6, 1)
    st.sidebar.subheader("Solver exacto")
    tlim = st.sidebar.number_input("Límite de tiempo (s)", 1, 3600, 60)
    st.sidebar.subheader("Algoritmo Genético")
    pop = st.sidebar.slider("Tamaño de población", 10, 300, 60, 10)
    gens = st.sidebar.slider("Generaciones máximas", 10, 1000, 200, 10)
    pc = st.sidebar.slider("Prob. de cruce", 0.0, 1.0, 0.9, 0.05)
    pm = st.sidebar.slider("Prob. de mutación (por gen)", 0.0, 0.2, 0.02, 0.005)
    k = st.sidebar.slider("Tamaño del torneo", 2, 10, 3)
    elite = st.sidebar.slider("Élite", 0, 10, 2)
    pat = st.sidebar.slider("Paciencia (parada)", 5, 300, 50, 5)
    runs = st.sidebar.slider("Corridas independientes del GA", 1, 30, 5)
    gseed = st.sidebar.number_input("Semilla base del GA", 0, 10**6, 0)
    if st.sidebar.button("▶ Ejecutar ambos métodos", type="primary"):
        inst = generate_instance(m, n, dens, (1, cmax), int(seed))
        with st.spinner("Resolviendo con el solver exacto..."):
            ex = solve_exact(inst, time_limit=int(tlim))
        with st.spinner("Ejecutando el Algoritmo Genético..."):
            res = [run_ga(inst, GAParams(pop, gens, pc, pm, k, elite, pat, 0.2, int(gseed) + r))
                   for r in range(runs)]
        params = dict(m=m, n=n, densidad=dens, costo_max=cmax, semilla=int(seed), limite_s=int(tlim),
                      poblacion=pop, generaciones=gens, p_cruce=pc, p_mutacion=pm, torneo=k,
                      elite=elite, paciencia=pat, corridas=runs, semilla_ga=int(gseed))
        st.session_state["out"] = (inst, ex, res, {"name": "", "params": params, "saved": None})
else:
    cases = storage.list_cases()
    if not cases:
        st.sidebar.info("Aún no hay casos guardados. Crea uno en «Nuevo caso» y pulsa Guardar.")
    else:
        names = {key: name for key, name in cases}
        sel = st.sidebar.selectbox("Caso", list(names), format_func=lambda x: names[x])
        c1, c2 = st.sidebar.columns(2)
        if c1.button("📂 Abrir", type="primary"):
            st.session_state["out"] = storage.load_case(sel)
        if c2.button("🗑 Eliminar"):
            storage.delete_case(sel)
            st.session_state.pop("out", None)
            st.rerun()

if "out" not in st.session_state:
    st.info("Configura los parámetros y pulsa **Ejecutar ambos métodos**, o abre un caso guardado.")
    with st.expander("📚 Historial de casos guardados", expanded=mode == "Casos guardados"):
        df = storage.summary_table()
        st.dataframe(df, hide_index=True) if len(df) else st.caption("Sin casos guardados todavía.")
    st.stop()

inst, ex, res, meta = st.session_state["out"]
vals = np.array([r["objective"] for r in res])
best_run = res[int(np.argmin(vals))]
gap_best = 100 * (vals.min() - ex["objective"]) / ex["objective"]
gap_mean = 100 * (vals.mean() - ex["objective"]) / ex["objective"]
t_ex, t_ga = ex["time"], float(np.mean([r["time"] for r in res]))
n_opt = int((np.abs(vals - ex["objective"]) < 1e-6).sum())
proven = ex["status"] == "Optimal"

# ------------------------------------------------------------------ Guardar / info del caso
if meta.get("saved"):
    st.markdown(f"<span class='tag'>Caso: {meta['name']}</span><span class='tag'>Guardado {meta['saved']}</span>"
                f"<span class='tag'>{inst.m}×{inst.n}</span>", unsafe_allow_html=True)
    if meta.get("description"):
        st.caption(meta["description"])
    with st.expander("Parámetros usados"):
        st.json(meta["params"])
else:
    with st.expander("💾 Guardar este caso", expanded=False):
        cn = st.text_input("Nombre del caso", placeholder="p. ej. Ambulancias Bucaramanga")
        cd = st.text_area("Descripción (opcional)", height=70)
        if st.button("Guardar caso"):
            if not cn.strip():
                st.warning("Escribe un nombre para el caso.")
            else:
                key = storage.save_case(cn, cd, inst, ex, res, meta["params"])
                st.session_state["out"] = storage.load_case(key)
                st.toast("Caso guardado ✅")
                st.rerun()

# ------------------------------------------------------------------ Métricas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Objetivo exacto", f"{ex['objective']:.0f}", ex["status"], delta_color="off")
c2.metric("Mejor GA", f"{vals.min():.0f}", f"gap {gap_best:.2f} %", delta_color="inverse")
c3.metric("GA promedio", f"{vals.mean():.2f}", f"gap {gap_mean:.2f} %", delta_color="inverse")
c4.metric("Óptimo hallado", f"{n_opt}/{len(vals)}", "corridas del GA", delta_color="off")

# ------------------------------------------------------------------ Veredicto
if vals.min() < ex["objective"] - 1e-6:
    calidad = "GA"
    txt_cal = (f"El GA encontró una solución más barata ({vals.min():.0f}) que el solver ({ex['objective']:.0f}); "
               f"solo ocurre si el solver no terminó (estado: {ex['status']}).")
elif vals.min() > ex["objective"] + 1e-6:
    calidad = "Exacto"
    txt_cal = f"El exacto es mejor: {ex['objective']:.0f} frente a {vals.min():.0f} del mejor GA (gap {gap_best:.2f} %)."
else:
    calidad = "Empate"
    txt_cal = (f"Empate: el GA igualó el costo ({ex['objective']:.0f}) en {n_opt} de {len(vals)} corridas "
               f"(promedio {vals.mean():.2f}, gap {gap_mean:.2f} %).")
rapido = "Exacto" if t_ex < t_ga else "GA"
txt_tim = (f"El {'exacto' if rapido == 'Exacto' else 'GA'} fue más rápido "
           f"({t_ex:.3f} s exacto vs {t_ga:.3f} s GA promedio).")
if calidad == "Exacto":
    ganador = "Mejor método: Exacto" + (" (también más rápido)" if rapido == "Exacto" else " en calidad")
elif calidad == "GA":
    ganador = "Mejor método: Algoritmo Genético (menor costo)"
elif rapido == "Exacto":
    ganador = "Mejor método: Exacto (misma calidad, más rápido" + (" y óptimo garantizado)" if proven else ")")
else:
    ganador = "Empate en calidad · el GA fue más rápido" + (" (el exacto garantiza optimalidad)" if proven else "")
st.markdown(f"<div class='verdict'><h3>🏆 {ganador}</h3>• <b>Calidad:</b> {txt_cal}<br>"
            f"• <b>Tiempo:</b> {txt_tim}<br><small>Vale solo para esta instancia y estos parámetros; con "
            f"instancias mayores o límites de tiempo ajustados puede cambiar. El tiempo del exacto incluye "
            f"el arranque de CBC.</small></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Comparación", "Convergencia", "Solución y cobertura",
                                        "Distribución del GA", "Historial de casos"])
with tab1:
    labels = ["Exacto (CBC)", "GA mejor", "GA promedio"]
    costs = [ex["objective"], vals.min(), vals.mean()]
    times = [t_ex, best_run["time"], t_ga]
    st.dataframe({"Método": labels, "Costo": costs, "Tiempo (s)": [round(t, 4) for t in times],
                  "# subconjuntos": [len(ex["selected"]), len(best_run["selected"]), None]}, hide_index=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, v, ttl in zip(ax, [costs, times], ["Costo de la solución", "Tiempo de cómputo (s)"]):
        a.bar(labels, v, color=[NAVY, BLUE, SKY]); a.set_title(ttl); a.tick_params(axis="x", rotation=12)
    plt.tight_layout(); st.pyplot(fig)
with tab2:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for r in res:
        ax.plot(r["hist_best"], color=SKY, alpha=0.35)
    ax.plot(best_run["hist_best"], color=BLUE, lw=2.4, label="Mejor individuo (mejor corrida)")
    ax.plot(best_run["hist_mean"], color="#7f93b8", lw=1.5, label="Promedio de la población")
    ax.axhline(ex["objective"], color=NAVY, ls="--", label="Óptimo exacto")
    ax.set_xlabel("Generación"); ax.set_ylabel("Costo"); ax.legend(frameon=False)
    st.pyplot(fig)
with tab3:
    st.write(f"**Exacto** selecciona: {sorted(ex['selected'])}")
    st.write(f"**GA (mejor)** selecciona: {sorted(best_run['selected'])} · factible: {best_run['feasible']}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, sel, ttl in zip(axs, [sorted(ex["selected"]), sorted(best_run["selected"])], ["Exacto", "GA"]):
        a.imshow(inst.A[:, sel] if sel else np.zeros((inst.m, 1)), aspect="auto", cmap="Blues")
        a.set_title(f"{ttl}: columnas elegidas"); a.set_xlabel("Subconjunto elegido"); a.grid(False)
    axs[0].set_ylabel("Elemento"); plt.tight_layout(); st.pyplot(fig)
with tab4:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(vals, bins=min(15, max(3, len(vals))), color=BLUE, alpha=0.85)
    ax.axvline(ex["objective"], color=NAVY, ls="--", label="Óptimo exacto")
    ax.set_xlabel("Costo final"); ax.set_ylabel("Corridas"); ax.legend(frameon=False)
    st.pyplot(fig)
    st.caption(f"Media {vals.mean():.2f} · desviación {vals.std():.2f} · mín {vals.min():.0f} · máx {vals.max():.0f}")
with tab5:
    df = storage.summary_table()
    st.dataframe(df, hide_index=True) if len(df) else st.caption("Sin casos guardados todavía.")