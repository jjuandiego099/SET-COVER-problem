"""Interfaz visual (Streamlit). Ejecutar con:  streamlit run app.py"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from setcover import generate_instance, solve_exact, run_ga, GAParams
import json, re, time, types
from pathlib import Path
import numpy as np
import pandas as pd
from setcover import Instance

DIR = Path(__file__).parent / "cases"


def _slug(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_") or "caso"


def _plain(d):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}


def save_case(name, desc, inst, ex, res, params):
    DIR.mkdir(exist_ok=True)
    data = {"name": name.strip(), "description": desc, "saved": time.strftime("%Y-%m-%d %H:%M"),
            "params": params, "A": inst.A.tolist(), "c": inst.c.tolist(),
            "exact": _plain(ex), "ga": [_plain(r) for r in res]}
    (DIR / f"{_slug(name)}.json").write_text(json.dumps(data), encoding="utf8")
    return _slug(name)


def list_cases():
    """Lista de (clave, nombre) ordenada por nombre."""
    out = []
    for f in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        try:
            out.append((f.stem, json.loads(f.read_text(encoding="utf8"))["name"]))
        except Exception:
            pass
    return sorted(out, key=lambda t: t[1].lower())


def load_case(key):
    d = json.loads((DIR / f"{key}.json").read_text(encoding="utf8"))
    inst = Instance(A=np.array(d["A"], dtype=np.int8), c=np.array(d["c"], dtype=float))
    ex = d["exact"]
    ex["solution"] = np.array(ex["solution"])
    meta = {"name": d["name"], "description": d.get("description", ""),
            "saved": d["saved"], "params": d["params"]}
    return inst, ex, d["ga"], meta


def delete_case(key):
    (DIR / f"{key}.json").unlink(missing_ok=True)


def summary_table():
    rows = []
    for key, _ in list_cases():
        inst, ex, res, meta = load_case(key)
        best = min(r["objective"] for r in res)
        mean = float(np.mean([r["objective"] for r in res]))
        rows.append({"Caso": meta["name"], "m": inst.m, "n": inst.n,
                     "Óptimo exacto": ex["objective"], "t exacto (s)": round(ex["time"], 3),
                     "GA mejor": best, "GA prom.": round(mean, 2),
                     "Gap prom. (%)": round(100 * (mean - ex["objective"]) / ex["objective"], 2),
                     "t GA (s)": round(float(np.mean([r["time"] for r in res])), 3),
                     "Guardado": meta["saved"]})
    return pd.DataFrame(rows)


def import_case(raw_bytes):
    """Importa un caso desde el contenido de un .json exportado por la app."""
    d = json.loads(raw_bytes)
    for k in ("name", "A", "c", "exact", "ga", "params", "saved"):
        if k not in d:
            raise ValueError(f"Falta el campo {k}")
    DIR.mkdir(exist_ok=True)
    (DIR / f"{_slug(d['name'])}.json").write_text(json.dumps(d), encoding="utf8")
    return _slug(d["name"])


storage = types.SimpleNamespace(save_case=save_case, list_cases=list_cases, load_case=load_case,
                                delete_case=delete_case, summary_table=summary_table,
                                import_case=import_case, slug=_slug, DIR=DIR)

BLUE, SKY, NAVY, ICE = "#1d4ed8", "#38bdf8", "#0b1e3f", "#e3ecfc"

try:  # color primario azul (en vez del rojo por defecto) sin archivo de configuración
    st._config.set_option("theme.primaryColor", BLUE)
except Exception:
    pass
st.set_page_config(page_title="Set Cover: Exacto vs GA", layout="wide")
st.markdown(f"""
<style>
[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {{
    background-color: {BLUE}; border-color: {BLUE}; color: white;}}
[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover {{
    background-color: #1e40af; border-color: #1e40af; color: white;}}
div[data-baseweb="slider"] [role="slider"] {{background-color: {BLUE} !important;}}
[data-baseweb="tab-highlight"] {{background-color: {BLUE} !important;}}
.stTabs [aria-selected="true"] {{color: {BLUE};}}
</style>
""", unsafe_allow_html=True)
st.title("Set Cover: método exacto vs Algoritmo Genético")

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
        st.sidebar.download_button("⬇ Descargar caso (.json)", data=(storage.DIR / f"{sel}.json").read_bytes(),
                                   file_name=f"{sel}.json", mime="application/json")
        if c2.button("🗑 Eliminar"):
            storage.delete_case(sel)
            st.session_state.pop("out", None)
            st.rerun()

if mode == "Casos guardados":
    up = st.sidebar.file_uploader("Importar caso (.json)", type="json")
    if up is not None and st.session_state.get("_imp") != (up.name, up.size):
        try:
            storage.import_case(up.getvalue())
            st.session_state["_imp"] = (up.name, up.size)
            st.toast("Caso importado ✅")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"No se pudo importar: {e}")

if "out" not in st.session_state:
    st.info("Configura los parámetros y pulsa **Ejecutar ambos métodos**, o abre un caso guardado.")
    with st.expander("📚 Historial de casos guardados", expanded=mode == "Casos guardados"):
        df = storage.summary_table()
        if len(df):
            st.dataframe(df, hide_index=True)
        else:
            st.caption("Sin casos guardados todavía.")
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
    st.caption(f"Caso: {meta['name']} · guardado {meta['saved']} · {inst.m}×{inst.n}")
    if meta.get("description"):
        st.caption(meta["description"])
    st.download_button("⬇ Descargar este caso (.json)", data=(storage.DIR / f"{storage.slug(meta['name'])}.json").read_bytes(),
                       file_name=f"{storage.slug(meta['name'])}.json", mime="application/json")
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
st.info(f"🏆 **{ganador}**")
st.markdown(f"- **Calidad:** {txt_cal}\n- **Tiempo:** {txt_tim}")
st.caption("Vale solo para esta instancia y estos parámetros; con instancias mayores o límites de tiempo "
           "ajustados puede cambiar. El tiempo del exacto incluye el arranque de CBC.")

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
    if len(df):
        st.dataframe(df, hide_index=True)
    else:
        st.caption("Sin casos guardados todavía.")