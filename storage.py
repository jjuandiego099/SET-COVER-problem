"""Persistencia de casos en disco (carpeta cases/, un JSON por caso)."""
import json, re, time
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
