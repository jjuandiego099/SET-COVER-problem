# Proyecto Set Cover: Exacto (PuLP/CBC) vs Algoritmo Genético

## Instalación y ejecución
    pip install -r requirements.txt
    streamlit run app.py          # interfaz visual
    python experiments.py         # genera resultados.csv (15 instancias x 10 corridas GA)
    python make_table.py          # genera tabla_resultados.tex para el informe

## Archivos
- setcover.py      instancias, modelo exacto, Algoritmo Genético
- app.py           interfaz Streamlit (parámetros, comparación, convergencia, cobertura, distribución)
- experiments.py   experimentos reproducibles (semillas fijas)
- informe.tex + referencias.bib   plantilla para Overleaf (subir también tabla_resultados.tex)
