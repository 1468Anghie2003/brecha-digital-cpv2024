# =============================================================================
#  VERIFICACIÓN DEL SISTEMA MLOPS
#  Comprueba que estén todos los artefactos y todas las librerías antes de
#  lanzar la aplicación. Ejecutar en la terminal:  python verificar_mlops.py
# =============================================================================
import os
import sys
import json
import importlib.util

DIR = "mlops"
ok_total = True


def linea(t=""):
    print(t)


def check(cond, texto, detalle_ok="", detalle_mal=""):
    global ok_total
    marca = "  [OK]  " if cond else "  [!!]  "
    print(f"{marca}{texto}")
    if cond and detalle_ok:
        print(f"         {detalle_ok}")
    if not cond:
        ok_total = False
        if detalle_mal:
            for l in detalle_mal.split("\n"):
                print(f"         {l}")
    return cond


linea("=" * 75)
linea("  VERIFICACIÓN DEL SISTEMA MLOPS · PROYECTO CPV 2024")
linea("=" * 75)
linea()
linea(f"  Intérprete: {sys.executable}")
linea(f"  Versión   : {sys.version.split()[0]}")
linea(f"  Carpeta   : {os.getcwd()}")
linea()

linea("-" * 75)
linea("  1. LIBRERÍAS")
linea("-" * 75)
BASICAS = ["streamlit", "plotly", "pandas", "numpy", "joblib", "sklearn"]
MODELOS = ["xgboost", "lightgbm", "catboost"]
faltan_basicas = [p for p in BASICAS if importlib.util.find_spec(p) is None]
faltan_modelos = [p for p in MODELOS if importlib.util.find_spec(p) is None]

check(not faltan_basicas, "Librerías base instaladas",
      ", ".join(BASICAS),
      "Faltan: " + ", ".join(faltan_basicas) +
      "\nInstale con:  pip install " + " ".join(faltan_basicas))

if faltan_modelos:
    print("  [!!]  Librerías de modelos faltantes: " + ", ".join(faltan_modelos))
    print("         Los modelos de esas librerías no podrán cargarse.")
    print("         La aplicación los oculta y sigue funcionando.")
    print("         Para habilitarlos:  pip install " + " ".join(faltan_modelos))
else:
    print("  [OK]  Librerías de modelos instaladas: " + ", ".join(MODELOS))

linea()
linea("-" * 75)
linea("  2. ARTEFACTOS DE LA CARPETA mlops/")
linea("-" * 75)
ARTEFACTOS = {
    "registro.json": "métricas y configuración de cada versión",
    "panorama.json": "distribuciones, correlaciones y ganadores",
    "curvas.json": "puntos de las curvas ROC y Precision-Recall",
    "matrices.json": "matrices de confusión",
    "importancias.json": "importancia de variables por modelo",
    "dispersion.json": "puntos de los diagramas de regresión",
    "cruces.json": "cruces entre departamento y variables",
    "ranking.json": "ranking consolidado de la Fase I",
    "mapa_departamentos.csv": "indicadores por departamento",
    "deriva.csv": "medición de deriva geográfica",
    "muestra.csv": "muestra para la vista previa",
}
for arch, desc in ARTEFACTOS.items():
    p = os.path.join(DIR, arch)
    check(os.path.exists(p), f"{arch:<28} {desc}",
          f"{os.path.getsize(p)/1024:.0f} KB" if os.path.exists(p) else "",
          "No existe. Ejecute:  python FASE_J_preparar_mlops.py")

linea()
linea("-" * 75)
linea("  3. CONTENIDO DE panorama.json")
linea("-" * 75)
p = os.path.join(DIR, "panorama.json")
if os.path.exists(p):
    pan = json.load(open(p, encoding="utf-8"))
    check("limpieza" in pan, "Trazabilidad de la depuración (gráfico de cascada)",
          "", "panorama.json es de una versión anterior.\n"
              "Ejecute de nuevo FASE_J_preparar_mlops.py en su versión actual.")
    check("ganadores" in pan, "Modelos ganadores del proyecto",
          ", ".join(f"{k}: {v['modelo']}"
                    for k, v in pan.get("ganadores", {}).items())
          if "ganadores" in pan else "",
          "panorama.json es de una versión anterior.\n"
          "Ejecute de nuevo FASE_J_preparar_mlops.py en su versión actual.")
    check("cv" in pan, "Referencias de validación cruzada")
    check("distribuciones" in pan,
          f"Distribuciones de {len(pan.get('distribuciones', {}))} variables")
else:
    print("  [!!]  No se puede analizar: falta el archivo.")

linea()
linea("-" * 75)
linea("  4. MODELOS SERIALIZADOS")
linea("-" * 75)
pm = os.path.join(DIR, "modelos")
if os.path.isdir(pm):
    arch = sorted(os.listdir(pm))
    print(f"  [OK]  {len(arch)} archivos en {pm}")
    tot = sum(os.path.getsize(os.path.join(pm, a)) for a in arch)
    print(f"         Tamaño total: {tot/1024**2:.1f} MB")
    malos = []
    for a in arch:
        for nom, lib in [("CatBoost", "catboost"), ("XGBoost", "xgboost"),
                         ("LightGBM", "lightgbm")]:
            if nom in a and lib in faltan_modelos:
                malos.append(a)
    if malos:
        print(f"  [!!]  {len(malos)} modelos no se podrán cargar por falta de "
              f"su librería:")
        for a in malos[:6]:
            print(f"         {a}")
        if len(malos) > 6:
            print(f"         ... y {len(malos)-6} más")
else:
    check(False, "Carpeta mlops/modelos",
          "", "No existe. Ejecute:  python FASE_J_preparar_mlops.py")

linea()
linea("-" * 75)
linea("  5. CARPETA entregables/  (necesaria para el ranking)")
linea("-" * 75)
for a in ["I_todas_metricas_clasificacion_binaria.csv",
          "I_todas_metricas_regresion.csv",
          "I_todas_metricas_multiclase.csv"]:
    r = os.path.join("entregables", a)
    check(os.path.exists(r), a, "",
          "No existe. Lo genera la Fase I del proyecto.\n"
          "Sin estos archivos, la sección Ranking del proyecto queda vacía.")

linea()
linea("-" * 75)
linea("  6. MLFLOW")
linea("-" * 75)
if os.path.exists("mlflow.db"):
    print("  [OK]  mlflow.db encontrado")
    print("         Abrir con:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
else:
    print("  [--]  mlflow.db no encontrado")
    print("         El registro en MLflow es opcional. El versionado del")
    print("         proyecto se implementa con registro.json y no depende")
    print("         de esta herramienta.")

linea()
linea("=" * 75)
if ok_total:
    linea("  RESULTADO: todo correcto. Lance la aplicación con:")
    linea("             streamlit run app_mlops.py")
else:
    linea("  RESULTADO: hay elementos faltantes. Revise las marcas [!!]")
    linea("             de arriba y siga las instrucciones indicadas.")
linea("=" * 75)
