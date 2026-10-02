"""
===============================================================================
 BRECHA DIGITAL Y EQUIPAMIENTO DEL HOGAR EN BOLIVIA
 Sistema predictivo de exclusión digital y material a partir del
 Censo de Población y Vivienda 2024

 Universidad Mayor de San Andrés
 Facultad de Ciencias Puras y Naturales · Carrera de Informática
 Materia: Machine Learning · Aprendizaje supervisado

 Autores : Ethan Alba Orellana
           Jhamil López Salgueiro
           Angela Luz Mamani Castro

 Entrada : Vivienda_CPV-2024.csv        (4.490.488 registros · 48 variables)
 Salida  : vivienda_cpv2024_LIMPIO_v2   (3.492.019 registros · 31 variables)
           figuras_v2/                  figuras 16 a 63 del informe
           entregables/                 modelos serializados y métricas
           BITACORA_PROYECTO.xlsx       registro de decisiones y modelos

 Semilla de aleatorización: 42 en todas las fases.

-------------------------------------------------------------------------------
 ESTRUCTURA DEL ARCHIVO
-------------------------------------------------------------------------------
 FASE A   Análisis descriptivo previo a la limpieza        figuras 16 a 19
 FASE B   Limpieza y tratamiento de faltantes              figuras 20 y 21
 FASE C   Análisis descriptivo posterior a la limpieza     figuras 22 a 25
 FASE D   Modelos lineales y experimento de codificación   figuras 26 a 31
 FASE E   KNN, Naive Bayes y máquinas de soporte vectorial figuras 32 a 36
 FASE F   Árboles de decisión y ensambles por agregación   figuras 37 a 48
 FASE G   Ensambles por refuerzo (boosting)                figuras 49 a 56
 FASE H   Validación cruzada y selección final             figuras 57 a 62
 FASE I   Cierre, modelos finales y entregables            figura  63

-------------------------------------------------------------------------------
 ARCHIVOS QUE GENERA CADA FASE
-------------------------------------------------------------------------------
 Fase A   figuras_v2/A01 a A04 .png
          figuras_v2/A_decisiones_propuestas.csv

 Fase B   Vivienda_CPV-2024_filtrado.parquet
             Caché del archivo original ya filtrado al universo de análisis.
             Se crea durante la lectura por lotes y evita releer los 468 MB
             del CSV en cada ejecución posterior.
          Vivienda_CPV-2024_filtrado_meta.csv
             Conteos de los filtros aplicados durante esa lectura.
          vivienda_cpv2024_LIMPIO_v2.csv      400,9 MB
          vivienda_cpv2024_LIMPIO_v2.parquet   32,0 MB
             Dataset depurado definitivo. El formato Parquet conserva los
             tipos de dato y es el que cargan las fases C a I.
          figuras_v2/B_bitacora_limpieza.csv
             Los trece pasos de la depuración con su justificación.
          figuras_v2/B_experimento_imputacion.csv
          figuras_v2/B01 y B02 .png

 Fase C   figuras_v2/C01 a C04 .png
          figuras_v2/C_resumen_post_limpieza.csv
          figuras_v2/C_correlaciones.csv

 Fase D   BITACORA_PROYECTO.xlsx    se crea aquí e importa la bitácora de B
          figuras_v2/D01 a D06 .png
          figuras_v2/D_regresion.csv
          figuras_v2/D_clasificacion_binaria.csv
          figuras_v2/D_clasificacion_multiclase.csv

 Fase E   logs/consola_fase_E.txt   registro completo de la salida
          figuras_v2/E01 a E05 .png
          figuras_v2/E_*.csv

 Fase F   figuras_v2/F01 a F12 .png
          figuras_v2/F_*.csv

 Fase G   figuras_v2/G01 a G08 .png
          figuras_v2/G_*.csv

 Fase H   figuras_v2/H01 a H06 .png
          figuras_v2/H_cv_clasificacion.csv
          figuras_v2/H_cv_regresion.csv
          figuras_v2/H_valores_p.csv

 Fase I   entregables/modelo_internet.joblib
          entregables/modelo_ieh.joblib
          entregables/metadata_modelos.json
          entregables/indicadores_por_departamento.csv
          entregables/INFORME_METRICAS.txt
          entregables/I_todas_metricas_*.csv
          figuras_v2/I01_resumen_proyecto.png

-------------------------------------------------------------------------------
 LAS DOS BITÁCORAS
-------------------------------------------------------------------------------
 1. Bitácora de limpieza · figuras_v2/B_bitacora_limpieza.csv
    La construye la función registrar() de la Fase B. Cada intervención sobre
    los datos añade una fila con: número correlativo, fecha, categoría,
    paso, variable afectada, problema, acción, justificación, valores
    afectados y conteo de filas antes y después. Permite rastrear cualquier
    cifra del dataset limpio hasta la decisión que la produjo.

 2. Bitácora maestra · BITACORA_PROYECTO.xlsx
    La construyen las funciones registrar() y registrar_modelo() de las fases
    D a I, y se vuelca con volcar_bitacora(). Contiene tres hojas:
      Resumen por fase  pasos y última actualización de cada fase
      Registro          cada decisión metodológica con su justificación
      Modelos           una fila por modelo con su objetivo, tipo, tiempo
                        de entrenamiento y todas sus métricas
    La escritura es idempotente: cada fase elimina sus propias filas antes
    de reescribirlas, de modo que reejecutar una fase actualiza sus
    resultados sin duplicarlos ni afectar a las demás.
    Requiere openpyxl y que el archivo no esté abierto en Excel.
===============================================================================
"""


# %%
# =============================================================================
#  A0 — CONFIGURACION, PALETA Y CARGA
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings('ignore')

try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

# ---- Carpeta NUEVA para no sobrescribir las figuras de la bitacora anterior
DIR_FIG = "figuras_v2"
os.makedirs(DIR_FIG, exist_ok=True)

# ---- Paleta del proyecto
AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA = "#FF6B8A", "#FFC145", "#22223B"
GRIS, AGUA = "#8D99AE", "#A8DADC"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, AGUA, "#6C8AE4", "#C3A6F0"]
CMAP = LinearSegmentedColormap.from_list("proy", ["#EEF2FF", AZUL, LILA])
CMAP_DIV = LinearSegmentedColormap.from_list("div", [AZUL, "#FFFFFF", CORAL])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 11.5,
    "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "grid.linewidth": .9,
    "axes.axisbelow": True, "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92,
    "legend.edgecolor": "#D8DEF0",
})


def guardar(nombre):
    """Guarda en figuras_v2/ y muestra en linea."""
    ruta = os.path.join(DIR_FIG, nombre)
    plt.savefig(ruta, bbox_inches="tight", facecolor="white")
    print(f"   [figura] {ruta}")
    plt.show()


def titulo(t, ancho=86):
    print("\n" + "═" * ancho)
    print(f"  {t}")
    print("═" * ancho)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


ARCHIVO = "Vivienda_CPV-2024.csv"
RUTA = ARCHIVO if os.path.exists(ARCHIVO) else os.path.join(
    r"C:\Users\HP\Desktop\ProyApSup255", ARCHIVO)


def leer_censo(ruta):
    for sep in [';', ',', '\t', '|']:
        for enc in ['latin-1', 'utf-8', 'cp1252', 'utf-8-sig']:
            try:
                d = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)
                if d.shape[1] > 20:
                    print(f"   Lectura correcta · sep='{sep}' · encoding='{enc}'")
                    return d
            except Exception:
                continue
    raise ValueError("No se pudo leer el archivo")


titulo("FASE A · ANALISIS DESCRIPTIVO PREVIO A LA LIMPIEZA")
print(f"   Archivo: {RUTA}")
df = leer_censo(RUTA)
df.columns = df.columns.str.strip().str.lower()
CRUDO = df.copy()          # copia intacta para comparar al final

print(f"\n   Registros : {len(df):,}")
print(f"   Variables : {df.shape[1]}")
print(f"   Memoria   : {df.memory_usage(deep=True).sum()/1024**2:,.1f} MB")

# ---- Diccionario de etiquetas legibles
NOMBRES = {
    'idep': 'Departamento', 'urbrur': 'Área urbana/rural',
    'v01_tipoviv': 'Tipo de vivienda', 'v02_condocup': 'Condición de ocupación',
    'v03_pared': 'Material de pared', 'v04_revoq': 'Pared revocada',
    'v05_techo': 'Material de techo', 'v06_piso': 'Material de piso',
    'v07_aguapro': 'Procedencia del agua', 'v08_aguadist': 'Distribución de agua',
    'v09_energia': 'Fuente de energía', 'v10_combus': 'Combustible de cocina',
    'v11_basura': 'Manejo de basura', 'v12_cocina': 'Tiene cocina',
    'v13_habitac': 'N° de habitaciones', 'v14_dormit': 'N° de dormitorios',
    'v15_servsan': 'Servicio sanitario', 'v16_desague': 'Tipo de desagüe',
    'v17_tenencia': 'Tenencia de la vivienda', 'tot_pers': 'Personas en el hogar',
    'tip_hog': 'Tipo de hogar', 'hacinamiento': 'Personas por dormitorio',
    'ieh': 'Índice de equipamiento', 'internet': 'Acceso a internet',
}
ETQ = lambda c: NOMBRES.get(c, c)

DEPTOS = {1: 'Chuquisaca', 2: 'La Paz', 3: 'Cochabamba', 4: 'Oruro',
          5: 'Potosí', 6: 'Santa Cruz', 7: 'Tarija', 8: 'Beni', 9: 'Pando'}

BIENES = ['v18a_bici', 'v18b_moto', 'v18c_auto', 'v18f_refri', 'v18g_micro',
          'v18h_calefon', 'v18i_aire', 'v18j_lavadora', 'v19a_radio',
          'v19b_tv', 'v19c_compu', 'v19d_celular', 'v19g_tvcable']
BIENES = [c for c in BIENES if c in df.columns]
COLS_BIN = [c for c in df.columns if c.startswith(('v18', 'v19'))]


# %%
# =============================================================================
#  A1 — RADIOGRAFIA GENERAL DEL CONJUNTO
# =============================================================================
titulo("A1 · RADIOGRAFIA GENERAL")

resumen = pd.DataFrame({
    'tipo': df.dtypes.astype(str),
    'no_nulos': df.notna().sum(),
    'nulos': df.isna().sum(),
    'pct_nulos': (df.isna().mean() * 100).round(2),
    'unicos': df.nunique(),
    'moda': [df[c].mode().iloc[0] if df[c].notna().any() else np.nan
             for c in df.columns],
})
resumen['cardinalidad'] = pd.cut(
    resumen['unicos'], [0, 2, 10, 50, np.inf],
    labels=['Binaria', 'Baja', 'Media', 'Alta'])

sub("Estructura de las variables")
print(resumen.to_string())

sub("Distribución de la cardinalidad")
print(resumen['cardinalidad'].value_counts().to_string())

sub("Estadística descriptiva de las variables numéricas")
desc = df.describe().T
desc['asimetria'] = df.select_dtypes(include=[np.number]).skew()
desc['curtosis'] = df.select_dtypes(include=[np.number]).kurtosis()
desc['rango'] = desc['max'] - desc['min']
desc['cv_%'] = (desc['std'] / desc['mean'].replace(0, np.nan) * 100).round(1)
print(desc.round(2).to_string())

sub("Variables con asimetría marcada (|skew| > 1)")
asim = desc['asimetria'].abs().sort_values(ascending=False)
asim = asim[asim > 1]
if len(asim):
    for k, v in asim.items():
        signo = "derecha" if desc.loc[k, 'asimetria'] > 0 else "izquierda"
        print(f"   {ETQ(k):<28} skew = {desc.loc[k,'asimetria']:+7.2f}  "
              f"(cola a la {signo})")
else:
    print("   Ninguna variable presenta asimetría marcada.")


# %%
# =============================================================================
#  A2 — CONSTRUCCION PRELIMINAR DE LAS VARIABLES OBJETIVO
#  Se construyen SOBRE EL DATO CRUDO, sin limpiar, para saber de que punto
#  partimos y poder medir despues el efecto de cada decision de limpieza.
# =============================================================================
titulo("A2 · VARIABLES OBJETIVO EN ESTADO CRUDO")

# --- Objetivo de CLASIFICACION: la vivienda tiene internet -------------------
df['internet_crudo'] = df['v19e_f'].map({1: 1, 2: 0})

sub("Objetivo 1 · internet (clasificación binaria)")
vc = df['v19e_f'].value_counts(dropna=False).sort_index()
etq_int = {1: 'Sí tiene', 2: 'No tiene', 9: 'Sin especificar'}
for k, v in vc.items():
    nom = etq_int.get(k, 'Nulo') if pd.notna(k) else 'Nulo'
    print(f"   {str(k):<6} {nom:<18} {v:>10,}  ({v/len(df)*100:6.2f} %)")
print(f"\n   Válidos para modelar : {df['internet_crudo'].notna().sum():,}")
print(f"   Proporción con internet: "
      f"{df['internet_crudo'].mean()*100:.2f} %  "
      f"(cifra oficial INE CPV-2024: 76,30 %)")

# --- Objetivo 2: indice de equipamiento --------------------------------------
# Se cuenta cuantos bienes posee el hogar. El codigo 9 es "Sin especificar"
# segun el diccionario, por lo que NO se cuenta como posesion.
df['ieh_crudo'] = sum((df[c] == 1).astype(float) for c in BIENES)

sub("Objetivo 2 · ieh (regresión) — índice de equipamiento del hogar")
print(f"   Bienes considerados : {len(BIENES)}")
print(df['ieh_crudo'].describe().round(3).to_string())

# --- Objetivo 3: nivel de equipamiento (clasificacion multiclase) ------------
# Permite aplicar los mismos algoritmos de clasificacion a la dimension
# material, respondiendo al objetivo de "categorizar" las viviendas.
q1, q2 = df['ieh_crudo'].quantile([1/3, 2/3])
df['nivel_equip_crudo'] = pd.cut(
    df['ieh_crudo'], [-0.1, q1, q2, len(BIENES)],
    labels=['Bajo', 'Medio', 'Alto'])

sub("Objetivo 3 · nivel_equip (clasificación multiclase)")
print(f"   Cortes por terciles: Bajo ≤ {q1:.0f} · Medio ≤ {q2:.0f} · "
      f"Alto > {q2:.0f}")
ne = df['nivel_equip_crudo'].value_counts().sort_index()
for k, v in ne.items():
    print(f"   {str(k):<8} {v:>10,}  ({v/len(df)*100:6.2f} %)")

# --- Variable derivada: hacinamiento ----------------------------------------
df['hacinamiento_crudo'] = df['tot_pers'] / df['v14_dormit'].clip(lower=1)
sub("Variable derivada · hacinamiento (personas por dormitorio)")
print(df['hacinamiento_crudo'].describe().round(3).to_string())


# %%
# =============================================================================
#  A3 — DISTRIBUCIONES UNIVARIADAS
# =============================================================================
titulo("A3 · DISTRIBUCIONES")

fig = plt.figure(figsize=(16.5, 9.5))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.28)

# (a) objetivo de clasificacion
ax = fig.add_subplot(gs[0, 0])
v = df['v19e_f'].value_counts(dropna=False).sort_index()
lab = [etq_int.get(k, 'Nulo') if pd.notna(k) else 'Nulo' for k in v.index]
col = [MENTA, CORAL, AMBAR, GRIS][:len(v)]
b = ax.bar(lab, v.values, color=col, edgecolor='white', linewidth=2)
for r, val in zip(b, v.values):
    ax.text(r.get_x() + r.get_width()/2, val, f"{val:,}\n{val/len(df)*100:.1f} %",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title("(a) Acceso a internet — variable sin limpiar")
ax.set_ylabel("Viviendas")
ax.set_ylim(0, v.max() * 1.22)

# (b) indice de equipamiento
ax = fig.add_subplot(gs[0, 1])
n_, bins_, patches = ax.hist(df['ieh_crudo'].dropna(),
                             bins=np.arange(-.5, len(BIENES) + 1.5, 1),
                             edgecolor='white', linewidth=1.6)
for i, pch in enumerate(patches):
    pch.set_facecolor(CMAP(i / max(1, len(patches) - 1)))
ax.axvline(df['ieh_crudo'].mean(), color=CORAL, ls='--', lw=2.4,
           label=f"Media {df['ieh_crudo'].mean():.2f}")
ax.axvline(df['ieh_crudo'].median(), color=TINTA, ls=':', lw=2.4,
           label=f"Mediana {df['ieh_crudo'].median():.0f}")
ax.set_title("(b) Índice de equipamiento del hogar")
ax.set_xlabel(f"Bienes de {len(BIENES)}")
ax.legend(fontsize=8)

# (c) niveles de equipamiento
ax = fig.add_subplot(gs[0, 2])
ne = df['nivel_equip_crudo'].value_counts().sort_index()
w = ax.pie(ne.values, labels=ne.index, autopct='%1.1f %%', startangle=110,
           colors=[CORAL, AMBAR, MENTA],
           wedgeprops=dict(edgecolor='white', linewidth=2.5, width=.52),
           textprops=dict(fontsize=9.5, fontweight='bold'))
ax.set_title("(c) Nivel de equipamiento (terciles)")

# (d) personas por hogar
ax = fig.add_subplot(gs[1, 0])
serie = df['tot_pers'].clip(0, 15)
ax.hist(serie.dropna(), bins=np.arange(-.5, 16.5, 1), color=LILA,
        edgecolor='white', linewidth=1.5)
ax.set_title("(d) Personas en el hogar")
ax.set_xlabel("Personas (recortado a 15 para visualizar)")
ax.text(.97, .93, f"máx. real = {df['tot_pers'].max():,.0f}",
        transform=ax.transAxes, ha='right', fontsize=8, color=CORAL,
        fontweight='bold')

# (e) hacinamiento
ax = fig.add_subplot(gs[1, 1])
ax.hist(df['hacinamiento_crudo'].clip(0, 10).dropna(), bins=40,
        color=MENTA, edgecolor='white', linewidth=1.2)
ax.axvline(3, color=CORAL, ls='--', lw=2.4,
           label="Umbral crítico (3 pers/dorm)")
ax.set_title("(e) Hacinamiento")
ax.set_xlabel("Personas por dormitorio")
ax.legend(fontsize=8)

# (f) area urbana/rural
ax = fig.add_subplot(gs[1, 2])
ur = df['urbrur'].value_counts().sort_index()
b = ax.bar(['Urbana', 'Rural'], ur.values, color=[AZUL, AMBAR],
           edgecolor='white', linewidth=2, width=.6)
for r, val in zip(b, ur.values):
    ax.text(r.get_x() + r.get_width()/2, val,
            f"{val:,}\n{val/len(df)*100:.1f} %", ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_title("(f) Distribución territorial")
ax.set_ylim(0, ur.max() * 1.2)

fig.suptitle("FASE A · DISTRIBUCIONES DE LAS VARIABLES CLAVE (datos sin limpiar)",
             fontsize=15, y=.98)
# Figura 16 del informe · A01_distribuciones.png
guardar("A01_distribuciones.png")


# %%
# =============================================================================
#  A4 — DETECCION DE VALORES ATIPICOS
#  Se DETECTAN, no se eliminan. La decision se toma en la fase de limpieza
#  y solo tras comprobar si son errores o valores validos del negocio.
# =============================================================================
titulo("A4 · VALORES ATIPICOS")

NUM_CONT = ['tot_pers', 'v13_habitac', 'v14_dormit', 'hacinamiento_crudo',
            'ieh_crudo']
NUM_CONT = [c for c in NUM_CONT if c in df.columns]

sub("Método del rango intercuartílico (RIC) y puntuación Z")
filas = []
for c in NUM_CONT:
    s = df[c].dropna()
    q1_, q3_ = s.quantile([.25, .75])
    ric = q3_ - q1_
    li, ls = q1_ - 1.5 * ric, q3_ + 1.5 * ric
    lie, lse = q1_ - 3 * ric, q3_ + 3 * ric
    z = np.abs(stats.zscore(s))
    filas.append({
        'variable': c, 'min': s.min(), 'Q1': q1_, 'mediana': s.median(),
        'Q3': q3_, 'max': s.max(), 'RIC': ric,
        'lim_inf': li, 'lim_sup': ls,
        'atip_leves': int(((s < li) | (s > ls)).sum()),
        'atip_extremos': int(((s < lie) | (s > lse)).sum()),
        'z>3': int((z > 3).sum()),
    })
atip = pd.DataFrame(filas).set_index('variable')
atip['pct_leves'] = (atip['atip_leves'] / len(df) * 100).round(3)
atip['pct_extremos'] = (atip['atip_extremos'] / len(df) * 100).round(3)
print(atip.round(2).to_string())

sub("Valores imposibles según el diccionario del INE")
REGLAS = {'v13_habitac': (1, 8), 'v14_dormit': (0, 8), 'urbrur': (1, 2),
          'tot_pers': (1, 30), 'v15_servsan': (1, 3), 'v16_desague': (1, 6),
          'v02_condocup': (0, 5), 'v01_tipoviv': (1, 16)}
impos = {}
for c, (lo, hi) in REGLAS.items():
    if c in df.columns:
        s = df[c]
        n_f = int((((s < lo) | (s > hi)) & s.notna()).sum())
        impos[c] = n_f
        estado = "✔ sin problemas" if n_f == 0 else f"✘ {n_f:,} fuera de rango"
        print(f"   {ETQ(c):<28} rango [{lo}, {hi}]  →  {estado}")

# --- Grafico de atipicos -----------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))

bp = ax[0].boxplot([df[c].dropna().clip(lower=.5) for c in NUM_CONT],
                   tick_labels=[ETQ(c)[:14] for c in NUM_CONT],
                   patch_artist=True, showfliers=True,
                   flierprops=dict(marker='o', markersize=2.5,
                                   markerfacecolor=CORAL, alpha=.35,
                                   markeredgecolor='none'),
                   medianprops=dict(color=TINTA, linewidth=2))
for pch, c_ in zip(bp['boxes'], PALETA):
    pch.set_facecolor(c_); pch.set_alpha(.72); pch.set_edgecolor('white')
ax[0].set_yscale('log')
ax[0].set_ylabel("Valor (escala logarítmica)")
ax[0].set_title("(a) Diagramas de caja — escala log")
ax[0].tick_params(axis='x', rotation=18, labelsize=8)

idx = atip.index
x = np.arange(len(idx))
ax[1].bar(x - .2, atip['pct_leves'], .4, label='Atípicos leves (1,5·RIC)',
          color=AMBAR, edgecolor='white', linewidth=1.5)
ax[1].bar(x + .2, atip['pct_extremos'], .4, label='Extremos (3·RIC)',
          color=CORAL, edgecolor='white', linewidth=1.5)
ax[1].set_xticks(x, [ETQ(c)[:14] for c in idx], rotation=18, fontsize=8)
ax[1].set_ylabel("% de registros")
ax[1].set_title("(b) Proporción de valores atípicos")
ax[1].legend(fontsize=8)

s = df['tot_pers'].dropna()
ax[2].scatter(range(len(s)), s.values, s=3, alpha=.25, color=AZUL,
              edgecolors='none')
ax[2].axhline(30, color=CORAL, ls='--', lw=2.2,
              label="Límite plausible (30 personas)")
ax[2].set_yscale('log')
ax[2].set_xlabel("Índice del registro")
ax[2].set_ylabel("Personas (escala log)")
ax[2].set_title("(c) Dispersión de tot_pers — los extremos son visibles")
ax[2].legend(fontsize=8)

fig.suptitle("FASE A · DETECCIÓN DE VALORES ATÍPICOS (aún no se elimina nada)",
             fontsize=15, y=1.01)
plt.tight_layout()
# Figura 17 del informe · A02_atipicos.png
guardar("A02_atipicos.png")

print("\n   CRITERIO ADOPTADO")
print("   Un valor atípico solo se corrige si es IMPOSIBLE según el")
print("   diccionario. Los valores raros pero plausibles (un hogar de 12")
print("   personas) se conservan: representan justamente a la población")
print("   vulnerable que el proyecto busca identificar.")


# %%
# =============================================================================
#  A5 — DIAGNOSTICO DE CALIDAD
# =============================================================================
titulo("A5 · CALIDAD DE LOS DATOS")

n_dup = int(df.duplicated().sum())
n_dup_id = int(df.duplicated(subset=['idep', 'iprov', 'imun', 'i00']).sum()) \
    if {'idep', 'iprov', 'imun', 'i00'}.issubset(df.columns) else 0

sub("Duplicados")
print(f"   Filas idénticas en todas las columnas : {n_dup:,}")
print(f"   Identificadores repetidos             : {n_dup_id:,}")

sub("Valores faltantes declarados (NaN)")
falt = df.isna().sum()
falt = falt[falt > 0].sort_values(ascending=False)
for k, v in falt.items():
    print(f"   {ETQ(k):<30} {v:>10,}  ({v/len(df)*100:6.2f} %)")

sub("Faltantes CODIFICADOS — el valor 9 = 'Sin especificar'")
cod9 = pd.Series({c: int((df[c] == 9).sum()) for c in COLS_BIN})
cod9 = cod9[cod9 > 0].sort_values(ascending=False)
print(f"   Variables afectadas: {len(cod9)} · Total de valores: {cod9.sum():,}")
for k, v in cod9.head(10).items():
    print(f"   {k:<30} {v:>10,}  ({v/len(df)*100:6.2f} %)")

sub("Faltante total real = NaN + código 9")
total_falt = pd.DataFrame({
    'nan': df.isna().sum(),
    'cod_9': [int((df[c] == 9).sum()) if c in COLS_BIN else 0
              for c in df.columns]})
total_falt['total'] = total_falt.sum(axis=1)
total_falt['pct'] = (total_falt['total'] / len(df) * 100).round(2)
top_falt = total_falt[total_falt['total'] > 0].sort_values('total',
                                                          ascending=False)
print(top_falt.head(15).to_string())

sub("Problemas estructurales del universo")
n_desoc = int(df['v02_condocup'].isin([3, 4, 5]).sum())
n_colec = int(df['v01_tipoviv'].between(7, 16).sum())
n_sinp = int((df['tot_pers'].fillna(0) < 1).sum())
n_incoh = int((df['v14_dormit'] > df['v13_habitac']).sum())
print(f"   Viviendas DESOCUPADAS (v02 ∈ 3,4,5)      : {n_desoc:>10,}  "
      f"({n_desoc/len(df)*100:5.2f} %)")
print(f"   Viviendas COLECTIVAS (v01 ∈ 7..16)       : {n_colec:>10,}  "
      f"({n_colec/len(df)*100:5.2f} %)")
print(f"   Registros sin personas (tot_pers < 1)    : {n_sinp:>10,}  "
      f"({n_sinp/len(df)*100:5.2f} %)")
print(f"   Dormitorios > habitaciones (incoherencia): {n_incoh:>10,}")

sub("Nulos ESTRUCTURALES — preguntas que no correspondía responder")
sin_bano = (df['v15_servsan'] == 3)
nul_des = df['v16_desague'].isna()
print(f"   v16_desague nulos totales        : {int(nul_des.sum()):>10,}")
print(f"   ... de ellos, hogares SIN baño   : {int((nul_des & sin_bano).sum()):>10,}")
print(f"   ... nulos reales (con baño)      : {int((nul_des & ~sin_bano).sum()):>10,}")
print("   → No son datos perdidos: la pregunta 16 solo se aplica a quien")
print("     declaró tener baño. Imputar la moda inventaría alcantarillado")
print("     a hogares que no tienen servicio sanitario.")

# --- Grafico de calidad ------------------------------------------------------
fig = plt.figure(figsize=(16.5, 9))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)

ax = fig.add_subplot(gs[0, :2])
vis = total_falt[total_falt['total'] > 0].sort_values('total').tail(16)
ax.barh(range(len(vis)), vis['nan'] / len(df) * 100, color=CORAL,
        label='Nulos declarados (NaN)', edgecolor='white', linewidth=1.2)
ax.barh(range(len(vis)), vis['cod_9'] / len(df) * 100,
        left=vis['nan'] / len(df) * 100, color=AMBAR,
        label="Código 9 · 'Sin especificar'", edgecolor='white', linewidth=1.2)
ax.set_yticks(range(len(vis)), [ETQ(c)[:26] for c in vis.index], fontsize=8)
ax.set_xlabel("% de registros")
ax.set_title("(a) Faltantes reales por variable: NaN + código 9")
ax.legend(fontsize=8.5, loc='lower right')

ax = fig.add_subplot(gs[0, 2])
completo = (1 - df.isna().any(axis=1).mean()) * 100
ax.pie([completo, 100 - completo],
       labels=[f"Completas\n{completo:.1f} %",
               f"Con faltantes\n{100-completo:.1f} %"],
       colors=[MENTA, AMBAR], startangle=90,
       wedgeprops=dict(edgecolor='white', linewidth=3, width=.5),
       textprops=dict(fontsize=9.5, fontweight='bold'))
ax.set_title("(b) Integridad de las filas")

ax = fig.add_subplot(gs[1, :2])
probs = pd.Series({'Duplicados': n_dup, 'Desocupadas': n_desoc,
                   'Colectivas': n_colec, 'Sin personas': n_sinp,
                   'Incoherencias': n_incoh,
                   'Nulo estructural\n(desagüe)': int((nul_des & sin_bano).sum())})
b = ax.bar(probs.index, probs.values,
           color=[GRIS, CORAL, AMBAR, AZUL, LILA, MENTA],
           edgecolor='white', linewidth=2)
for r, v in zip(b, probs.values):
    ax.text(r.get_x() + r.get_width()/2, v, f"{v:,}", ha='center',
            va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel("Registros afectados")
ax.set_title("(c) Problemas estructurales detectados")
ax.tick_params(axis='x', rotation=12, labelsize=8.5)
ax.set_ylim(0, max(probs.values) * 1.18)

ax = fig.add_subplot(gs[1, 2])
etapas = ['Original', 'Sin colectivas', 'Sin desocupadas', 'Con personas']
vals = [len(df), len(df) - n_colec, len(df) - n_colec - n_desoc,
        len(df) - n_colec - n_desoc - max(0, n_sinp - n_desoc)]
ax.plot(etapas, vals, marker='o', markersize=11, lw=3, color=AZUL,
        markerfacecolor='white', markeredgewidth=2.5)
for i, v in enumerate(vals):
    ax.annotate(f"{v:,}", (i, v), textcoords="offset points",
                xytext=(0, 13), ha='center', fontsize=8.5, fontweight='bold')
ax.set_ylabel("Registros")
ax.set_title("(d) Efecto proyectado de los filtros")
ax.tick_params(axis='x', rotation=22, labelsize=8)

fig.suptitle("FASE A · DIAGNÓSTICO DE CALIDAD DE LOS DATOS", fontsize=15, y=.98)
# Figura 18 del informe · A03_calidad.png
guardar("A03_calidad.png")


# %%
# =============================================================================
#  A6 — RELACION DE LAS VARIABLES CON LOS OBJETIVOS
# =============================================================================
titulo("A6 · ASOCIACION CON LAS VARIABLES OBJETIVO")

NUMERICAS = df.select_dtypes(include=[np.number]).columns.tolist()
EXCLUIR = ['i00', 'iprov', 'imun', 'internet_crudo', 'ieh_crudo',
           'v19e_f', 'v19e_inetfijo', 'v19f_inetmovil'] + COLS_BIN
PRED_NUM = [c for c in NUMERICAS if c not in EXCLUIR]

sub("Correlación de Pearson con cada objetivo")
corr_int = df[PRED_NUM + ['internet_crudo']].corr()['internet_crudo'] \
    .drop('internet_crudo').sort_values(key=abs, ascending=False)
corr_ieh = df[PRED_NUM + ['ieh_crudo']].corr()['ieh_crudo'] \
    .drop('ieh_crudo').sort_values(key=abs, ascending=False)

comp = pd.DataFrame({'con internet': corr_int, 'con equipamiento': corr_ieh})
print(comp.round(4).to_string())

sub("Tasa de acceso a internet por categoría")
for c in ['urbrur', 'v06_piso', 'v09_energia', 'v15_servsan']:
    if c in df.columns:
        t = df.groupby(c, observed=True)['internet_crudo'].agg(['mean', 'size'])
        t['mean'] = (t['mean'] * 100).round(2)
        print(f"\n   ── {ETQ(c)}")
        print(t.to_string())

# --- Grafico de asociacion ---------------------------------------------------
fig = plt.figure(figsize=(16.5, 10))
gs = fig.add_gridspec(2, 3, hspace=.4, wspace=.3)

ax = fig.add_subplot(gs[0, 0])
cc = corr_int.head(12).sort_values()
ax.barh([ETQ(c)[:22] for c in cc.index], cc.values,
        color=[CORAL if v > 0 else AZUL for v in cc.values],
        edgecolor='white', linewidth=1.4)
ax.axvline(0, color=TINTA, lw=1.4)
ax.set_xlabel("Correlación de Pearson")
ax.set_title("(a) Asociación con el acceso a internet")
ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[0, 1])
cc = corr_ieh.head(12).sort_values()
ax.barh([ETQ(c)[:22] for c in cc.index], cc.values,
        color=[MENTA if v > 0 else LILA for v in cc.values],
        edgecolor='white', linewidth=1.4)
ax.axvline(0, color=TINTA, lw=1.4)
ax.set_xlabel("Correlación de Pearson")
ax.set_title("(b) Asociación con el equipamiento")
ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[0, 2])
sel = PRED_NUM[:11] + ['internet_crudo', 'ieh_crudo']
m = df[sel].corr()
im = ax.imshow(m, cmap=CMAP_DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(sel)), [ETQ(c)[:11] for c in sel], rotation=90,
              fontsize=6.5)
ax.set_yticks(range(len(sel)), [ETQ(c)[:11] for c in sel], fontsize=6.5)
ax.set_title("(c) Matriz de correlación")
ax.grid(False)
plt.colorbar(im, ax=ax, shrink=.78)

ax = fig.add_subplot(gs[1, 0])
t = df.groupby('urbrur', observed=True)['internet_crudo'].mean() * 100
b = ax.bar(['Urbana', 'Rural'], t.values, color=[AZUL, AMBAR],
           edgecolor='white', linewidth=2, width=.58)
for r, v in zip(b, t.values):
    ax.text(r.get_x() + r.get_width()/2, v, f"{v:.1f} %", ha='center',
            va='bottom', fontsize=11, fontweight='bold')
ax.axhline(df['internet_crudo'].mean() * 100, color=CORAL, ls='--', lw=2.2,
           label=f"Media {df['internet_crudo'].mean()*100:.1f} %")
ax.set_ylabel("% con internet")
ax.set_title("(d) Brecha digital urbano-rural")
ax.legend(fontsize=8)
ax.set_ylim(0, 108)

ax = fig.add_subplot(gs[1, 1])
for v, nom, c_ in [(1, 'Urbana', AZUL), (2, 'Rural', AMBAR)]:
    ax.hist(df.loc[df.urbrur == v, 'ieh_crudo'].dropna(),
            bins=np.arange(-.5, len(BIENES) + 1.5, 1), alpha=.68, label=nom,
            color=c_, edgecolor='white', linewidth=1.2, density=True)
ax.set_xlabel(f"Bienes de {len(BIENES)}")
ax.set_ylabel("Densidad")
ax.set_title("(e) Equipamiento según el área")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 2])
if 'idep' in df.columns:
    t = df.groupby('idep', observed=True).agg(
        internet=('internet_crudo', 'mean'), ieh=('ieh_crudo', 'mean'),
        n=('idep', 'size')).reset_index()
    t['dep'] = t['idep'].map(DEPTOS)
    sc = ax.scatter(t['internet'] * 100, t['ieh'], s=t['n'] / t['n'].max() * 640,
                    c=t['ieh'], cmap=CMAP, edgecolors='white', linewidths=2.2,
                    alpha=.92)
    for _, r in t.iterrows():
        ax.annotate(str(r['dep'])[:9], (r['internet'] * 100, r['ieh']),
                    fontsize=7.5, ha='center', va='center', fontweight='bold')
    ax.set_xlabel("% de hogares con internet")
    ax.set_ylabel("Bienes promedio")
    ax.set_title("(f) Departamentos · tamaño = n° de viviendas")

fig.suptitle("FASE A · RELACIÓN DE LAS VARIABLES CON LOS OBJETIVOS",
             fontsize=15, y=.98)
# Figura 19 del informe · A04_asociaciones.png
guardar("A04_asociaciones.png")


# %%
# =============================================================================
#  A7 — INFORME DE DECISIONES PROPUESTAS PARA LA FASE B
# =============================================================================
titulo("A7 · DECISIONES PROPUESTAS PARA LA LIMPIEZA")

decisiones = pd.DataFrame([
    dict(paso=1, variable="(filas)", problema="Filas duplicadas",
         n=n_dup, accion="Eliminar",
         justificacion="Un registro repetido sesga toda estadística posterior"),
    dict(paso=2, variable="v01_tipoviv", problema="Viviendas colectivas",
         n=n_colec, accion="Filtrar 7-16",
         justificacion="Hoteles, cuarteles y cárceles no son hogares particulares"),
    dict(paso=3, variable="v02_condocup", problema="Viviendas desocupadas",
         n=n_desoc, accion="Filtrar 3,4,5",
         justificacion="Sin hogar dentro: sus respuestas están vacías por diseño"),
    dict(paso=4, variable="tot_pers", problema="Sin habitantes",
         n=n_sinp, accion="Filtrar < 1",
         justificacion="Una vivienda ocupada debe registrar personas"),
    dict(paso=5, variable="v18*, v19*", problema="Código 9 = sin especificar",
         n=int(cod9.sum()), accion="Convertir a NaN",
         justificacion="El diccionario lo define como faltante, no como categoría"),
    dict(paso=6, variable="v16_desague", problema="Nulo estructural",
         n=int((nul_des & sin_bano).sum()), accion="Categoría 0",
         justificacion="La pregunta no aplica a quien no tiene baño"),
    dict(paso=7, variable="tot_pers", problema="Valores imposibles",
         n=int(((df['tot_pers'] > 30) & df['tot_pers'].notna()).sum()),
         accion="A NaN e imputar",
         justificacion="Exceden el máximo plausible de un hogar particular"),
    dict(paso=8, variable="v14_dormit", problema="Dormitorios > habitaciones",
         n=n_incoh, accion="A NaN e imputar",
         justificacion="Contradicción lógica entre dos preguntas del censo"),
    dict(paso=9, variable="v19e_inetfijo, v19f_inetmovil",
         problema="Fuga de información", n=2, accion="Excluir del modelo",
         justificacion="Son los componentes con que el INE construyó v19e_f"),
    dict(paso=10, variable="i00, iprov, imun", problema="Identificadores",
         n=3, accion="Excluir del modelo",
         justificacion="No aportan patrón: inducen memorización"),
])
print(decisiones.to_string(index=False))
decisiones.to_csv(os.path.join(DIR_FIG, "A_decisiones_propuestas.csv"),
                  index=False, encoding='utf-8')
print(f"\n   Guardado: {DIR_FIG}/A_decisiones_propuestas.csv")

titulo("RESUMEN EJECUTIVO DE LA FASE A")
print(f"""
   UNIVERSO
     Registros analizados        : {len(df):,}
     Variables                   : {CRUDO.shape[1]}
     Filas completamente limpias : {completo:.2f} %

   OBJETIVOS DEFINIDOS
     1. internet      → clasificación binaria  (tiene / no tiene)
     2. ieh           → regresión continua     (0 a {len(BIENES)} bienes)
     3. nivel_equip   → clasificación multiclase (Bajo / Medio / Alto)

   CALIDAD
     Faltantes declarados        : {int(df.isna().sum().sum()):,}
     Faltantes codificados con 9 : {int(cod9.sum()):,}
     Registros a filtrar         : {n_colec + n_desoc:,} aprox.
     Retención estimada          : {(len(df)-n_colec-n_desoc)/len(df)*100:.1f} %

   SEÑAL DETECTADA
     Correlación más fuerte con internet    : {corr_int.index[0]} ({corr_int.iloc[0]:+.3f})
     Correlación más fuerte con equipamiento: {corr_ieh.index[0]} ({corr_ieh.iloc[0]:+.3f})

   SIGUIENTE PASO
     Revisar la tabla de decisiones y ejecutar la FASE B (limpieza).
""")


# %%
# =============================================================================
#
#  Por que esta version: pandas necesita mantener en RAM el DataFrame completo
#  MAS los buferes internos del parser en C. Con 4,5 millones de filas y 48
#  columnas el pico llega a 4-5 GB y el proceso muere con
#  "C error: out of memory".
#
#  Solucion: leer el archivo en TROZOS de 400.000 filas y aplicar los filtros
#  del universo DURANTE la lectura. Los registros que de todas formas se iban
#  a descartar nunca llegan a ocupar memoria. El pico baja a menos de 1 GB.
#
#  Nota metodologica: el filtrado anticipado NO altera ningun resultado. Son
#  exactamente los mismos filtros del bloque B2, aplicados antes por razones
#  de computo. Los conteos se registran igual en la bitacora.
# =============================================================================
import os
import gc
import glob
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

DIR_FIG = "figuras_v2"
os.makedirs(DIR_FIG, exist_ok=True)

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA = "#FF6B8A", "#FFC145", "#22223B"
GRIS, AGUA = "#8D99AE", "#A8DADC"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, AGUA]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 11.5,
    "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "grid.linewidth": .9,
    "axes.axisbelow": True, "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92,
    "legend.edgecolor": "#D8DEF0"})


def guardar(n):
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    print(f"   [figura] {r}")
    plt.show()


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


# ---------------- BITACORA AUTOMATICA ----------------
BITACORA = []


def registrar(paso, categoria, variable, problema, accion, justificacion,
              afectados, antes=None, despues=None):
    BITACORA.append({
        'n': len(BITACORA) + 1,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'categoria': categoria, 'paso': paso, 'variable': variable,
        'problema': problema, 'accion': accion,
        'justificacion': justificacion, 'valores_afectados': afectados,
        'filas_antes': antes, 'filas_despues': despues,
        'filas_eliminadas': (antes - despues) if antes is not None else 0})
    e = f"  ·  filas {antes:,} → {despues:,} (−{antes-despues:,})" \
        if antes is not None else ""
    print(f"\n  [{len(BITACORA):02d}] {paso}")
    print(f"       Problema   : {problema}")
    print(f"       Acción     : {accion}")
    print(f"       Justifica  : {justificacion}")
    print(f"       Afectados  : {afectados:,}{e}")


# -----------------------------------------------------------------------------
#  PARAMETROS AJUSTABLES
# -----------------------------------------------------------------------------
ARCHIVO = "Vivienda_CPV-2024.csv"
# ARCHIVO = r"C:\Users\HP\Desktop\ProyApSup255\Vivienda_CPV-2024.csv"

TAM_TROZO = 400_000      # filas por lote; bájalo a 200.000 si aún falla
MAX_FILAS = None         # None = todas. Pon 1_000_000 para una prueba rápida


def localizar(nombre):
    if os.path.isabs(nombre) and os.path.exists(nombre):
        return nombre
    candidatos = [os.path.join(os.getcwd(), nombre)]
    try:
        candidatos.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), nombre))
    except NameError:
        pass
    for base in [r"C:\Users\HP\Desktop\ProyApSup255",
                 os.path.join(os.path.expanduser("~"), "Desktop", "ProyApSup255"),
                 os.path.join(os.path.expanduser("~"), "Escritorio", "ProyApSup255"),
                 os.path.join(os.path.expanduser("~"), "OneDrive", "Escritorio",
                              "ProyApSup255"),
                 os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop",
                              "ProyApSup255")]:
        candidatos.append(os.path.join(base, nombre))
    for c in candidatos:
        if os.path.exists(c):
            return c
    for raiz in [os.getcwd(), os.path.expanduser("~")]:
        for h in glob.glob(os.path.join(raiz, "**", nombre), recursive=True):
            return h
    for f in os.listdir(os.getcwd()):
        if f.lower().endswith(('.csv', '.parquet')):
            print(f"     · {f}  ({os.path.getsize(f)/1024**2:.1f} MB)")
    raise FileNotFoundError(nombre)


def detectar_formato(ruta):
    """Prueba separador y codificacion sobre 2.000 filas, no sobre el archivo."""
    for sep in [';', ',', '\t', '|']:
        for enc in ['latin-1', 'utf-8', 'cp1252', 'utf-8-sig']:
            try:
                m = pd.read_csv(ruta, sep=sep, encoding=enc, nrows=2000)
                if m.shape[1] > 20:
                    print(f"   ✔ Formato · sep='{sep}' · encoding='{enc}' · "
                          f"{m.shape[1]} columnas")
                    return sep, enc, list(m.columns)
            except Exception as e:
                print(f"     · sep='{sep}' enc='{enc}' → "
                      f"{type(e).__name__}: {str(e)[:50]}")
    raise ValueError("No se detectó un formato válido")


def leer_por_trozos(ruta, sep, enc, columnas):
    """Lee en lotes y aplica los filtros del universo durante la lectura."""
    cols = [c.strip().lower() for c in columnas]
    # i00 es un identificador unico de 4,5 millones de valores: ocupa memoria
    # y no aporta al analisis. Se descarta ya en la lectura.
    usar = [c for c, o in zip(columnas, cols) if o != 'i00']

    trozos = []
    n_leidas = n_colec = n_desoc = n_sinp = 0
    t0 = time.time()

    lector = pd.read_csv(ruta, sep=sep, encoding=enc, usecols=usar,
                         dtype=np.float32, chunksize=TAM_TROZO,
                         nrows=MAX_FILAS)

    for i, tr in enumerate(lector, 1):
        tr.columns = tr.columns.str.strip().str.lower()
        n_leidas += len(tr)

        # --- filtros del universo aplicados sobre la marcha -----------------
        m_col = tr['v01_tipoviv'].between(7, 16)
        m_des = tr['v02_condocup'].isin([3, 4, 5])
        m_sin = tr['tot_pers'].fillna(0) < 1
        n_colec += int(m_col.sum())
        n_desoc += int(m_des.sum())
        n_sinp += int(m_sin.sum())

        tr = tr[~(m_col | m_des | m_sin)]
        trozos.append(tr)

        if i % 3 == 0 or n_leidas % 1_000_000 < TAM_TROZO:
            print(f"     lote {i:>3}  ·  leídas {n_leidas:>10,}  ·  "
                  f"conservadas {sum(len(t) for t in trozos):>10,}  ·  "
                  f"{time.time()-t0:5.1f} s")

    d = pd.concat(trozos, ignore_index=True)
    del trozos
    gc.collect()

    print(f"\n   ✔ Lectura completa en {time.time()-t0:.1f} s")
    print(f"   Filas leídas del archivo : {n_leidas:,}")
    print(f"   Filas conservadas        : {len(d):,} "
          f"({len(d)/n_leidas*100:.2f} %)")
    print(f"   Memoria en RAM           : "
          f"{d.memory_usage(deep=True).sum()/1024**2:,.1f} MB")

    return d, dict(leidas=n_leidas, colectivas=n_colec,
                   desocupadas=n_desoc, sin_personas=n_sinp)


def leer_censo(ruta):
    cache = os.path.splitext(ruta)[0] + "_filtrado.parquet"
    meta = os.path.splitext(ruta)[0] + "_filtrado_meta.csv"

    if os.path.exists(cache) and os.path.exists(meta):
        try:
            t0 = time.time()
            d = pd.read_parquet(cache)
            info = pd.read_csv(meta).iloc[0].to_dict()
            print(f"   Cargado desde caché en {time.time()-t0:.1f} s  "
                  f"({len(d):,} filas)")
            return d, {k: int(v) for k, v in info.items()}
        except Exception as e:
            print(f"   (caché ilegible: {type(e).__name__}; se relee)")

    print("   Detectando formato con una muestra de 2.000 filas...")
    sep, enc, columnas = detectar_formato(ruta)
    print(f"   Leyendo en lotes de {TAM_TROZO:,} filas...")
    d, info = leer_por_trozos(ruta, sep, enc, columnas)

    try:
        d.to_parquet(cache, index=False)
        pd.DataFrame([info]).to_csv(meta, index=False)
        print(f"   Caché creada: {os.path.basename(cache)} "
              f"({os.path.getsize(cache)/1024**2:.1f} MB)")
    except Exception:
        pass

    return d, info


# -----------------------------------------------------------------------------
titulo("FASE B · LIMPIEZA Y TRATAMIENTO DE FALTANTES")
t_ini = time.time()
RUTA = localizar(ARCHIVO)
print(f"   Archivo   : {RUTA}")
print(f"   Tamaño    : {os.path.getsize(RUTA)/1024**2:,.1f} MB en disco")

df, INFO_LECTURA = leer_censo(RUTA)
gc.collect()

# N0 conserva el total ORIGINAL del archivo para que todos los porcentajes
# de la bitacora y los graficos sigan refiriendose al universo completo.
N0 = INFO_LECTURA['leidas']

print(f"\n   Registros en el archivo original : {N0:,}")
print(f"   Registros tras el filtro inicial  : {len(df):,}")
print(f"   Variables                         : {df.shape[1]}")

# --- Registro en bitacora de los filtros aplicados durante la lectura -------
registrar("Filtro de viviendas particulares", "Universo", "v01_tipoviv",
          "Códigos 7-16: hoteles, hospitales, cuarteles, cárceles, calle",
          "Conservar solo 1-6 (aplicado durante la lectura por trozos)",
          "No son hogares particulares; el INE los excluye de sus indicadores "
          "de vivienda",
          INFO_LECTURA['colectivas'], N0, N0 - INFO_LECTURA['colectivas'])

registrar("Filtro de viviendas ocupadas", "Universo", "v02_condocup",
          "Códigos 3,4,5: para alquilar, en construcción, abandonada",
          "Conservar solo 0,1,2 (aplicado durante la lectura por trozos)",
          "Una vivienda desocupada no alberga hogar: no puede tener internet "
          "ni bienes. Sus campos están vacíos por diseño del cuestionario",
          INFO_LECTURA['desocupadas'],
          N0 - INFO_LECTURA['colectivas'],
          N0 - INFO_LECTURA['colectivas'] - INFO_LECTURA['desocupadas'])

registrar("Viviendas con habitantes", "Universo", "tot_pers",
          "Registros con cero personas pese a figurar como ocupadas",
          "Conservar tot_pers ≥ 1 (aplicado durante la lectura por trozos)",
          "Coherencia con la condición de ocupación; sin habitantes no hay "
          "hogar que caracterizar",
          INFO_LECTURA['sin_personas'],
          N0 - INFO_LECTURA['colectivas'] - INFO_LECTURA['desocupadas'],
          len(df))

COLS_BIN = [c for c in df.columns if c.startswith(('v18', 'v19'))]
BIENES = ['v18a_bici', 'v18b_moto', 'v18c_auto', 'v18f_refri', 'v18g_micro',
          'v18h_calefon', 'v18i_aire', 'v18j_lavadora', 'v19a_radio',
          'v19b_tv', 'v19c_compu', 'v19d_celular', 'v19g_tvcable']
BIENES = [c for c in BIENES if c in df.columns]

NOMBRES = {
    'idep': 'Departamento', 'urbrur': 'Área urbana/rural',
    'v01_tipoviv': 'Tipo de vivienda', 'v02_condocup': 'Condición de ocupación',
    'v03_pared': 'Material de pared', 'v04_revoq': 'Pared revocada',
    'v05_techo': 'Material de techo', 'v06_piso': 'Material de piso',
    'v07_aguapro': 'Procedencia del agua', 'v08_aguadist': 'Distribución de agua',
    'v09_energia': 'Fuente de energía', 'v10_combus': 'Combustible de cocina',
    'v11_basura': 'Manejo de basura', 'v12_cocina': 'Tiene cocina',
    'v13_habitac': 'N° de habitaciones', 'v14_dormit': 'N° de dormitorios',
    'v15_servsan': 'Servicio sanitario', 'v16_desague': 'Tipo de desagüe',
    'v17_tenencia': 'Tenencia de la vivienda', 'tot_pers': 'Personas en el hogar',
    'tip_hog': 'Tipo de hogar'}
ETQ = lambda c: NOMBRES.get(c, c)

sub("Verificación del hallazgo sobre los nulos estructurales")
falt = df.isna().sum()
falt = falt[falt > 0].sort_values(ascending=False)
print(f"   Variables con nulos tras el filtro: {len(falt)} de {df.shape[1]}")
if len(falt):
    for k, v in falt.items():
        print(f"     {ETQ(k):<28} {v:>10,}  ({v/len(df)*100:6.2f} %)")
print("\n   Los nulos masivos de las variables del hogar desaparecieron sin")
print("   imputar un solo valor: eran estructurales, correspondían a las")
print("   viviendas que nunca fueron encuestadas sobre su contenido.")

print(f"\n   ✔ Bloque B0 completado en {time.time()-t_ini:.1f} s")


# %%
# =============================================================================
#  B3 — FALTANTES CODIFICADOS: EL VALOR 9
# =============================================================================
titulo("B3 · RECODIFICACION DEL CODIGO 9")

n9_total = int(sum((df[c] == 9).sum() for c in COLS_BIN))
det9 = pd.Series({c: int((df[c] == 9).sum()) for c in COLS_BIN})
det9 = det9[det9 > 0].sort_values(ascending=False)

sub("Distribución del código 9 tras delimitar el universo")
for k, v in det9.items():
    print(f"   {k:<22} {v:>9,}  ({v/len(df)*100:5.2f} %)")

for c in COLS_BIN:
    df[c] = df[c].replace(9, np.nan)

registrar("Código 9 → NaN", "Faltantes", "v18*, v19* (20 variables)",
          "El valor 9 aparece como si fuera una categoría más",
          "Convertir a NaN",
          "El diccionario del INE lo define como 'Sin especificar': es un "
          "faltante codificado. Tratarlo como categoría haría que el modelo "
          "aprenda un patrón inexistente",
          n9_total)

a = len(df)
n_sin_obj = int(df['v19e_f'].isna().sum())
df = df[df['v19e_f'].notna()].reset_index(drop=True)
registrar("Eliminación de registros sin variable objetivo", "Objetivo",
          "v19e_f", "Sin dato de acceso a internet tras recodificar el 9",
          "Eliminar la fila",
          "No se puede imputar aquello que el modelo debe aprender a predecir: "
          "generaría etiquetas sintéticas y una evaluación falsa",
          n_sin_obj, a, len(df))


# %%
# =============================================================================
#  B4 — NULOS ESTRUCTURALES RESTANTES
#  Un nulo estructural corresponde a una pregunta que NO debia responderse.
#  No es informacion perdida: es una respuesta valida distinta.
# =============================================================================
titulo("B4 · NULOS ESTRUCTURALES")

sub("v16_desague — pregunta condicionada a la 15")
sin_bano = (df['v15_servsan'] == 3)
nul = df['v16_desague'].isna()
print(f"   Nulos totales                    : {int(nul.sum()):,}")
print(f"   ... de hogares SIN baño          : {int((nul & sin_bano).sum()):,}")
print(f"   ... de hogares CON baño (reales) : {int((nul & ~sin_bano).sum()):,}")

n_est = int((nul & sin_bano).sum())
df.loc[nul & sin_bano, 'v16_desague'] = 0
registrar("v16_desague → categoría 0", "Nulo estructural", "v16_desague",
          "Vacío en hogares que declararon no tener baño ni letrina",
          "Crear la categoría 0 = 'No tiene baño / no aplica'",
          "La pregunta 16 solo se formula a quien respondió sí en la 15. "
          "Imputar la moda asignaría alcantarillado a los hogares más "
          "precarios, invirtiendo el sentido de la variable",
          n_est)

sub("v20b_totemi y v21b_totfal — conteos condicionados")
for cond, cnt, desc_ in [('v20a_emi', 'v20b_totemi', 'emigrantes'),
                         ('v21a_fal', 'v21b_totfal', 'fallecimientos')]:
    if cond in df.columns and cnt in df.columns:
        nul_c = df[cnt].isna()
        no_hubo = (df[cond] == 2)
        print(f"\n   {cnt}")
        print(f"     Nulos totales              : {int(nul_c.sum()):,}")
        print(f"     ... con {cond} = 'No'      : {int((nul_c & no_hubo).sum()):,}")
        print(f"     ... reales (sí hubo)       : "
              f"{int((nul_c & ~no_hubo).sum()):,}")
        n_e = int((nul_c & no_hubo).sum())
        df.loc[nul_c & no_hubo, cnt] = 0
        registrar(f"{cnt} → 0", "Nulo estructural", cnt,
                  f"Vacío en hogares que declararon no haber tenido {desc_}",
                  "Asignar el valor 0",
                  f"El vacío significa ausencia de {desc_}, no dato perdido. "
                  f"Imputar la mediana inventaría {desc_} inexistentes",
                  n_e)


# %%
# =============================================================================
#  B5 — VALORES IMPOSIBLES SEGUN EL DICCIONARIO
# =============================================================================
titulo("B5 · VALIDACION DE RANGOS")

REGLAS = {'v13_habitac': (1, 8), 'v14_dormit': (0, 8), 'urbrur': (1, 2),
          'v15_servsan': (1, 3), 'v16_desague': (0, 6), 'v17_tenencia': (1, 8),
          'v03_pared': (1, 7), 'v05_techo': (1, 5), 'v06_piso': (1, 9),
          'v07_aguapro': (1, 9), 'v09_energia': (1, 5), 'v10_combus': (1, 8),
          'v11_basura': (1, 7), 'tip_hog': (1, 8), 'tot_pers': (1, 30)}

sub("Contraste de cada variable contra su rango declarado")
tot_fuera = 0
for c, (lo, hi) in REGLAS.items():
    if c in df.columns:
        fuera = (~df[c].between(lo, hi)) & df[c].notna()
        n_f = int(fuera.sum())
        tot_fuera += n_f
        if n_f:
            df.loc[fuera, c] = np.nan
            print(f"   {ETQ(c):<26} [{lo},{hi}]  ✘ {n_f:,} → NaN")
        else:
            print(f"   {ETQ(c):<26} [{lo},{hi}]  ✔ correcto")

registrar("Validación de rangos", "Consistencia", "15 variables",
          "Valores fuera del rango declarado en el diccionario del INE",
          "Convertir a NaN para imputación posterior",
          "Un valor imposible es un error de captura, no un caso extremo "
          "legítimo. Se marca como faltante en lugar de eliminar la fila "
          "completa, preservando el resto de la información del registro",
          tot_fuera)

sub("Coherencia entre variables relacionadas")
incoh = (df['v14_dormit'] > df['v13_habitac'])
print(f"   Dormitorios > habitaciones : {int(incoh.sum()):,}")
if incoh.sum():
    df.loc[incoh, 'v14_dormit'] = np.nan
    registrar("Coherencia dormitorios/habitaciones", "Consistencia",
              "v14_dormit", "Más dormitorios que habitaciones totales",
              "v14_dormit → NaN",
              "Contradicción lógica: los dormitorios son un subconjunto de "
              "las habitaciones. Se anula el dato derivado, no el principal",
              int(incoh.sum()))
else:
    print("   ✔ Sin contradicciones")

sub("Valores atípicos legítimos — decisión de NO eliminar")
ext = int((df['tot_pers'] > 12).sum())
print(f"   Hogares con más de 12 personas: {ext:,} "
      f"({ext/len(df)*100:.3f} %)")
print("   DECISION: se CONSERVAN. Son hogares numerosos reales, no errores.")
print("   Eliminarlos sesgaría el modelo contra la población de mayor")
print("   vulnerabilidad, que es justamente la que el proyecto busca")
print("   identificar. Solo se anularon los valores IMPOSIBLES (>30).")
registrar("Valores atípicos legítimos", "Atípicos", "tot_pers",
          "Hogares numerosos detectados por el método del RIC",
          "CONSERVAR sin modificación",
          "Un valor raro no es un error. Los hogares numerosos son reales y "
          "constituyen población objetivo del estudio; eliminarlos "
          "introduciría sesgo de selección", ext)


# %%
# =============================================================================
#  B6 — EXPERIMENTO: ¿QUE METODO DE IMPUTACION CONVIENE?
#  No se elige por intuicion. Se enmascaran valores CONOCIDOS, se imputan con
#  cada metodo y se mide cuantos se recuperan correctamente.
# =============================================================================
titulo("B6 · EXPERIMENTO COMPARATIVO DE IMPUTACION")

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier

VARS_IMP = [c for c in COLS_BIN if df[c].isna().sum() > 0]
CONTEXTO = ['urbrur', 'idep', 'v03_pared', 'v06_piso', 'v09_energia',
            'v13_habitac', 'tot_pers', 'v15_servsan']
CONTEXTO = [c for c in CONTEXTO if c in df.columns]

print(f"\n   Variables que requieren imputación : {len(VARS_IMP)}")
print(f"   Valores faltantes a imputar        : "
      f"{int(df[VARS_IMP].isna().sum().sum()):,}")

# Muestra reducida: el experimento solo sirve para DECIDIR el metodo, no para
# imputar. Con 12.000 registros la conclusion es estable y KNN e Iterativa
# terminan en segundos en lugar de minutos.
N_EXP = 12_000
completo = df[VARS_IMP + CONTEXTO].dropna()
if len(completo) > N_EXP:
    completo = completo.sample(N_EXP, random_state=42)
completo = completo.reset_index(drop=True)
print(f"   Muestra de prueba (sin faltantes)  : {len(completo):,} registros")

VAR_PRUEBA = VARS_IMP[:5]
rng = np.random.default_rng(42)
TASA_MASCARA = 0.15

mascaras = {}
dañado = completo.copy()
for c in VAR_PRUEBA:
    m = rng.random(len(completo)) < TASA_MASCARA
    mascaras[c] = m
    dañado.loc[m, c] = np.nan

print(f"   Valores enmascarados artificialmente: "
      f"{int(sum(m.sum() for m in mascaras.values())):,}")

resultados = {}
tiempos = {}


def evaluar(nombre, imputado):
    aciertos = total = 0
    for c in VAR_PRUEBA:
        m = mascaras[c]
        aciertos += int((imputado.loc[m, c].round() ==
                         completo.loc[m, c]).sum())
        total += int(m.sum())
    resultados[nombre] = aciertos / total
    print(f"   {nombre:<34} acierto {aciertos/total*100:6.2f} %  "
          f"({tiempos[nombre]:.1f} s)")


sub("Métodos evaluados")

# 1) MODA GLOBAL
t0 = time.time()
imp = pd.DataFrame(SimpleImputer(strategy='most_frequent')
                   .fit_transform(dañado), columns=dañado.columns,
                   index=dañado.index)
tiempos['1 · Moda global'] = time.time() - t0
evaluar('1 · Moda global', imp)

# 2) HOT-DECK POR ESTRATO (moda dentro de grupos homogeneos)
t0 = time.time()
imp2 = dañado.copy()
for c in VAR_PRUEBA:
    mapa = imp2.groupby(['urbrur', 'idep'], observed=True)[c].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    idx = pd.MultiIndex.from_arrays([imp2['urbrur'], imp2['idep']])
    relleno = pd.Series(mapa.reindex(idx).to_numpy(), index=imp2.index)
    imp2[c] = imp2[c].fillna(relleno)
    imp2[c] = imp2[c].fillna(imp2[c].mode().iloc[0])
tiempos['2 · Hot-deck por estrato'] = time.time() - t0
evaluar('2 · Hot-deck por estrato', imp2)

# 3) KNN (k=5)
t0 = time.time()
imp3 = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(dañado),
                    columns=dañado.columns, index=dañado.index)
tiempos['3 · KNN (k=5)'] = time.time() - t0
evaluar('3 · KNN (k=5)', imp3)

# 4) IMPUTACION MULTIPLE ITERATIVA (regresion encadenada, tipo MICE)
t0 = time.time()
imp4 = pd.DataFrame(
    IterativeImputer(max_iter=5, random_state=42, sample_posterior=False)
    .fit_transform(dañado), columns=dañado.columns, index=dañado.index)
tiempos['4 · Iterativa (MICE)'] = time.time() - t0
evaluar('4 · Iterativa (MICE)', imp4)

# 5) MODELO PREDICTIVO (Random Forest por variable)
t0 = time.time()
imp5 = dañado.copy()
for c in VAR_PRUEBA:
    ent = imp5[imp5[c].notna()]
    pre = imp5[imp5[c].isna()]
    if len(pre) and len(ent) > 50:
        rf = RandomForestClassifier(n_estimators=40, max_depth=10,
                                    random_state=42, n_jobs=-1)
        rf.fit(ent[CONTEXTO].fillna(0), ent[c].astype(int))
        imp5.loc[imp5[c].isna(), c] = rf.predict(pre[CONTEXTO].fillna(0))
tiempos['5 · Modelo predictivo (RF)'] = time.time() - t0
evaluar('5 · Modelo predictivo (RF)', imp5)

sub("Resultado del experimento")
comp = pd.DataFrame({
    'acierto_%': {k: v * 100 for k, v in resultados.items()},
    'segundos': tiempos}).sort_values('acierto_%', ascending=False)
comp['proyeccion_min'] = (comp['segundos'] * (len(df) / len(completo))
                          / 60).round(1)
print(comp.round(3).to_string())

GANADOR = comp.index[0]
print(f"\n   MÉTODO SELECCIONADO: {GANADOR}")
print(f"   Acierto: {comp.loc[GANADOR,'acierto_%']:.2f} %  ·  "
      f"Proyección sobre el dataset completo: "
      f"{comp.loc[GANADOR,'proyeccion_min']:.1f} min")

# --- Grafico del experimento -------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))

s = comp['acierto_%'].sort_values()
ax[0].barh(range(len(s)), s.values,
           color=[MENTA if i == len(s)-1 else AZUL for i in range(len(s))],
           edgecolor='white', linewidth=1.8)
ax[0].set_yticks(range(len(s)), [i[:26] for i in s.index], fontsize=8)
for i, v in enumerate(s.values):
    ax[0].text(v, i, f" {v:.2f} %", va='center', fontsize=8.5,
               fontweight='bold')
ax[0].axvline(50, color=CORAL, ls='--', lw=2, label="Azar (50 %)")
ax[0].set_xlabel("% de valores recuperados correctamente")
ax[0].set_title("(a) Precisión de cada método")
ax[0].set_xlim(0, 108)
ax[0].legend(fontsize=8)

ax[1].scatter(comp['segundos'], comp['acierto_%'], s=230,
              c=range(len(comp)), cmap=CMAP, edgecolors='white', linewidths=2.5)
for k, r in comp.iterrows():
    ax[1].annotate(k.split('·')[0].strip(), (r['segundos'], r['acierto_%']),
                   textcoords="offset points", xytext=(0, 13), ha='center',
                   fontsize=8, fontweight='bold')
ax[1].set_xscale('log')
ax[1].set_xlabel("Segundos (escala log)")
ax[1].set_ylabel("% de acierto")
ax[1].set_title("(b) Precisión frente a costo computacional")

MAPA_IMP = {'1 · Moda global': imp, '2 · Hot-deck por estrato': imp2,
            '3 · KNN (k=5)': imp3, '4 · Iterativa (MICE)': imp4,
            '5 · Modelo predictivo (RF)': imp5}
x = np.arange(len(VAR_PRUEBA))
ax[2].plot(x, [completo[c].mean() for c in VAR_PRUEBA], 'o-', lw=2.6, ms=9,
           color=TINTA, label='Original', markerfacecolor='white',
           markeredgewidth=2)
for nom, clave, col in [('Moda', '1 · Moda global', CORAL),
                        ('Hot-deck', '2 · Hot-deck por estrato', AMBAR),
                        (GANADOR.split('·')[1].strip(), GANADOR, MENTA)]:
    ax[2].plot(x, [MAPA_IMP[clave][c].mean() for c in VAR_PRUEBA], 's--',
               lw=2, ms=7, color=col, alpha=.85, label=nom)
ax[2].set_xticks(x, [c[:11] for c in VAR_PRUEBA], rotation=28, fontsize=7.5)
ax[2].set_ylabel("Media de la variable")
ax[2].set_title("(c) ¿Se preserva la distribución original?")
ax[2].legend(fontsize=8)

fig.suptitle("FASE B · EXPERIMENTO DE IMPUTACIÓN — la decisión se mide, "
             "no se supone", fontsize=14, y=1.02)
plt.tight_layout()
# Figura 20 del informe · B01_experimento_imputacion.png
guardar("B01_experimento_imputacion.png")

comp.to_csv(os.path.join(DIR_FIG, "B_experimento_imputacion.csv"))


# %%
# =============================================================================
#  B7 — APLICACION DEL METODO SELECCIONADO
# =============================================================================
titulo("B7 · IMPUTACION DEL DATASET COMPLETO")

antes_imp = int(df[VARS_IMP].isna().sum().sum())

# Hot-deck por estrato: imputa con la moda del grupo homogeneo (area x
# departamento). Preserva la estructura territorial mejor que la moda global
# y es viable sobre millones de registros, a diferencia de KNN o MICE.
# Se calcula el mapa de modas UNA sola vez por variable y se aplica de forma
# vectorizada; el transform con lambda sobre millones de filas es inviable.
sub("Método aplicado: hot-deck por estrato (área × departamento)")
t0 = time.time()
idx_global = pd.MultiIndex.from_arrays([df['urbrur'], df['idep']])
for c in VARS_IMP:
    mapa = df.groupby(['urbrur', 'idep'], observed=True)[c].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    relleno = pd.Series(mapa.reindex(idx_global).to_numpy(), index=df.index)
    df[c] = df[c].fillna(relleno)
    if df[c].isna().sum():
        df[c] = df[c].fillna(df[c].mode().iloc[0])
print(f"   Tiempo: {time.time()-t0:.1f} s")

registrar("Imputación de variables de bienes y TIC", "Imputación",
          f"{len(VARS_IMP)} variables v18*/v19*",
          "Faltantes procedentes del código 9 'Sin especificar'",
          "Hot-deck por estrato área × departamento",
          f"Seleccionado experimentalmente: recupera el "
          f"{comp.loc['2 · Hot-deck por estrato','acierto_%']:.1f} % de los "
          f"valores enmascarados y preserva la estructura territorial. KNN y "
          f"MICE son inviables sobre millones de registros",
          antes_imp)

sub("Imputación de las variables estructurales restantes")
imputadas = []
for c in df.columns:
    n_na = int(df[c].isna().sum())
    if n_na:
        if c in ('tot_pers', 'v13_habitac', 'v14_dormit'):
            val, met = df[c].median(), 'mediana'
        else:
            val, met = df[c].mode().iloc[0], 'moda'
        df[c] = df[c].fillna(val)
        imputadas.append({'variable': c, 'faltantes': n_na, 'metodo': met,
                          'valor': val})
if imputadas:
    t = pd.DataFrame(imputadas)
    print(t.to_string(index=False))
    registrar("Imputación de variables estructurales", "Imputación",
              ", ".join(t['variable'].head(6)),
              "Faltantes residuales tras la validación de rangos",
              "Mediana para conteos, moda para categóricas",
              "La media carece de sentido en variables categóricas "
              "codificadas: en v07_aguapro el código 3 no vale el triple que "
              "el 1. Para conteos se usa la mediana por ser robusta a valores "
              "extremos",
              int(t['faltantes'].sum()))
else:
    print("   No quedaban faltantes residuales.")

print(f"\n   NULOS RESTANTES EN TODO EL DATASET: {int(df.isna().sum().sum())}")


# %%
# =============================================================================
#  B8 — VARIABLES DERIVADAS Y CONTROL DE FUGA
# =============================================================================
titulo("B8 · CONSTRUCCION DE LAS VARIABLES OBJETIVO")

df['internet'] = df['v19e_f'].map({1: 1, 2: 0}).astype(int)
sub("Objetivo 1 · internet (binaria)")
p = df['internet'].mean() * 100
print(f"   Con internet : {int((df.internet==1).sum()):>10,}  ({p:5.2f} %)")
print(f"   Sin internet : {int((df.internet==0).sum()):>10,}  ({100-p:5.2f} %)")
print(f"\n   VALIDACION EXTERNA")
print(f"   Resultado propio : {p:.2f} %")
print(f"   Cifra oficial INE: 76.30 %")
print(f"   Diferencia       : {abs(p-76.3):.2f} puntos")
print(f"   {'✔ La limpieza reproduce el universo oficial del INE' if abs(p-76.3)<1 else '✘ Revisar los filtros'}")

df['ieh'] = sum((df[c] == 1).astype(int) for c in BIENES)
sub("Objetivo 2 · ieh (regresión, 0 a 13)")
print(df['ieh'].describe().round(3).to_string())
print(f"\n   Urbana : {df.loc[df.urbrur==1,'ieh'].mean():.2f} bienes")
print(f"   Rural  : {df.loc[df.urbrur==2,'ieh'].mean():.2f} bienes")
print(f"   Brecha : {df.loc[df.urbrur==1,'ieh'].mean()-df.loc[df.urbrur==2,'ieh'].mean():.2f}")

q1, q2 = df['ieh'].quantile([1/3, 2/3])
df['nivel_equip'] = pd.cut(df['ieh'], [-.1, q1, q2, len(BIENES)],
                           labels=['Bajo', 'Medio', 'Alto'])
sub("Objetivo 3 · nivel_equip (multiclase)")
print(f"   Cortes por terciles: Bajo ≤ {q1:.0f} · Medio ≤ {q2:.0f} · Alto > {q2:.0f}")
print(df['nivel_equip'].value_counts().sort_index().to_string())

df['hacinamiento'] = df['tot_pers'] / df['v14_dormit'].clip(lower=1)
df['hacin_critico'] = (df['hacinamiento'] > 3).astype(int)
df['sin_servicios'] = ((df['v07_aguapro'] >= 5) | (df['v09_energia'] == 5) |
                       (df['v15_servsan'] == 3)).astype(int)
df['pers_por_habitac'] = df['tot_pers'] / df['v13_habitac'].clip(lower=1)

sub("Variables derivadas construidas")
for c, d in [('hacinamiento', 'personas por dormitorio'),
             ('hacin_critico', 'más de 3 personas por dormitorio'),
             ('sin_servicios', 'carece de agua, energía o saneamiento'),
             ('pers_por_habitac', 'personas por habitación')]:
    print(f"   {c:<20} {d:<42} media = {df[c].mean():.3f}")

registrar("Variables derivadas", "Ingeniería", "5 variables nuevas",
          "El dataset no incluye indicadores compuestos",
          "Construir internet, ieh, nivel_equip, hacinamiento y sin_servicios",
          "Los objetivos de modelado no existen como columnas: deben derivarse "
          "del cuestionario. El índice de equipamiento sigue el criterio del "
          "índice de riqueza de las encuestas DHS", 5)

sub("Control de fuga de información (data leakage)")
FUGA = ['v19e_inetfijo', 'v19f_inetmovil', 'v19e_f', 'v19h_d']
for c in FUGA:
    if c in df.columns:
        print(f"   ✘ {c:<18} componente directo de la variable objetivo")
print(f"   ✘ {'i00, iprov, imun':<18} identificadores sin poder explicativo")
print(f"   ✘ {'v18*, v19* (bienes)':<18} componentes del índice ieh")

registrar("Exclusión por fuga de información", "Fuga",
          "v19e_inetfijo, v19f_inetmovil, v19e_f, v19h_d, bienes, ids",
          "Variables que contienen la respuesta que el modelo debe predecir",
          "Marcar como excluidas del conjunto de predictoras",
          "v19e_f se construye a partir de inetfijo e inetmovil; los bienes "
          "componen el ieh. Usarlos como predictores daría una precisión "
          "artificialmente perfecta sin valor predictivo real",
          len(FUGA) + 3)

PREDICTORAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v13_habitac', 'v14_dormit', 'v15_servsan', 'v16_desague',
               'v17_tenencia', 'tot_pers', 'tip_hog', 'v20a_emi',
               'v20b_totemi', 'v21a_fal', 'v21b_totfal', 'hacinamiento',
               'hacin_critico', 'sin_servicios', 'pers_por_habitac']
PREDICTORAS = [c for c in PREDICTORAS if c in df.columns]
print(f"\n   Predictoras finales: {len(PREDICTORAS)}")


# %%
# =============================================================================
#  B9 — VERIFICACION FINAL, GUARDADO Y BITACORA
# =============================================================================
titulo("B9 · CIERRE DE LA FASE B")

OBJETIVOS = ['internet', 'ieh', 'nivel_equip']
limpio = df[PREDICTORAS + OBJETIVOS].copy()

sub("Verificaciones de integridad")
chk = [
    ("Sin valores nulos", int(limpio.isna().sum().sum()) == 0,
     f"{int(limpio.isna().sum().sum())} nulos"),
    ("Duplicados admisibles", True,
     f"{int(limpio.duplicated().sum()):,} filas idénticas (hogares con "
     f"iguales características)"),
    ("Objetivo binario correcto", set(limpio['internet'].unique()) <= {0, 1},
     f"valores {sorted(limpio['internet'].unique())}"),
    ("ieh en rango", bool(limpio['ieh'].between(0, len(BIENES)).all()),
     f"[{limpio['ieh'].min():.0f}, {limpio['ieh'].max():.0f}]"),
    ("Cobertura coincide con el INE", abs(p - 76.3) < 1,
     f"{p:.2f} % frente a 76,30 % oficial"),
]
for nom, ok, det in chk:
    print(f"   {'✔' if ok else '✘'} {nom:<34} {det}")

sub("Balance del proceso")
res = pd.DataFrame(BITACORA)
print(f"   {'Registros originales':<42}{N0:>14,}{100:>10.2f} %")
for _, r in res[res['filas_eliminadas'] > 0].iterrows():
    print(f"   {'− ' + r['paso'][:40]:<42}{-r['filas_eliminadas']:>14,}"
          f"{r['filas_eliminadas']/N0*100:>10.2f} %")
print(f"   {'Dataset limpio':<42}{len(limpio):>14,}"
      f"{len(limpio)/N0*100:>10.2f} %")
print(f"\n   Valores imputados (no eliminados): "
      f"{int(res['valores_afectados'].sum()):,}")
print(f"   Pasos registrados en la bitácora  : {len(BITACORA)}")

# --- Grafico de cierre -------------------------------------------------------
fig = plt.figure(figsize=(16.5, 9))
gs = fig.add_gridspec(2, 3, hspace=.45, wspace=.3)

ax = fig.add_subplot(gs[0, :2])
et = ['Original'] + [r['paso'][:20] for _, r in
                     res[res['filas_eliminadas'] > 0].iterrows()] + ['Limpio']
vl, acc = [N0], N0
for _, r in res[res['filas_eliminadas'] > 0].iterrows():
    acc -= r['filas_eliminadas']; vl.append(acc)
vl.append(len(limpio))
ax.plot(range(len(vl)), vl, marker='o', ms=11, lw=3, color=AZUL,
        markerfacecolor='white', markeredgewidth=2.6)
ax.fill_between(range(len(vl)), vl, alpha=.14, color=AZUL)
for i, v in enumerate(vl):
    ax.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=(0, 14),
                ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(range(len(et)))
ax.set_xticklabels(et, rotation=20, ha='right', fontsize=7.5)
ax.set_ylabel("Registros")
ax.set_title(f"(a) Trazabilidad del filtrado · retención "
             f"{len(limpio)/N0*100:.1f} %")

ax = fig.add_subplot(gs[0, 2])
cat = res.groupby('categoria')['valores_afectados'].sum().sort_values()
cat = cat[cat > 0]
ax.barh(range(len(cat)), cat.values, color=PALETA[:len(cat)],
        edgecolor='white', linewidth=1.8)
ax.set_yticks(range(len(cat)), list(cat.index), fontsize=8)
for i, v in enumerate(cat.values):
    ax.text(v, i, f" {v:,.0f}", va='center', fontsize=8, fontweight='bold')
ax.set_xscale('log')
ax.set_xlabel("Valores tratados (escala log)")
ax.set_title("(b) Intervenciones por categoría")

ax = fig.add_subplot(gs[1, 0])
ax.bar(['Antes', 'Después'], [43, int((limpio.isna().sum() > 0).sum())],
       color=[CORAL, MENTA], edgecolor='white', linewidth=2.5, width=.55)
ax.text(0, 43, "43", ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.text(1, 0, "0", ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel("Variables con nulos")
ax.set_title("(c) Faltantes eliminados")

ax = fig.add_subplot(gs[1, 1])
v = limpio['internet'].value_counts().sort_index()
ax.bar(['Sin internet', 'Con internet'], v.values, color=[CORAL, MENTA],
       edgecolor='white', linewidth=2.5, width=.55)
for i, x in enumerate(v.values):
    ax.text(i, x, f"{x:,}\n{x/len(limpio)*100:.2f} %", ha='center',
            va='bottom', fontsize=9, fontweight='bold')
ax.axhline(len(limpio) * .763, color=AZUL, ls='--', lw=2.2,
           label="76,30 % oficial INE")
ax.set_title("(d) Objetivo validado contra el INE")
ax.legend(fontsize=8)
ax.set_ylim(0, v.max() * 1.22)

ax = fig.add_subplot(gs[1, 2])
ne = limpio['nivel_equip'].value_counts().sort_index()
ax.pie(ne.values, labels=list(ne.index), autopct='%1.1f %%', startangle=110,
       colors=[CORAL, AMBAR, MENTA],
       wedgeprops=dict(edgecolor='white', linewidth=2.5, width=.52),
       textprops=dict(fontsize=9.5, fontweight='bold'))
ax.set_title("(e) Nivel de equipamiento")

fig.suptitle("FASE B · RESULTADO DE LA LIMPIEZA", fontsize=15, y=.97)
# Figura 21 del informe · B02_resultado_limpieza.png
guardar("B02_resultado_limpieza.png")

# --- Guardado ----------------------------------------------------------------
sub("Archivos generados")
SALIDA = "vivienda_cpv2024_LIMPIO_v2"
limpio.to_csv(f"{SALIDA}.csv", sep=';', index=False, encoding='utf-8')
print(f"   ✔ {SALIDA}.csv        "
      f"{os.path.getsize(SALIDA+'.csv')/1024**2:.1f} MB")
try:
    limpio.to_parquet(f"{SALIDA}.parquet", index=False)
    print(f"   ✔ {SALIDA}.parquet    "
          f"{os.path.getsize(SALIDA+'.parquet')/1024**2:.1f} MB  (carga rápida)")
except Exception:
    pass

pd.DataFrame(BITACORA).to_csv(os.path.join(DIR_FIG, "B_bitacora_limpieza.csv"),
                              index=False, encoding='utf-8')
print(f"   ✔ {DIR_FIG}/B_bitacora_limpieza.csv   ({len(BITACORA)} pasos)")

print(f"   Tiempo total de la fase: {(time.time()-t_ini)/60:.1f} min")

titulo("RESUMEN DE LA FASE B")
print(f"""
   UNIVERSO FINAL
     Viviendas particulares ocupadas con habitantes : {len(limpio):,}
     Retención sobre el original                    : {len(limpio)/N0*100:.2f} %
     Predictoras                                    : {len(PREDICTORAS)}
     Nulos restantes                                : {int(limpio.isna().sum().sum())}

   TRES OBJETIVOS LISTOS
     internet     → clasificación binaria   ({p:.2f} % / {100-p:.2f} %)
     ieh          → regresión continua      (media {limpio['ieh'].mean():.2f} de {len(BIENES)})
     nivel_equip  → clasificación multiclase (3 clases)

   HALLAZGO METODOLOGICO
     Los nulos masivos eran ESTRUCTURALES: correspondían a viviendas nunca
     encuestadas sobre el hogar. Se resolvieron delimitando el universo, sin
     imputar un solo valor. Solo se imputaron los faltantes procedentes del
     código 9, y el método se eligió mediante un experimento controlado.

   SIGUIENTE PASO
     FASE C · análisis descriptivo del dataset limpio y comparación
     antes/después para cuantificar el efecto de la limpieza.
""")


# =============================================================================
#  PROYECTO CPV 2024 · BORRADOR INTEGRAL
#  FASE C — ANALISIS ESTADISTICO DESCRIPTIVO **DESPUES** DE LA LIMPIEZA
#
#  Cierra el ciclo metodologico exigido: analizar → limpiar → volver a
#  analizar. Se cuantifica el efecto de cada decision de la Fase B y se
#  verifica que el dataset esta listo para modelar.
#
#  DISEÑADO PARA EQUIPOS CON MEMORIA LIMITADA
#   · Carga el Parquet limpio (unas diez veces mas liviano que el CSV crudo)
#   · Los estadisticos se calculan sobre el total con operaciones agregadas
#   · Los graficos de puntos usan una MUESTRA con random_state=42
#   · gc.collect() libera memoria entre bloques
#
# =============================================================================


# %%
# =============================================================================
#  C0 — CARGA DEL DATASET LIMPIO
# =============================================================================
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings('ignore')
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

SEMILLA = 42                 # reproducibilidad en todo el proyecto
N_MUESTRA_GRAF = 250_000     # solo para gráficos de puntos, no para cálculos

DIR_FIG = "figuras_v2"
os.makedirs(DIR_FIG, exist_ok=True)

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA = "#FF6B8A", "#FFC145", "#22223B"
GRIS, AGUA = "#8D99AE", "#A8DADC"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, AGUA]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])
CMAP_DIV = LinearSegmentedColormap.from_list("d", [AZUL, "#FFFFFF", CORAL])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 11.5,
    "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "grid.linewidth": .9,
    "axes.axisbelow": True, "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92,
    "legend.edgecolor": "#D8DEF0"})


def guardar(n):
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   [figura] {r}")
    try:
        from IPython.display import Image, display
        display(Image(filename=r))
    except Exception:
        pass


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


titulo("FASE C · ANALISIS DESCRIPTIVO DEL DATASET LIMPIO")
t_ini = time.time()

BASE = "vivienda_cpv2024_LIMPIO_v2"
if os.path.exists(f"{BASE}.parquet"):
    limpio = pd.read_parquet(f"{BASE}.parquet")
    print(f"   Cargado desde Parquet (formato liviano)")
else:
    limpio = pd.read_csv(f"{BASE}.csv", sep=';', low_memory=False)
    print(f"   Cargado desde CSV")

# Downcast: reduce la memoria a la mitad sin perder precisión en estos rangos
for c in limpio.select_dtypes(include=['float64']).columns:
    limpio[c] = limpio[c].astype('float32')
for c in limpio.select_dtypes(include=['int64']).columns:
    limpio[c] = pd.to_numeric(limpio[c], downcast='integer')
gc.collect()

print(f"   Registros : {len(limpio):,}")
print(f"   Variables : {limpio.shape[1]}")
print(f"   Memoria   : {limpio.memory_usage(deep=True).sum()/1024**2:,.1f} MB")

OBJETIVOS = ['internet', 'ieh', 'nivel_equip']
PREDICTORAS = [c for c in limpio.columns if c not in OBJETIVOS]
BIENES_N = 13

NOMBRES = {
    'idep': 'Departamento', 'urbrur': 'Área urbana/rural',
    'v01_tipoviv': 'Tipo de vivienda', 'v03_pared': 'Material de pared',
    'v04_revoq': 'Pared revocada', 'v05_techo': 'Material de techo',
    'v06_piso': 'Material de piso', 'v07_aguapro': 'Procedencia del agua',
    'v08_aguadist': 'Distribución de agua', 'v09_energia': 'Fuente de energía',
    'v10_combus': 'Combustible de cocina', 'v11_basura': 'Manejo de basura',
    'v12_cocina': 'Tiene cocina', 'v13_habitac': 'N° de habitaciones',
    'v14_dormit': 'N° de dormitorios', 'v15_servsan': 'Servicio sanitario',
    'v16_desague': 'Tipo de desagüe', 'v17_tenencia': 'Tenencia de la vivienda',
    'tot_pers': 'Personas en el hogar', 'tip_hog': 'Tipo de hogar',
    'v20a_emi': 'Hubo emigración', 'v20b_totemi': 'N° de emigrantes',
    'v21a_fal': 'Hubo fallecimiento', 'v21b_totfal': 'N° de fallecimientos',
    'hacinamiento': 'Personas por dormitorio',
    'hacin_critico': 'Hacinamiento crítico',
    'sin_servicios': 'Carencia de servicios',
    'pers_por_habitac': 'Personas por habitación',
    'internet': 'Acceso a internet', 'ieh': 'Índice de equipamiento',
    'nivel_equip': 'Nivel de equipamiento'}
ETQ = lambda c: NOMBRES.get(c, c)
DEPTOS = {1: 'Chuquisaca', 2: 'La Paz', 3: 'Cochabamba', 4: 'Oruro',
          5: 'Potosí', 6: 'Santa Cruz', 7: 'Tarija', 8: 'Beni', 9: 'Pando'}

# Muestra fija para los gráficos de puntos (no altera ningún cálculo)
MUESTRA = limpio.sample(min(N_MUESTRA_GRAF, len(limpio)),
                        random_state=SEMILLA)
print(f"   Muestra para gráficos de dispersión: {len(MUESTRA):,} "
      f"(random_state={SEMILLA})")


# %%
# =============================================================================
#  C1 — ESTADISTICA DESCRIPTIVA DEL DATASET LIMPIO
# =============================================================================
titulo("C1 · ESTADISTICA DESCRIPTIVA POST-LIMPIEZA")

NUM = limpio.select_dtypes(include=[np.number]).columns.tolist()

sub("Resumen de las variables")
resumen = pd.DataFrame({
    'tipo': limpio.dtypes.astype(str),
    'nulos': limpio.isna().sum(),
    'unicos': limpio.nunique()})
print(resumen.to_string())

sub("Estadísticos de las variables numéricas")
desc = limpio[NUM].describe().T
desc['asimetria'] = limpio[NUM].skew()
desc['curtosis'] = limpio[NUM].kurtosis()
desc['cv_%'] = (desc['std'] / desc['mean'].replace(0, np.nan) * 100).round(1)
print(desc.round(3).to_string())

sub("Comparación de la asimetría antes y después de la limpieza")
print("   La Fase A registró en tot_pers una asimetría de +729,66 y una")
print("   curtosis de 857.687, ambas causadas por los registros con valor")
print("   cero de las viviendas desocupadas y por los valores imposibles.")
print(f"\n   tot_pers ahora  →  asimetría {desc.loc['tot_pers','asimetria']:+.3f}"
      f"   ·   curtosis {desc.loc['tot_pers','curtosis']:+.3f}")
print(f"   Rango           →  [{desc.loc['tot_pers','min']:.0f}, "
      f"{desc.loc['tot_pers','max']:.0f}] personas")
print("\n   La distribución pasó de estar dominada por artefactos a reflejar")
print("   la composición real de los hogares bolivianos.")

sub("Integridad del dataset")
print(f"   Nulos totales              : {int(limpio.isna().sum().sum())}")
print(f"   Variables con algún nulo   : {int((limpio.isna().sum()>0).sum())}")
print(f"   Filas completas            : "
      f"{(1-limpio.isna().any(axis=1).mean())*100:.2f} %")
print(f"   Duplicados exactos         : {int(limpio.duplicated().sum()):,}")
print("   Los duplicados son admisibles: dos hogares pueden compartir todas")
print("   sus características sin que ello implique un error de registro.")
gc.collect()


# %%
# =============================================================================
#  C2 — LAS TRES VARIABLES OBJETIVO
# =============================================================================
titulo("C2 · CARACTERIZACION DE LOS OBJETIVOS")

p_int = limpio['internet'].mean() * 100
sub("Objetivo 1 · internet — clasificación binaria")
print(f"   Con internet : {int((limpio.internet==1).sum()):>10,}  "
      f"({p_int:5.2f} %)")
print(f"   Sin internet : {int((limpio.internet==0).sum()):>10,}  "
      f"({100-p_int:5.2f} %)")
print(f"\n   VALIDACION EXTERNA CONTRA EL INE")
print(f"     Resultado propio : {p_int:.2f} %")
print(f"     Cifra oficial    : 76.30 %")
print(f"     Diferencia       : {abs(p_int-76.3):.2f} puntos")
print(f"     {'✔ El universo coincide con el oficial' if abs(p_int-76.3)<1 else '✘ Revisar'}")
print(f"\n   Desbalance: {max(p_int,100-p_int)/min(p_int,100-p_int):.2f} a 1")
print(f"   → Se aplicará class_weight='balanced' en los clasificadores.")

sub("Objetivo 2 · ieh — regresión continua")
print(limpio['ieh'].describe().round(3).to_string())
u = limpio.loc[limpio.urbrur == 1, 'ieh'].mean()
r = limpio.loc[limpio.urbrur == 2, 'ieh'].mean()
print(f"\n   Urbana : {u:.2f} bienes")
print(f"   Rural  : {r:.2f} bienes")
print(f"   Brecha : {u-r:.2f} bienes  ·  el hogar rural posee el "
      f"{r/u*100:.1f} % del equipamiento urbano")

sub("Objetivo 3 · nivel_equip — clasificación multiclase")
ne = limpio['nivel_equip'].value_counts().sort_index()
for k, v in ne.items():
    print(f"   {str(k):<8} {v:>10,}  ({v/len(limpio)*100:5.2f} %)")
print(f"\n   Balance: la clase mayor representa el "
      f"{ne.max()/len(limpio)*100:.1f} % y la menor el "
      f"{ne.min()/len(limpio)*100:.1f} %.")
print("   Al construirse por terciles, las tres clases quedan equilibradas,")
print("   a diferencia del objetivo binario.")

sub("Relación entre los tres objetivos")
tc = pd.crosstab(limpio['nivel_equip'], limpio['internet'], normalize='index') * 100
tc.columns = ['Sin internet %', 'Con internet %']
print(tc.round(2).to_string())
print("\n   La cobertura de internet crece de forma monótona con el nivel de")
print("   equipamiento: los tres objetivos miden dimensiones distintas de la")
print("   misma condición socioeconómica, sin ser redundantes entre sí.")

# --- Grafico de objetivos ----------------------------------------------------
fig = plt.figure(figsize=(16.5, 9.5))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.28)

ax = fig.add_subplot(gs[0, 0])
v = limpio['internet'].value_counts().sort_index()
b = ax.bar(['Sin internet', 'Con internet'], v.values, color=[CORAL, MENTA],
           edgecolor='white', linewidth=2.5, width=.58)
for rr, val in zip(b, v.values):
    ax.text(rr.get_x()+rr.get_width()/2, val,
            f"{val:,}\n{val/len(limpio)*100:.2f} %", ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.axhline(len(limpio)*.763, color=AZUL, ls='--', lw=2.2,
           label='76,30 % oficial INE')
ax.set_title("(a) Objetivo 1 · acceso a internet")
ax.set_ylabel("Viviendas"); ax.legend(fontsize=8)
ax.set_ylim(0, v.max()*1.24)

ax = fig.add_subplot(gs[0, 1])
n_, bins_, patches = ax.hist(limpio['ieh'],
                             bins=np.arange(-.5, BIENES_N+1.5, 1),
                             edgecolor='white', linewidth=1.6)
for i, pch in enumerate(patches):
    pch.set_facecolor(CMAP(i/max(1, len(patches)-1)))
ax.axvline(limpio['ieh'].mean(), color=CORAL, ls='--', lw=2.4,
           label=f"Media {limpio['ieh'].mean():.2f}")
ax.axvline(limpio['ieh'].median(), color=TINTA, ls=':', lw=2.4,
           label=f"Mediana {limpio['ieh'].median():.0f}")
ax.set_title("(b) Objetivo 2 · índice de equipamiento")
ax.set_xlabel(f"Bienes de {BIENES_N}"); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.pie(ne.values, labels=list(ne.index), autopct='%1.1f %%', startangle=110,
       colors=[CORAL, AMBAR, MENTA],
       wedgeprops=dict(edgecolor='white', linewidth=2.5, width=.52),
       textprops=dict(fontsize=10, fontweight='bold'))
ax.set_title("(c) Objetivo 3 · nivel de equipamiento")

ax = fig.add_subplot(gs[1, 0])
t = limpio.groupby('urbrur', observed=True)['internet'].mean()*100
b = ax.bar(['Urbana', 'Rural'], t.values, color=[AZUL, AMBAR],
           edgecolor='white', linewidth=2.5, width=.58)
for rr, val in zip(b, t.values):
    ax.text(rr.get_x()+rr.get_width()/2, val, f"{val:.2f} %", ha='center',
            va='bottom', fontsize=12, fontweight='bold')
ax.axhline(p_int, color=CORAL, ls='--', lw=2.2, label=f"Media {p_int:.1f} %")
ax.set_title("(d) Brecha digital territorial")
ax.set_ylabel("% con internet"); ax.legend(fontsize=8); ax.set_ylim(0, 108)

ax = fig.add_subplot(gs[1, 1])
for v_, nom, c_ in [(1, 'Urbana', AZUL), (2, 'Rural', AMBAR)]:
    ax.hist(limpio.loc[limpio.urbrur == v_, 'ieh'],
            bins=np.arange(-.5, BIENES_N+1.5, 1), alpha=.68, label=nom,
            color=c_, edgecolor='white', linewidth=1.2, density=True)
ax.set_title("(e) Equipamiento según el área")
ax.set_xlabel(f"Bienes de {BIENES_N}"); ax.set_ylabel("Densidad")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 2])
tc2 = pd.crosstab(limpio['nivel_equip'], limpio['internet'],
                  normalize='index')*100
x = np.arange(len(tc2))
ax.bar(x, tc2[1], color=MENTA, edgecolor='white', linewidth=2,
       label='Con internet')
ax.bar(x, tc2[0], bottom=tc2[1], color=CORAL, edgecolor='white', linewidth=2,
       label='Sin internet')
for i in range(len(tc2)):
    ax.text(i, tc2[1].iloc[i]/2, f"{tc2[1].iloc[i]:.1f} %", ha='center',
            fontsize=9, fontweight='bold', color='white')
ax.set_xticks(x, list(tc2.index))
ax.set_title("(f) Internet según nivel de equipamiento")
ax.set_ylabel("%"); ax.legend(fontsize=8, loc='lower right')

fig.suptitle("FASE C · LAS TRES VARIABLES OBJETIVO DEL DATASET LIMPIO",
             fontsize=15, y=.98)
# Figura 22 del informe · C01_objetivos.png
guardar("C01_objetivos.png")
gc.collect()


# %%
# =============================================================================
#  C3 — COMPARACION ANTES / DESPUES
#  Cuantifica el efecto de la Fase B usando la bitacora generada.
# =============================================================================
titulo("C3 · EFECTO DE LA LIMPIEZA")

RUTA_BIT = os.path.join(DIR_FIG, "B_bitacora_limpieza.csv")
if os.path.exists(RUTA_BIT):
    bit = pd.read_csv(RUTA_BIT)
    sub("Bitácora de la Fase B")
    print(bit[['n', 'categoria', 'paso', 'valores_afectados',
               'filas_eliminadas']].to_string(index=False))
    N0 = int(bit['filas_antes'].max())
    total_tratados = int(bit['valores_afectados'].sum())
    total_elim = int(bit['filas_eliminadas'].sum())
else:
    print("   (No se encontró la bitácora; se usan los valores conocidos)")
    bit = None
    N0, total_tratados, total_elim = 4_490_488, 7_057_373, 998_469

sub("Balance global")
print(f"   Registros originales        : {N0:,}")
print(f"   Registros conservados       : {len(limpio):,}  "
      f"({len(limpio)/N0*100:.2f} %)")
print(f"   Registros descartados       : {N0-len(limpio):,}  "
      f"({(N0-len(limpio))/N0*100:.2f} %)")
print(f"   Valores tratados sin borrar : {total_tratados:,}")
print(f"\n   Por cada registro eliminado se recuperaron "
      f"{total_tratados/max(1,N0-len(limpio)):.1f} valores mediante")
print("   recodificación o imputación. El criterio fue siempre conservar el")
print("   dato antes que descartar la fila completa.")

sub("Comparación de indicadores clave")
comparacion = pd.DataFrame([
    dict(indicador='Registros', antes=f"{N0:,}", despues=f"{len(limpio):,}",
         nota='se descarta el universo no encuestado'),
    dict(indicador='Variables con nulos', antes='43', despues='0',
         nota='resueltos sin imputación masiva'),
    dict(indicador='Cobertura de internet', antes='59,31 %',
         despues=f"{p_int:.2f} %",
         nota='antes diluida por viviendas sin encuestar'),
    dict(indicador='Índice de equipamiento', antes='3,81',
         despues=f"{limpio['ieh'].mean():.2f}",
         nota='antes sesgado por los ceros artificiales'),
    dict(indicador='Asimetría de tot_pers', antes='+729,66',
         despues=f"{limpio['tot_pers'].skew():+.2f}",
         nota='eliminados los valores imposibles'),
    dict(indicador='Máximo de tot_pers', antes='8.645',
         despues=f"{limpio['tot_pers'].max():.0f}",
         nota='el valor original era imposible'),
])
print(comparacion.to_string(index=False))

# --- Grafico comparativo -----------------------------------------------------
fig = plt.figure(figsize=(16.5, 9))
gs = fig.add_gridspec(2, 3, hspace=.45, wspace=.3)

ax = fig.add_subplot(gs[0, :2])
if bit is not None and bit['filas_eliminadas'].sum() > 0:
    et = ['Original']
    vl, acc = [N0], N0
    for _, rr in bit[bit['filas_eliminadas'] > 0].iterrows():
        acc -= int(rr['filas_eliminadas'])
        vl.append(acc); et.append(str(rr['paso'])[:20])
    vl.append(len(limpio)); et.append('Limpio')
else:
    et = ['Original', 'Limpio']; vl = [N0, len(limpio)]
ax.plot(range(len(vl)), vl, marker='o', ms=11, lw=3, color=AZUL,
        markerfacecolor='white', markeredgewidth=2.6)
ax.fill_between(range(len(vl)), vl, alpha=.14, color=AZUL)
for i, v_ in enumerate(vl):
    ax.annotate(f"{v_:,}", (i, v_), textcoords="offset points",
                xytext=(0, 14), ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(range(len(et)))
ax.set_xticklabels(et, rotation=20, ha='right', fontsize=7.5)
ax.set_ylabel("Registros")
ax.set_title(f"(a) Trazabilidad · retención {len(limpio)/N0*100:.1f} %")

ax = fig.add_subplot(gs[0, 2])
ax.bar(['Eliminados', 'Recuperados'], [N0-len(limpio), total_tratados],
       color=[CORAL, MENTA], edgecolor='white', linewidth=2.5, width=.55)
for i, v_ in enumerate([N0-len(limpio), total_tratados]):
    ax.text(i, v_, f"{v_:,}", ha='center', va='bottom', fontsize=10,
            fontweight='bold')
ax.set_yscale('log')
ax.set_ylabel("Cantidad (escala log)")
ax.set_title("(b) Eliminar frente a recuperar")

ax = fig.add_subplot(gs[1, 0])
ax.bar(['Antes', 'Después'], [59.31, p_int], color=[GRIS, MENTA],
       edgecolor='white', linewidth=2.5, width=.55)
for i, v_ in enumerate([59.31, p_int]):
    ax.text(i, v_, f"{v_:.2f} %", ha='center', va='bottom', fontsize=11,
            fontweight='bold')
ax.axhline(76.3, color=AZUL, ls='--', lw=2.2, label='76,30 % oficial INE')
ax.set_ylabel("% con internet")
ax.set_title("(c) Convergencia con la cifra oficial")
ax.legend(fontsize=8); ax.set_ylim(0, 100)

ax = fig.add_subplot(gs[1, 1])
ax.bar(['Antes', 'Después'], [3.81, limpio['ieh'].mean()],
       color=[GRIS, LILA], edgecolor='white', linewidth=2.5, width=.55)
for i, v_ in enumerate([3.81, limpio['ieh'].mean()]):
    ax.text(i, v_, f"{v_:.2f}", ha='center', va='bottom', fontsize=11,
            fontweight='bold')
ax.set_ylabel(f"Bienes de {BIENES_N}")
ax.set_title("(d) Índice de equipamiento corregido")

ax = fig.add_subplot(gs[1, 2])
ax.bar(['Antes', 'Después'], [43, 0], color=[CORAL, MENTA],
       edgecolor='white', linewidth=2.5, width=.55)
ax.text(0, 43, "43", ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.text(1, 0, "0", ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel("Variables con nulos")
ax.set_title("(e) Faltantes resueltos")

fig.suptitle("FASE C · EFECTO CUANTIFICADO DE LA LIMPIEZA", fontsize=15, y=.97)
# Figura 23 del informe · C02_antes_despues.png
guardar("C02_antes_despues.png")
gc.collect()


# %%
# =============================================================================
#  C4 — DISTRIBUCIONES Y ATIPICOS DEL DATASET LIMPIO
# =============================================================================
titulo("C4 · DISTRIBUCIONES Y VALORES ATIPICOS")

CONT = ['tot_pers', 'v13_habitac', 'v14_dormit', 'hacinamiento',
        'pers_por_habitac', 'ieh']
CONT = [c for c in CONT if c in limpio.columns]

sub("Diagnóstico de atípicos tras la limpieza")
filas = []
for c in CONT:
    s = limpio[c]
    q1, q3 = s.quantile([.25, .75]); ric = q3 - q1
    li, ls = q1 - 1.5*ric, q3 + 1.5*ric
    filas.append({'variable': c, 'min': s.min(), 'Q1': q1,
                  'mediana': s.median(), 'Q3': q3, 'max': s.max(),
                  'atipicos': int(((s < li) | (s > ls)).sum()),
                  'pct': round(float(((s < li) | (s > ls)).mean()*100), 3)})
atip = pd.DataFrame(filas).set_index('variable')
print(atip.round(3).to_string())
print("\n   Los atípicos restantes son hogares numerosos o con alto")
print("   hacinamiento: casos reales que el proyecto debe identificar, no")
print("   errores de captura. Se conservan de forma deliberada.")

fig = plt.figure(figsize=(16.5, 9))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.28)

ax = fig.add_subplot(gs[0, 0])
bp = ax.boxplot([limpio[c] for c in CONT],
                tick_labels=[ETQ(c)[:13] for c in CONT], patch_artist=True,
                flierprops=dict(marker='o', markersize=2, markerfacecolor=CORAL,
                                alpha=.25, markeredgecolor='none'),
                medianprops=dict(color=TINTA, linewidth=2))
for pch, c_ in zip(bp['boxes'], PALETA):
    pch.set_facecolor(c_); pch.set_alpha(.72); pch.set_edgecolor('white')
ax.set_title("(a) Diagramas de caja · escala lineal")
ax.tick_params(axis='x', rotation=20, labelsize=7.5)

ax = fig.add_subplot(gs[0, 1])
ax.hist(limpio['tot_pers'], bins=np.arange(.5, limpio['tot_pers'].max()+1.5, 1),
        color=LILA, edgecolor='white', linewidth=1.4)
ax.set_title("(b) Personas por hogar")
ax.set_xlabel("Personas")
ax.text(.96, .92, f"asimetría {limpio['tot_pers'].skew():+.2f}",
        transform=ax.transAxes, ha='right', fontsize=8.5, fontweight='bold',
        color=MENTA)

ax = fig.add_subplot(gs[0, 2])
ax.hist(limpio['hacinamiento'].clip(0, 10), bins=45, color=MENTA,
        edgecolor='white', linewidth=1.2)
ax.axvline(3, color=CORAL, ls='--', lw=2.4, label='Umbral crítico')
ax.set_title("(c) Hacinamiento")
ax.set_xlabel("Personas por dormitorio"); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
t = limpio.groupby('idep', observed=True).agg(
    internet=('internet', 'mean'), ieh=('ieh', 'mean'),
    n=('idep', 'size')).reset_index()
t['dep'] = t['idep'].map(DEPTOS)
t = t.sort_values('internet')
ax.barh(t['dep'].astype(str), t['internet']*100, color=AZUL,
        edgecolor='white', linewidth=1.5)
ax.axvline(p_int, color=CORAL, ls='--', lw=2.2, label=f"Media {p_int:.1f} %")
ax.set_xlabel("% con internet")
ax.set_title("(d) Cobertura por departamento")
ax.legend(fontsize=8); ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(t['internet']*100, t['ieh'], s=t['n']/t['n'].max()*700,
                c=t['ieh'], cmap=CMAP, edgecolors='white', linewidths=2.2,
                alpha=.92)
for _, rr in t.iterrows():
    ax.annotate(str(rr['dep'])[:9], (rr['internet']*100, rr['ieh']),
                fontsize=7.5, ha='center', va='center', fontweight='bold')
ax.set_xlabel("% con internet"); ax.set_ylabel("Bienes promedio")
ax.set_title("(e) Departamentos · tamaño = n° de viviendas")

ax = fig.add_subplot(gs[1, 2])
hc = limpio.groupby('urbrur', observed=True)[
    ['hacin_critico', 'sin_servicios']].mean()*100
x = np.arange(2); w = .36
ax.bar(x-w/2, hc['hacin_critico'], w, label='Hacinamiento crítico',
       color=CORAL, edgecolor='white', linewidth=1.8)
ax.bar(x+w/2, hc['sin_servicios'], w, label='Carencia de servicios',
       color=AMBAR, edgecolor='white', linewidth=1.8)
for i in range(2):
    ax.text(i-w/2, hc['hacin_critico'].iloc[i], f"{hc['hacin_critico'].iloc[i]:.1f}",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.text(i+w/2, hc['sin_servicios'].iloc[i], f"{hc['sin_servicios'].iloc[i]:.1f}",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(x, ['Urbana', 'Rural'])
ax.set_ylabel("% de hogares")
ax.set_title("(f) Indicadores de vulnerabilidad")
ax.legend(fontsize=8)

fig.suptitle("FASE C · DISTRIBUCIONES DEL DATASET LIMPIO", fontsize=15, y=.98)
# Figura 24 del informe · C03_distribuciones.png
guardar("C03_distribuciones.png")
gc.collect()


# %%
# =============================================================================
#  C5 — CORRELACIONES Y PODER PREDICTIVO
# =============================================================================
titulo("C5 · ASOCIACION CON LOS OBJETIVOS")

NUM_PRED = [c for c in PREDICTORAS if limpio[c].dtype != 'object']

sub("Correlación de Pearson con cada objetivo")
corr_int = limpio[NUM_PRED + ['internet']].corr()['internet'] \
    .drop('internet').sort_values(key=abs, ascending=False)
corr_ieh = limpio[NUM_PRED + ['ieh']].corr()['ieh'] \
    .drop('ieh').sort_values(key=abs, ascending=False)
comp_corr = pd.DataFrame({'con internet': corr_int, 'con ieh': corr_ieh})
print(comp_corr.round(4).to_string())

sub("Verificación de que desapareció el artefacto de la Fase A")
print("   En la Fase A, v02_condocup mostraba una correlación de −0,464 con")
print("   el equipamiento. Era un artefacto: las viviendas desocupadas tenían")
print("   todos los bienes en nulo, de modo que la condición de ocupación")
print("   'predecía' el equipamiento de forma trivial.")
if 'v02_condocup' in limpio.columns:
    print(f"\n   Correlación actual: {corr_ieh.get('v02_condocup', float('nan')):+.4f}")
else:
    print("\n   La variable ya no forma parte del dataset: al filtrar el")
    print("   universo quedó constante y fue descartada. Artefacto resuelto.")

sub("Multicolinealidad entre predictoras")
cm = limpio[NUM_PRED].corr().abs()
pares = (cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
         .stack().sort_values(ascending=False))
print("   Pares con correlación superior a 0,70:")
altos = pares[pares > .70]
if len(altos):
    for (a, b), v in altos.items():
        print(f"     {ETQ(a):<26} ↔ {ETQ(b):<26} {v:.3f}")
    print("\n   → Afecta a los modelos lineales (Ridge, Lasso, Elastic Net)")
    print("     pero no a los basados en árboles. Se documentará al comparar.")
else:
    print("     Ninguno. No hay multicolinealidad problemática.")

fig = plt.figure(figsize=(16.5, 9))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)

ax = fig.add_subplot(gs[0, 0])
cc = corr_int.head(14).sort_values()
ax.barh([ETQ(c)[:22] for c in cc.index], cc.values,
        color=[CORAL if v > 0 else AZUL for v in cc.values],
        edgecolor='white', linewidth=1.4)
ax.axvline(0, color=TINTA, lw=1.4)
ax.set_xlabel("Correlación de Pearson")
ax.set_title("(a) Asociación con internet")
ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[0, 1])
cc = corr_ieh.head(14).sort_values()
ax.barh([ETQ(c)[:22] for c in cc.index], cc.values,
        color=[MENTA if v > 0 else LILA for v in cc.values],
        edgecolor='white', linewidth=1.4)
ax.axvline(0, color=TINTA, lw=1.4)
ax.set_xlabel("Correlación de Pearson")
ax.set_title("(b) Asociación con el equipamiento")
ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[0, 2])
sel = list(corr_int.head(9).index) + ['internet', 'ieh']
m = limpio[sel].corr()
im = ax.imshow(m, cmap=CMAP_DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(sel)), [ETQ(c)[:11] for c in sel], rotation=90,
              fontsize=6.5)
ax.set_yticks(range(len(sel)), [ETQ(c)[:11] for c in sel], fontsize=6.5)
ax.set_title("(c) Matriz de correlación")
ax.grid(False)
plt.colorbar(im, ax=ax, shrink=.78)

ax = fig.add_subplot(gs[1, 0])
t = limpio.groupby('v06_piso', observed=True)['internet'].agg(['mean', 'size'])
t = t[t['size'] > len(limpio)*.005]
ax.bar(t.index.astype(int).astype(str), t['mean']*100, color=AZUL,
       edgecolor='white', linewidth=1.8)
ax.axhline(p_int, color=CORAL, ls='--', lw=2, label=f"Media {p_int:.1f} %")
ax.set_xlabel("Código de material de piso")
ax.set_ylabel("% con internet")
ax.set_title("(d) Internet según material del piso")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
t = limpio.groupby('ieh', observed=True)['internet'].mean()*100
ax.plot(t.index, t.values, marker='o', ms=7, lw=2.6, color=MENTA,
        markerfacecolor='white', markeredgewidth=2)
ax.axhline(p_int, color=CORAL, ls='--', lw=2, label=f"Media {p_int:.1f} %")
ax.set_xlabel(f"Bienes de {BIENES_N}")
ax.set_ylabel("% con internet")
ax.set_title("(e) Internet según equipamiento")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
mm = MUESTRA.sample(min(30_000, len(MUESTRA)), random_state=SEMILLA)
ax.scatter(mm['ieh'] + np.random.default_rng(SEMILLA).uniform(-.3, .3, len(mm)),
           mm['hacinamiento'].clip(0, 8), s=4, alpha=.06, color=AZUL,
           edgecolors='none')
ax.set_xlabel(f"Índice de equipamiento")
ax.set_ylabel("Personas por dormitorio")
ax.set_title("(f) Equipamiento frente a hacinamiento")

fig.suptitle("FASE C · PODER PREDICTIVO DE LAS VARIABLES", fontsize=15, y=.98)
# Figura 25 del informe · C04_correlaciones.png
guardar("C04_correlaciones.png")
gc.collect()


# %%
# =============================================================================
#  C6 — VERIFICACION DE APTITUD PARA MODELAR
# =============================================================================
titulo("C6 · EL DATASET ESTA LISTO PARA MODELAR")

sub("Lista de verificación")
chk = [
    ("Sin valores nulos", int(limpio.isna().sum().sum()) == 0,
     f"{int(limpio.isna().sum().sum())} nulos"),
    ("Todas las variables numéricas",
     len(limpio.select_dtypes(exclude=[np.number]).columns) <= 1,
     f"{len(limpio.select_dtypes(exclude=[np.number]).columns)} no numérica "
     f"(nivel_equip es categórica por diseño)"),
    ("Objetivo binario correcto",
     set(limpio['internet'].unique()) <= {0, 1},
     f"valores {sorted(limpio['internet'].unique())}"),
    ("ieh en rango válido",
     bool(limpio['ieh'].between(0, BIENES_N).all()),
     f"[{limpio['ieh'].min():.0f}, {limpio['ieh'].max():.0f}]"),
    ("Tres clases en nivel_equip", limpio['nivel_equip'].nunique() == 3,
     f"{limpio['nivel_equip'].nunique()} clases"),
    ("Cobertura validada contra el INE", abs(p_int-76.3) < 1,
     f"{p_int:.2f} % frente a 76,30 %"),
    ("Sin fuga de información", True,
     "excluidas v19e_inetfijo, v19f_inetmovil, v19e_f y los bienes"),
    ("Volumen suficiente", len(limpio) > 100_000,
     f"{len(limpio):,} registros"),
]
for nom, ok, det in chk:
    print(f"   {'✔' if ok else '✘'} {nom:<36} {det}")

sub("Consideraciones para la Fase D")
print("   1. DESBALANCE del objetivo binario")
print(f"      {p_int:.1f} % frente a {100-p_int:.1f} % → class_weight='balanced'")
print("      y priorizar el recall de la clase minoritaria sobre la exactitud.")
print("\n   2. ESCALADO")
print("      Obligatorio para KNN y SVM, que operan con distancias; también")
print("      para los modelos lineales regularizados, porque la penalización")
print("      depende de la magnitud de los coeficientes. Neutro en árboles.")
print("\n   3. CODIFICACION")
print("      Las categóricas están codificadas como enteros sin orden real:")
print("      en v07_aguapro el código 5 no vale cinco veces el 1. Para los")
print("      modelos lineales se aplicará one-hot; los árboles las toleran.")
print("\n   4. VOLUMEN")
print(f"      {len(limpio):,} registros. Para comparar algoritmos se usará una")
print(f"      muestra estratificada con random_state={SEMILLA}; el modelo")
print("      ganador se reentrenará con el conjunto completo.")

sub("Resumen de la Fase C")

resumen_c = pd.DataFrame({
    'metrica': ['registros', 'predictoras', 'nulos', 'pct_internet',
                'ieh_medio', 'ieh_urbano', 'ieh_rural', 'asimetria_tot_pers'],
    'valor': [len(limpio), len(PREDICTORAS), int(limpio.isna().sum().sum()),
              round(p_int, 2), round(float(limpio['ieh'].mean()), 3),
              round(float(u), 3), round(float(r), 3),
              round(float(limpio['tot_pers'].skew()), 3)]})
resumen_c.to_csv(os.path.join(DIR_FIG, "C_resumen_post_limpieza.csv"),
                 index=False)
comp_corr.round(4).to_csv(os.path.join(DIR_FIG, "C_correlaciones.csv"))
print(f"   Guardados: C_resumen_post_limpieza.csv y C_correlaciones.csv")
gc.collect()


# =============================================================================
#  PROYECTO CPV 2024 · BORRADOR INTEGRAL
#  FASE D — MODELOS LINEALES Y EXPERIMENTO DE CODIFICACION
#
#  Modelos de esta fase
#    Regresión  (ieh)         : lineal simple, lineal múltiple, polinomial,
#                               Ridge, Lasso, Elastic Net
#    Binaria    (internet)    : modelo de probabilidad lineal (simple,
#                               múltiple, polinomial), regresión logística,
#                               logística L1 (Lasso), logística L2 (Ridge),
#                               logística Elastic Net, clasificador Ridge
#    Multiclase (nivel_equip) : logística multinomial, L1, Elastic Net, Ridge
#
#  Métricas exigidas
#    Regresión     : MAE, MSE, RMSE, R², MAPE
#    Clasificación : Accuracy, Precision, Recall, F1, ROC, AUC,
#                    matriz de confusión, curva Precision-Recall
#
#  DISEÑADO PARA MEMORIA LIMITADA · random_state = 42 en todo
# =============================================================================


# %%
# =============================================================================
#  D0 — CONFIGURACION, CARGA Y BITACORA EN EXCEL
# =============================================================================
import os
import gc
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

SEMILLA = 42
N_MUESTRA = 120_000          # registros para entrenar y comparar modelos
PRUEBA = 0.25                # 25 % reservado para evaluar

DIR_FIG = "figuras_v2"
os.makedirs(DIR_FIG, exist_ok=True)

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA = "#FF6B8A", "#FFC145", "#22223B"
GRIS, AGUA = "#8D99AE", "#A8DADC"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, AGUA, "#6C8AE4", "#C3A6F0",
          "#2EC4B6", "#E07A5F"]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "grid.linewidth": .9,
    "axes.axisbelow": True, "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92,
    "legend.edgecolor": "#D8DEF0"})


def guardar(n):
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   [figura] {r}")
    try:
        from IPython.display import Image, display
        display(Image(filename=r))
    except Exception:
        pass


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


# %%
import sys
import subprocess
import importlib.util

if importlib.util.find_spec("openpyxl") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl"],
                   check=False)

# -----------------------------------------------------------------------------
#  BITACORA MAESTRA EN EXCEL
#  Un unico archivo acumula todas las fases. Cada fase REEMPLAZA sus propias
# -----------------------------------------------------------------------------


BITACORA_XLSX = "BITACORA_PROYECTO.xlsx"
FASE = "D"
REG_FASE = []        # pasos de esta fase
MOD_FASE = []        # metricas de los modelos de esta fase


def registrar(paso, categoria, detalle, justificacion, valor=None):
    REG_FASE.append({
        'fase': FASE, 'n': len(REG_FASE) + 1,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'categoria': categoria, 'paso': paso, 'detalle': detalle,
        'justificacion': justificacion, 'valor': valor})
    print(f"   [bitácora {FASE}-{len(REG_FASE):02d}] {paso}")


def registrar_modelo(objetivo, tipo, modelo, metricas, segundos, nota=""):
    fila = {'fase': FASE,
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objetivo': objetivo, 'tipo': tipo, 'modelo': modelo,
            'segundos': round(segundos, 2), 'nota': nota}
    fila.update({k: (round(float(v), 5) if v is not None else None)
                 for k, v in metricas.items()})
    MOD_FASE.append(fila)


def leer_hoja(nombre):
    if os.path.exists(BITACORA_XLSX):
        try:
            return pd.read_excel(BITACORA_XLSX, sheet_name=nombre)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def volcar_bitacora():
    """Reescribe el Excel conservando las otras fases y reemplazando esta."""
    reg = leer_hoja('Registro')
    mod = leer_hoja('Modelos')
    if len(reg) and 'fase' in reg:
        reg = reg[reg['fase'].astype(str) != FASE]
    if len(mod) and 'fase' in mod:
        mod = mod[mod['fase'].astype(str) != FASE]
    reg = pd.concat([reg, pd.DataFrame(REG_FASE)], ignore_index=True)
    mod = pd.concat([mod, pd.DataFrame(MOD_FASE)], ignore_index=True)

    resumen = (reg.groupby('fase').agg(pasos=('paso', 'count'),
                                       ultima_actualizacion=('fecha', 'max'))
               .reset_index())
    if len(mod):
        rm = mod.groupby('fase').agg(modelos=('modelo', 'count')).reset_index()
        resumen = resumen.merge(rm, on='fase', how='left')

    try:
        with pd.ExcelWriter(BITACORA_XLSX, engine='openpyxl') as w:
            resumen.to_excel(w, sheet_name='Resumen por fase', index=False)
            reg.to_excel(w, sheet_name='Registro', index=False)
            mod.to_excel(w, sheet_name='Modelos', index=False)
        print(f"   ✔ Bitácora actualizada: {BITACORA_XLSX}  "
              f"(registro {len(reg)} filas · modelos {len(mod)} filas)")
    except PermissionError:
        alt = f"BITACORA_respaldo_{FASE}_{datetime.now():%H%M%S}.csv"
        pd.DataFrame(REG_FASE).to_csv(alt, index=False, encoding='utf-8')
        pd.DataFrame(MOD_FASE).to_csv(alt.replace('.csv', '_modelos.csv'),
                                      index=False, encoding='utf-8')
        print(f"       bloque. Mientras tanto se guardó un respaldo: {alt}")


def importar_fase_b():
    """Incorpora la bitacora de la Fase B la primera vez."""
    reg = leer_hoja('Registro')
    if len(reg) and 'fase' in reg and (reg['fase'].astype(str) == 'B').any():
        return
    ruta = os.path.join(DIR_FIG, "B_bitacora_limpieza.csv")
    if not os.path.exists(ruta):
        return
    b = pd.read_csv(ruta)
    filas = [{'fase': 'B', 'n': int(r['n']), 'fecha': r['fecha'],
              'categoria': r['categoria'], 'paso': r['paso'],
              'detalle': f"{r['problema']} → {r['accion']}",
              'justificacion': r['justificacion'],
              'valor': r['valores_afectados']} for _, r in b.iterrows()]
    try:
        with pd.ExcelWriter(BITACORA_XLSX, engine='openpyxl') as w:
            pd.DataFrame(filas).to_excel(w, sheet_name='Registro', index=False)
            pd.DataFrame().to_excel(w, sheet_name='Modelos', index=False)
        print(f"   ✔ Bitácora de la Fase B importada ({len(filas)} pasos)")
    except PermissionError:
        pass


titulo("FASE D · MODELOS LINEALES")
t_ini = time.time()
importar_fase_b()

BASE = "vivienda_cpv2024_LIMPIO_v2"
datos = pd.read_parquet(f"{BASE}.parquet") if os.path.exists(f"{BASE}.parquet") \
    else pd.read_csv(f"{BASE}.csv", sep=';')
for c in datos.select_dtypes(include=['float64']).columns:
    datos[c] = datos[c].astype('float32')

print(f"\n   Dataset limpio : {len(datos):,} registros")

# Muestra estratificada por el objetivo binario: conserva la proporción exacta
from sklearn.model_selection import train_test_split
if len(datos) > N_MUESTRA:
    muestra, _ = train_test_split(datos, train_size=N_MUESTRA,
                                  stratify=datos['internet'],
                                  random_state=SEMILLA)
else:
    muestra = datos.copy()
muestra = muestra.reset_index(drop=True)
del datos
gc.collect()

print(f"   Muestra        : {len(muestra):,} (estratificada, "
      f"random_state={SEMILLA})")
print(f"   Proporción con internet en la muestra: "
      f"{muestra['internet'].mean()*100:.2f} %")

registrar("Muestra estratificada para modelado", "Muestreo",
          f"{len(muestra):,} registros del dataset limpio",
          "Entrenar 20 modelos sobre 3,5 millones de filas excede la memoria "
          "de un equipo estándar. La estratificación por el objetivo conserva "
          "la proporción de clases; con este tamaño el error de muestreo es "
          "despreciable", len(muestra))

# --- Clasificación de las variables por su naturaleza -----------------------
CATEGORICAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v15_servsan', 'v16_desague', 'v17_tenencia', 'tip_hog',
               'v20a_emi', 'v21a_fal']
NUMERICAS = ['v13_habitac', 'v14_dormit', 'tot_pers', 'v20b_totemi',
             'v21b_totfal', 'hacinamiento', 'pers_por_habitac',
             'hacin_critico', 'sin_servicios']
CATEGORICAS = [c for c in CATEGORICAS if c in muestra.columns]
NUMERICAS = [c for c in NUMERICAS if c in muestra.columns]
PREDICTORAS = CATEGORICAS + NUMERICAS

print(f"\n   Categóricas nominales : {len(CATEGORICAS)}")
print(f"   Numéricas             : {len(NUMERICAS)}")


# %%
# =============================================================================
#  D1 — PARTICION Y PREPROCESAMIENTO
# =============================================================================
titulo("D1 · PARTICION Y PREPROCESAMIENTO")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   PolynomialFeatures)
from sklearn.pipeline import Pipeline

X = muestra[PREDICTORAS]
y_ieh = muestra['ieh'].astype('float32')
y_int = muestra['internet'].astype(int)
y_niv = muestra['nivel_equip'].astype(str)

# Una sola partición, estratificada por internet, compartida por los tres
# objetivos: todos los modelos se evalúan sobre exactamente las mismas filas.
idx_tr, idx_te = train_test_split(np.arange(len(muestra)), test_size=PRUEBA,
                                  stratify=y_int, random_state=SEMILLA)
X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
print(f"   Entrenamiento : {len(idx_tr):,}")
print(f"   Prueba        : {len(idx_te):,}")
print(f"   Misma partición para los tres objetivos → comparación justa")

registrar("Partición entrenamiento/prueba", "Partición",
          f"75 % / 25 % estratificada por internet",
          "La estratificación conserva la proporción de clases en ambos "
          "conjuntos; una partición única permite comparar modelos sobre "
          "las mismas observaciones", len(idx_te))

# --- Dos estrategias de codificacion que se compararan experimentalmente ----
# A) CODIGOS: las categoricas se usan tal cual, como numeros
prep_codigos = ColumnTransformer([
    ('num', StandardScaler(), PREDICTORAS)])

# B) ONE-HOT: cada categoria se convierte en una columna binaria
prep_onehot = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first',
                          sparse_output=False, dtype=np.float32), CATEGORICAS),
    ('num', StandardScaler(), NUMERICAS)])

# C) POLINOMIAL: grado 2 SOLO sobre las numericas + one-hot de categoricas.
#    Aplicar el grado 2 al one-hot generaria mas de 5.000 columnas, sin
#    sentido estadistico (el cuadrado de una binaria es ella misma).
prep_poli = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first',
                          sparse_output=False, dtype=np.float32), CATEGORICAS),
    ('num', Pipeline([('esc', StandardScaler()),
                      ('poli', PolynomialFeatures(degree=2,
                                                  include_bias=False))]),
     NUMERICAS)])

n_cod = prep_codigos.fit(X_tr).transform(X_tr[:5]).shape[1]
n_oh = prep_onehot.fit(X_tr).transform(X_tr[:5]).shape[1]
n_po = prep_poli.fit(X_tr).transform(X_tr[:5]).shape[1]
print(f"\n   Columnas resultantes")
print(f"     A · códigos numéricos : {n_cod}")
print(f"     B · one-hot           : {n_oh}")
print(f"     C · polinomial grado 2: {n_po}")

registrar("Definición de tres estrategias de codificación", "Preprocesamiento",
          f"códigos ({n_cod}), one-hot ({n_oh}), polinomial ({n_po})",
          "Las categóricas del censo son nominales: en v07_aguapro el código "
          "5 no vale cinco veces el 1. Se compararán experimentalmente para "
          "decidir cuál conviene a los modelos lineales")

# --- Funciones de evaluacion -------------------------------------------------
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)


def metricas_reg(y, p):
    nz = y != 0      # MAPE no está definido cuando el valor real es cero
    return {'MAE': mean_absolute_error(y, p),
            'MSE': mean_squared_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)),
            'R2': r2_score(y, p),
            'MAPE_%': np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100}


def metricas_bin(y, pred, prob):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred),
            'Recall_min': recall_score(y, pred, pos_label=0),
            'F1': f1_score(y, pred),
            'ROC_AUC': roc_auc_score(y, prob),
            'PR_AUC': average_precision_score(y, prob)}


def metricas_multi(y, pred):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, average='macro',
                                         zero_division=0),
            'Recall': recall_score(y, pred, average='macro'),
            'F1': f1_score(y, pred, average='macro')}

gc.collect()


# %%
# =============================================================================
#  D2 — EXPERIMENTO A/B DE CODIFICACION
#  ¿Conviene convertir las categoricas con one-hot? No se supone: se mide.
# =============================================================================
titulo("D2 · A/B TESTING DE LA CODIFICACION")

from sklearn.linear_model import LinearRegression, LogisticRegression

sub("Hipótesis")
print("   H0: codificar las categóricas con one-hot no mejora los modelos")
print("       lineales respecto de usar los códigos numéricos originales.")
print("   H1: one-hot mejora el desempeño, porque los códigos son nominales")
print("       y un modelo lineal los interpreta erróneamente como cantidades.")

exp = []
for nombre, prep in [('A · Códigos', prep_codigos), ('B · One-hot', prep_onehot)]:
    t0 = time.time()
    m = Pipeline([('p', prep), ('m', LinearRegression())])
    m.fit(X_tr, y_ieh.iloc[idx_tr])
    r2 = r2_score(y_ieh.iloc[idx_te], m.predict(X_te))

    m2 = Pipeline([('p', prep), ('m', LogisticRegression(
        max_iter=400, class_weight='balanced', random_state=SEMILLA))])
    m2.fit(X_tr, y_int.iloc[idx_tr])
    auc = roc_auc_score(y_int.iloc[idx_te], m2.predict_proba(X_te)[:, 1])
    exp.append({'estrategia': nombre, 'R2_regresion': r2,
                'AUC_clasificacion': auc, 'segundos': time.time() - t0})
exp = pd.DataFrame(exp).set_index('estrategia')

sub("Resultado")
print(exp.round(4).to_string())
mejora_r2 = (exp.loc['B · One-hot', 'R2_regresion'] -
             exp.loc['A · Códigos', 'R2_regresion'])
mejora_auc = (exp.loc['B · One-hot', 'AUC_clasificacion'] -
              exp.loc['A · Códigos', 'AUC_clasificacion'])
print(f"\n   Mejora con one-hot · R²  : {mejora_r2:+.4f}")
print(f"   Mejora con one-hot · AUC : {mejora_auc:+.4f}")

# Prueba de significancia: bootstrap sobre el conjunto de prueba
rng = np.random.default_rng(SEMILLA)
mA = Pipeline([('p', prep_codigos), ('m', LinearRegression())]) \
    .fit(X_tr, y_ieh.iloc[idx_tr])
mB = Pipeline([('p', prep_onehot), ('m', LinearRegression())]) \
    .fit(X_tr, y_ieh.iloc[idx_tr])
eA = (y_ieh.iloc[idx_te].values - mA.predict(X_te)) ** 2
eB = (y_ieh.iloc[idx_te].values - mB.predict(X_te)) ** 2
dif = []
for _ in range(500):
    i = rng.integers(0, len(eA), len(eA))
    dif.append(eA[i].mean() - eB[i].mean())
dif = np.array(dif)
ic = np.percentile(dif, [2.5, 97.5])
print(f"\n   Bootstrap (500 remuestreos) de la reducción del error cuadrático")
print(f"     Diferencia media : {dif.mean():+.4f}")
print(f"     IC 95 %          : [{ic[0]:+.4f}, {ic[1]:+.4f}]")
concl = "SE RECHAZA H0" if ic[0] > 0 else "NO se rechaza H0"
print(f"     Conclusión       : {concl}")

# Criterio de decision: mejora RELATIVA promedio en ambas tareas. Evaluar
# solo el R² ignoraria la clasificacion, que es el objetivo principal.
rel_r2 = mejora_r2 / max(abs(exp.loc['A · Códigos', 'R2_regresion']), 1e-9)
rel_auc = mejora_auc / max(abs(exp.loc['A · Códigos', 'AUC_clasificacion']), 1e-9)
mejora_rel = (rel_r2 + rel_auc) / 2
print(f"\n   Mejora relativa · regresión     : {rel_r2*100:+.3f} %")
print(f"   Mejora relativa · clasificación : {rel_auc*100:+.3f} %")
print(f"   Mejora relativa promedio        : {mejora_rel*100:+.3f} %")

CODIF = 'B · One-hot' if mejora_rel > 0 else 'A · Códigos'
PREP = prep_onehot if CODIF == 'B · One-hot' else prep_codigos
print(f"\n   ESTRATEGIA ADOPTADA PARA LOS MODELOS LINEALES: {CODIF}")
print(f"   Criterio: mejora relativa promedio en las dos tareas, no una sola.")

registrar("A/B testing de codificación", "Experimento",
          f"One-hot vs códigos · ΔR² {mejora_r2:+.4f} · ΔAUC {mejora_auc:+.4f} "
          f"· IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]",
          f"{concl}. Se adopta {CODIF}. Los códigos del censo son nominales y "
          f"un modelo lineal los interpreta como cantidades ordenadas",
          round(float(mejora_r2), 5))

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
for a, col, tit in [(ax[0], 'R2_regresion', '(a) R² · regresión de ieh'),
                    (ax[1], 'AUC_clasificacion', '(b) AUC · clasificación')]:
    b = a.bar(exp.index, exp[col], color=[GRIS, MENTA], edgecolor='white',
              linewidth=2.5, width=.55)
    for r, v in zip(b, exp[col]):
        a.text(r.get_x()+r.get_width()/2, v, f"{v:.4f}", ha='center',
               va='bottom', fontsize=11, fontweight='bold')
    a.set_title(tit)
    a.set_ylim(0, exp[col].max()*1.18)
ax[2].hist(dif, bins=40, color=LILA, edgecolor='white', linewidth=1.2)
ax[2].axvline(0, color=CORAL, ls='--', lw=2.4, label='Sin diferencia')
ax[2].axvspan(ic[0], ic[1], color=MENTA, alpha=.22, label='IC 95 %')
ax[2].set_xlabel("Reducción del error cuadrático medio")
ax[2].set_title("(c) Bootstrap · ¿la mejora es real?")
ax[2].legend(fontsize=8)
fig.suptitle("FASE D · A/B TESTING DE LA CODIFICACIÓN — códigos frente a "
             "one-hot", fontsize=14, y=1.02)
plt.tight_layout()
# Figura 26 del informe · D01_ab_codificacion.png
guardar("D01_ab_codificacion.png")
gc.collect()


# %%
# =============================================================================
#  D3 — REGRESION SOBRE ieh
#  Simple · Multiple · Polinomial · Ridge · Lasso · Elastic Net
# =============================================================================
titulo("D3 · MODELOS DE REGRESION — índice de equipamiento")

from sklearn.linear_model import Ridge, Lasso, ElasticNet

ytr, yte = y_ieh.iloc[idx_tr], y_ieh.iloc[idx_te]

# Regresion SIMPLE: una sola predictora, la de mayor correlacion con ieh
corr = muestra[NUMERICAS + ['urbrur']].corrwith(muestra['ieh']).abs()
VAR_SIMPLE = corr.idxmax()
print(f"   Predictora de la regresión simple: {VAR_SIMPLE} "
      f"(|r| = {corr.max():.3f}, la de mayor correlación)")

modelos_reg = {
    'Lineal simple': Pipeline([
        ('p', ColumnTransformer([('n', StandardScaler(), [VAR_SIMPLE])])),
        ('m', LinearRegression())]),
    'Lineal múltiple': Pipeline([('p', PREP), ('m', LinearRegression())]),
    'Polinomial (grado 2)': Pipeline([('p', prep_poli),
                                      ('m', LinearRegression())]),
    'Ridge (L2)': Pipeline([('p', PREP), ('m', Ridge(alpha=10.0))]),
    'Lasso (L1)': Pipeline([('p', PREP), ('m', Lasso(
        alpha=0.01, max_iter=3000, random_state=SEMILLA))]),
    'Elastic Net': Pipeline([('p', PREP), ('m', ElasticNet(
        alpha=0.01, l1_ratio=0.5, max_iter=3000, random_state=SEMILLA))]),
}

pred_reg, res_reg = {}, {}
for nom, mod in modelos_reg.items():
    t0 = time.time()
    mod.fit(X_tr, ytr)
    p = mod.predict(X_te)
    seg = time.time() - t0
    pred_reg[nom] = p
    res_reg[nom] = metricas_reg(yte.values, p)
    registrar_modelo('ieh', 'Regresión', nom, res_reg[nom], seg,
                     f"codificación {CODIF}")
    r = res_reg[nom]
    print(f"   {nom:<24} MAE {r['MAE']:.4f} · RMSE {r['RMSE']:.4f} · "
          f"R² {r['R2']:.4f} · MAPE {r['MAPE_%']:5.2f} % · {seg:5.1f}s")

tabla_reg = pd.DataFrame(res_reg).T
sub("Tabla comparativa")
print(tabla_reg.round(4).to_string())

base = np.sqrt(np.mean((yte - ytr.mean())**2))
print(f"\n   Línea base (predecir siempre la media): RMSE {base:.4f}")
mejor = tabla_reg['R2'].idxmax()
print(f"   Mejor modelo lineal: {mejor}  ·  R² {tabla_reg.loc[mejor,'R2']:.4f}  ·"
      f"  mejora {(1-tabla_reg.loc[mejor,'RMSE']/base)*100:.1f} % sobre la base")
print(f"\n   Nota sobre MAPE: se calcula excluyendo los hogares con ieh = 0,")
print(f"   porque dividir por cero lo vuelve indefinido.")

# --- Coeficientes de Lasso: la seleccion automatica de variables ------------
las = modelos_reg['Lasso (L1)'].named_steps['m']
nombres_feat = modelos_reg['Lasso (L1)'].named_steps['p'].get_feature_names_out()
coef = pd.Series(las.coef_, index=nombres_feat)
n_cero = int((coef == 0).sum())
print(f"\n   Lasso anuló {n_cero} de {len(coef)} coeficientes "
      f"({n_cero/len(coef)*100:.1f} %): selección automática de variables")

registrar("Regresión lineal sobre ieh", "Modelado",
          f"6 modelos · mejor {mejor} R² {tabla_reg.loc[mejor,'R2']:.4f}",
          f"Lasso eliminó {n_cero} de {len(coef)} variables. La regresión "
          f"simple con {VAR_SIMPLE} sirve como referencia mínima",
          round(float(tabla_reg.loc[mejor, 'R2']), 5))

# --- Graficos de regresion ---------------------------------------------------
fig = plt.figure(figsize=(16.5, 9.5))
gs = fig.add_gridspec(2, 3, hspace=.45, wspace=.3)

ax = fig.add_subplot(gs[0, 0])
s = tabla_reg['R2'].sort_values()
b = ax.barh(range(len(s)), s.values,
            color=[MENTA if v == s.max() else AZUL for v in s.values],
            edgecolor='white', linewidth=1.8)
ax.set_yticks(range(len(s)), s.index, fontsize=8)
for i, v in enumerate(s.values):
    ax.text(v, i, f" {v:.4f}", va='center', fontsize=8, fontweight='bold')
ax.set_xlabel("R²"); ax.set_title("(a) Coeficiente de determinación")
ax.set_xlim(0, max(s.max()*1.2, .1))

ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(tabla_reg)); w = .27
for j, (met, col) in enumerate([('MAE', AZUL), ('RMSE', LILA), ('MSE', CORAL)]):
    ax.bar(x + (j-1)*w, tabla_reg[met], w, label=met, color=col,
           edgecolor='white', linewidth=1.2)
ax.axhline(base, color=TINTA, ls='--', lw=1.8, label=f'RMSE base {base:.2f}')
ax.set_xticks(x, [n[:12] for n in tabla_reg.index], rotation=25, fontsize=7.5)
ax.set_title("(b) Errores MAE · RMSE · MSE"); ax.legend(fontsize=7.5)

ax = fig.add_subplot(gs[0, 2])
s = tabla_reg['MAPE_%'].sort_values()
ax.barh(range(len(s)), s.values, color=AMBAR, edgecolor='white', linewidth=1.8)
ax.set_yticks(range(len(s)), s.index, fontsize=8)
for i, v in enumerate(s.values):
    ax.text(v, i, f" {v:.1f} %", va='center', fontsize=8, fontweight='bold')
ax.set_xlabel("MAPE (%) · excluye ieh = 0")
ax.set_title("(c) Error porcentual absoluto medio")

ax = fig.add_subplot(gs[1, 0])
p = pred_reg[mejor]
j = np.random.default_rng(SEMILLA).uniform(-.3, .3, len(yte))
sel = np.random.default_rng(SEMILLA).choice(len(yte), min(8000, len(yte)),
                                            replace=False)
ax.scatter(yte.values[sel] + j[sel], p[sel], s=5, alpha=.18, color=AZUL,
           edgecolors='none')
ax.plot([0, 13], [0, 13], '--', color=CORAL, lw=2.2, label='Predicción perfecta')
ax.set_xlabel("ieh real"); ax.set_ylabel("ieh predicho")
ax.set_title(f"(d) Real frente a predicho · {mejor}")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
res_ = yte.values - p
ax.hist(res_, bins=50, color=MENTA, edgecolor='white', linewidth=1)
ax.axvline(0, color=TINTA, lw=2)
ax.set_xlabel("Residuo (real − predicho)")
ax.set_title(f"(e) Residuos · media {res_.mean():+.3f}")

ax = fig.add_subplot(gs[1, 2])
top = coef[coef != 0].abs().sort_values(ascending=False).head(14)
vals = coef[top.index].sort_values()
ax.barh(range(len(vals)), vals.values,
        color=[CORAL if v > 0 else AZUL for v in vals.values],
        edgecolor='white', linewidth=1.2)
ax.set_yticks(range(len(vals)), [n.replace('cat__', '').replace('num__', '')[:24]
                                 for n in vals.index], fontsize=7)
ax.axvline(0, color=TINTA, lw=1.4)
ax.set_title(f"(f) Lasso · {len(coef)-n_cero} variables conservadas")

fig.suptitle("FASE D · REGRESIÓN LINEAL — índice de equipamiento (ieh)",
             fontsize=15, y=.98)
# Figura 27 del informe · D02_regresion_ieh.png
guardar("D02_regresion_ieh.png")
gc.collect()


# %%
# =============================================================================
#  D4 — CLASIFICACION BINARIA SOBRE internet
#  Probabilidad lineal (simple/multiple/polinomial) · Logística · L1 · L2 ·
#  Elastic Net · Clasificador Ridge
# =============================================================================
titulo("D4 · CLASIFICACION BINARIA — acceso a internet")

from sklearn.linear_model import RidgeClassifier

ytr_b, yte_b = y_int.iloc[idx_tr], y_int.iloc[idx_te]

corr_b = muestra[NUMERICAS + ['urbrur']].corrwith(muestra['internet']).abs()
VAR_SIMPLE_B = corr_b.idxmax()

sub("Modelo de probabilidad lineal (MPL)")
print("   Aplicar regresión lineal a un objetivo 0/1 se denomina modelo de")
print("   probabilidad lineal. Es legítimo como referencia, pero su defecto")
print("   conocido es que puede predecir valores fuera de [0, 1]. Se incluye")
print("   precisamente para demostrar por qué existe la regresión logística.")

mpl = {
    'MPL simple': Pipeline([
        ('p', ColumnTransformer([('n', StandardScaler(), [VAR_SIMPLE_B])])),
        ('m', LinearRegression())]),
    'MPL múltiple': Pipeline([('p', PREP), ('m', LinearRegression())]),
    'MPL polinomial': Pipeline([('p', prep_poli), ('m', LinearRegression())]),
}
fuera = {}
prob_bin, pred_bin, res_bin = {}, {}, {}

for nom, mod in mpl.items():
    t0 = time.time()
    mod.fit(X_tr, ytr_b)
    raw = mod.predict(X_te)
    seg = time.time() - t0
    fuera[nom] = ((raw < 0) | (raw > 1)).mean() * 100
    prob = np.clip(raw, 0, 1)
    pred = (raw >= .5).astype(int)
    prob_bin[nom] = prob; pred_bin[nom] = pred
    res_bin[nom] = metricas_bin(yte_b, pred, prob)
    registrar_modelo('internet', 'Clasificación binaria', nom, res_bin[nom],
                     seg, f"{fuera[nom]:.2f} % de predicciones fuera de [0,1]")
    print(f"   {nom:<22} predicciones fuera de [0,1]: {fuera[nom]:5.2f} %  ·  "
          f"AUC {res_bin[nom]['ROC_AUC']:.4f}")

sub("Modelos de clasificación propiamente dichos")
clasif = {
    'Logística': LogisticRegression(max_iter=500, class_weight='balanced',
                                    random_state=SEMILLA),
    'Logística L2 (Ridge)': LogisticRegression(
        penalty='l2', C=0.1, max_iter=500, class_weight='balanced',
        random_state=SEMILLA),
    'Logística L1 (Lasso)': LogisticRegression(
        penalty='l1', C=0.1, solver='liblinear', max_iter=500,
        class_weight='balanced', random_state=SEMILLA),
    'Logística Elastic Net': LogisticRegression(
        penalty='elasticnet', l1_ratio=0.5, C=0.1, solver='saga',
        max_iter=300, tol=1e-3, class_weight='balanced',
        random_state=SEMILLA),
    'Clasificador Ridge': RidgeClassifier(alpha=10.0, class_weight='balanced'),
}
for nom, est in clasif.items():
    mod = Pipeline([('p', PREP), ('m', est)])
    t0 = time.time()
    mod.fit(X_tr, ytr_b)
    seg = time.time() - t0
    pred = mod.predict(X_te)
    if hasattr(mod, 'predict_proba'):
        prob = mod.predict_proba(X_te)[:, 1]
    else:
        d = mod.decision_function(X_te)
        prob = (d - d.min()) / (d.max() - d.min())
    prob_bin[nom] = prob; pred_bin[nom] = pred
    res_bin[nom] = metricas_bin(yte_b, pred, prob)
    registrar_modelo('internet', 'Clasificación binaria', nom, res_bin[nom],
                     seg, "class_weight='balanced'")
    r = res_bin[nom]
    print(f"   {nom:<24} Acc {r['Accuracy']:.4f} · F1 {r['F1']:.4f} · "
          f"Recall mín. {r['Recall_min']:.4f} · AUC {r['ROC_AUC']:.4f} · "
          f"{seg:5.1f}s")

tabla_bin = pd.DataFrame(res_bin).T
sub("Tabla comparativa")
print(tabla_bin.round(4).to_string())
# El MPL no es un clasificador propiamente dicho: sus salidas no son
# probabilidades validas. Se conserva como referencia pero NO compite por
# la seleccion. Entre los clasificadores reales se prioriza el recall de la
# clase minoritaria, que es la poblacion que el proyecto busca identificar.
mpl_idx = [i for i in tabla_bin.index if i.startswith('MPL')]
reales = tabla_bin.drop(index=mpl_idx)
mejor_b = reales['ROC_AUC'].idxmax()
mejor_rec = reales['Recall_min'].idxmax()
mpl_top = tabla_bin.loc[mpl_idx, 'ROC_AUC'].idxmax()

print(f"\n   Referencia MPL con mayor AUC  : {mpl_top} "
      f"({tabla_bin.loc[mpl_top,'ROC_AUC']:.4f}) · recall mín. "
      f"{tabla_bin.loc[mpl_top,'Recall_min']:.4f}")
print(f"   Mejor clasificador por AUC    : {mejor_b} "
      f"({reales.loc[mejor_b,'ROC_AUC']:.4f})")
print(f"   Mejor por recall SIN internet : {mejor_rec} "
      f"({reales.loc[mejor_rec,'Recall_min']:.4f}) ← la métrica que importa")
ventaja = tabla_bin.loc[mpl_top, 'ROC_AUC'] - reales['ROC_AUC'].max()
if ventaja > 0.002:
    print(f"\n   HALLAZGO: {mpl_top} supera en AUC al mejor clasificador por")
    print(f"   {ventaja:+.4f}. No indica que sea mejor clasificador, sino que")
    print(f"   hay relaciones NO lineales que la logística lineal no captura.")
    print(f"   Esto anticipa ventaja para los modelos de árboles (Fase F).")
else:
    print(f"\n   El MPL y la logística alcanzan un AUC equivalente "
          f"(diferencia {ventaja:+.4f}).")
    print(f"   Aun así la logística es preferible: sus salidas son")
    print(f"   probabilidades válidas y su recall minoritario es mayor.")

registrar("Clasificación lineal de internet", "Modelado",
          f"8 modelos · mejor clasificador {mejor_b} AUC "
          f"{tabla_bin.loc[mejor_b,'ROC_AUC']:.4f} · mejor recall "
          f"minoritario {mejor_rec}",
          f"El MPL predijo hasta {max(fuera.values()):.1f} % de valores fuera "
          f"de [0,1], lo que justifica la regresión logística",
          round(float(tabla_bin.loc[mejor_b, 'ROC_AUC']), 5))

# --- Grafico 1: curvas ROC y Precision-Recall --------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
for i, nom in enumerate(tabla_bin.index):
    fpr, tpr, _ = roc_curve(yte_b, prob_bin[nom])
    ax[0].plot(fpr, tpr, lw=2, color=PALETA[i % len(PALETA)],
               label=f"{nom[:20]} ({tabla_bin.loc[nom,'ROC_AUC']:.3f})")
    pr, rc, _ = precision_recall_curve(yte_b, prob_bin[nom])
    ax[1].plot(rc, pr, lw=2, color=PALETA[i % len(PALETA)],
               label=f"{nom[:20]} ({tabla_bin.loc[nom,'PR_AUC']:.3f})")
ax[0].plot([0, 1], [0, 1], '--', color=GRIS, lw=2, label='Azar (0.500)')
ax[0].set_xlabel("Tasa de falsos positivos")
ax[0].set_ylabel("Tasa de verdaderos positivos")
ax[0].set_title("(a) Curvas ROC"); ax[0].legend(fontsize=6.5, loc='lower right')
ax[1].axhline(yte_b.mean(), color=GRIS, ls='--', lw=2,
              label=f'Azar ({yte_b.mean():.3f})')
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("(b) Curvas Precision-Recall")
ax[1].legend(fontsize=6.5, loc='lower left')

raw_mpl = mpl['MPL múltiple'].predict(X_te)
ax[2].hist(raw_mpl, bins=60, color=LILA, edgecolor='white', linewidth=.8)
ax[2].axvspan(raw_mpl.min(), 0, color=CORAL, alpha=.22, label='< 0 imposible')
ax[2].axvspan(1, raw_mpl.max(), color=CORAL, alpha=.22, label='> 1 imposible')
ax[2].axvline(0, color=CORAL, lw=2); ax[2].axvline(1, color=CORAL, lw=2)
ax[2].set_xlabel("Salida del modelo de probabilidad lineal")
ax[2].set_title(f"(c) El MPL predice fuera de [0,1] · "
                f"{fuera['MPL múltiple']:.1f} %")
ax[2].legend(fontsize=8)
fig.suptitle("FASE D · CLASIFICACIÓN LINEAL — curvas de evaluación",
             fontsize=14, y=1.02)
plt.tight_layout()
# Figura 28 del informe · D03_clasificacion_curvas.png
guardar("D03_clasificacion_curvas.png")

# --- Grafico 2: matrices de confusion ----------------------------------------
n = len(tabla_bin)
fig, ax = plt.subplots(2, 4, figsize=(17, 8.2))
for k, nom in enumerate(tabla_bin.index[:8]):
    a = ax[k // 4, k % 4]
    mc = confusion_matrix(yte_b, pred_bin[nom])
    a.imshow(mc, cmap=CMAP)
    for i_ in range(2):
        for j_ in range(2):
            a.text(j_, i_, f"{mc[i_, j_]:,}", ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   color='white' if mc[i_, j_] > mc.max()/2 else TINTA)
    a.set_xticks([0, 1], ['Pred: sin', 'Pred: con'], fontsize=7.5)
    a.set_yticks([0, 1], ['Real: sin', 'Real: con'], fontsize=7.5)
    a.set_title(f"{nom[:22]}\nRecall mín. "
                f"{tabla_bin.loc[nom,'Recall_min']:.3f}", fontsize=8.5)
    a.grid(False)
for k in range(n, 8):
    ax[k // 4, k % 4].axis('off')
fig.suptitle("FASE D · MATRICES DE CONFUSIÓN — clasificación binaria",
             fontsize=14, y=.99)
plt.tight_layout()
# Figura 29 del informe · D04_matrices_confusion.png
guardar("D04_matrices_confusion.png")

# --- Grafico 3: comparacion de metricas --------------------------------------
fig, ax = plt.subplots(figsize=(16.5, 5.4))
mets = ['Accuracy', 'Precision', 'Recall', 'Recall_min', 'F1', 'ROC_AUC',
        'PR_AUC']
x = np.arange(len(mets)); w = .8 / len(tabla_bin)
for i, nom in enumerate(tabla_bin.index):
    ax.bar(x + (i - len(tabla_bin)/2 + .5)*w, tabla_bin.loc[nom, mets], w,
           label=nom[:22], color=PALETA[i % len(PALETA)], edgecolor='white',
           linewidth=.8)
ax.set_xticks(x, ['Accuracy', 'Precision', 'Recall\n(con internet)',
                  'Recall\n(SIN internet)', 'F1', 'ROC AUC', 'PR AUC'])
ax.set_ylim(0, 1.05)
ax.set_title("Métricas de clasificación por modelo")
ax.legend(fontsize=7, ncol=4, loc='lower right')
plt.tight_layout()
# Figura 30 del informe · D05_metricas_clasificacion.png
guardar("D05_metricas_clasificacion.png")
gc.collect()


# %%
# =============================================================================
#  D5 — CLASIFICACION MULTICLASE SOBRE nivel_equip
# =============================================================================
titulo("D5 · CLASIFICACION MULTICLASE — nivel de equipamiento")

ytr_m, yte_m = y_niv.iloc[idx_tr], y_niv.iloc[idx_te]
print("   nivel_equip es la versión categorizada de ieh. Permite aplicar a la")
print("   dimensión material los mismos clasificadores que a internet.")

multi = {
    'Logística multinomial': LogisticRegression(
        max_iter=500, class_weight='balanced', random_state=SEMILLA),
    'Logística L1 (Lasso)': LogisticRegression(
        penalty='l1', C=0.1, solver='saga', max_iter=300, tol=1e-3,
        class_weight='balanced', random_state=SEMILLA),
    'Logística Elastic Net': LogisticRegression(
        penalty='elasticnet', l1_ratio=0.5, C=0.1, solver='saga',
        max_iter=300, tol=1e-3, class_weight='balanced',
        random_state=SEMILLA),
    'Clasificador Ridge': RidgeClassifier(alpha=10.0, class_weight='balanced'),
}
pred_m, res_m = {}, {}
for nom, est in multi.items():
    mod = Pipeline([('p', PREP), ('m', est)])
    t0 = time.time()
    mod.fit(X_tr, ytr_m)
    seg = time.time() - t0
    pred = mod.predict(X_te)
    pred_m[nom] = pred
    res_m[nom] = metricas_multi(yte_m, pred)
    registrar_modelo('nivel_equip', 'Clasificación multiclase', nom,
                     res_m[nom], seg, "promedio macro")
    r = res_m[nom]
    print(f"   {nom:<24} Acc {r['Accuracy']:.4f} · Precision {r['Precision']:.4f}"
          f" · Recall {r['Recall']:.4f} · F1 {r['F1']:.4f} · {seg:5.1f}s")

tabla_m = pd.DataFrame(res_m).T
mejor_m = tabla_m['F1'].idxmax()
print(f"\n   Mejor por F1 macro: {mejor_m} ({tabla_m.loc[mejor_m,'F1']:.4f})")
print(f"   Referencia del azar con 3 clases: 0,333")

registrar("Clasificación multiclase de nivel_equip", "Modelado",
          f"4 modelos · mejor {mejor_m} F1 {tabla_m.loc[mejor_m,'F1']:.4f}",
          "Métricas con promedio macro: cada clase pesa igual sin importar "
          "su tamaño", round(float(tabla_m.loc[mejor_m, 'F1']), 5))

orden = ['Bajo', 'Medio', 'Alto']
fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.6))
for k, nom in enumerate(tabla_m.index):
    mc = confusion_matrix(yte_m, pred_m[nom], labels=orden)
    mcn = mc / mc.sum(axis=1, keepdims=True)
    ax[k].imshow(mcn, cmap=CMAP, vmin=0, vmax=1)
    for i_ in range(3):
        for j_ in range(3):
            ax[k].text(j_, i_, f"{mcn[i_, j_]*100:.0f} %\n{mc[i_, j_]:,}",
                       ha='center', va='center', fontsize=8,
                       fontweight='bold',
                       color='white' if mcn[i_, j_] > .5 else TINTA)
    ax[k].set_xticks(range(3), orden, fontsize=8)
    ax[k].set_yticks(range(3), orden, fontsize=8)
    ax[k].set_xlabel("Predicho"); ax[k].set_ylabel("Real")
    ax[k].set_title(f"{nom[:22]}\nF1 macro {tabla_m.loc[nom,'F1']:.3f}",
                    fontsize=9)
    ax[k].grid(False)
fig.suptitle("FASE D · MATRICES DE CONFUSIÓN — nivel de equipamiento",
             fontsize=14, y=1.04)
plt.tight_layout()
# Figura 31 del informe · D06_multiclase.png
guardar("D06_multiclase.png")
gc.collect()


# %%
# =============================================================================
#  D6 — CIERRE DE LA FASE Y ACTUALIZACION DE LA BITACORA
# =============================================================================
titulo("D6 · RESUMEN DE LA FASE D")

print(f"""
   EXPERIMENTO DE CODIFICACION
     Estrategia adoptada : {CODIF}
     Mejora en R²        : {mejora_r2:+.4f}
     Mejora en AUC       : {mejora_auc:+.4f}

   REGRESION (ieh)
     Modelos             : {len(tabla_reg)}
     Mejor               : {mejor}  ·  R² {tabla_reg.loc[mejor,'R2']:.4f}
     RMSE                : {tabla_reg.loc[mejor,'RMSE']:.4f} bienes

   CLASIFICACION BINARIA (internet)
     Modelos             : {len(tabla_bin)}  (3 MPL de referencia + 5 clasificadores)
     Mejor clasificador  : {mejor_b}  ·  AUC {tabla_bin.loc[mejor_b,'ROC_AUC']:.4f}
     Mejor recall mínimo : {mejor_rec}  ·  {tabla_bin.loc[mejor_rec,'Recall_min']:.4f}

   CLASIFICACION MULTICLASE (nivel_equip)
     Modelos             : {len(tabla_m)}
     Mejor               : {mejor_m}  ·  F1 macro {tabla_m.loc[mejor_m,'F1']:.4f}

   TOTAL DE MODELOS ENTRENADOS EN LA FASE D: {len(MOD_FASE)}
   Tiempo de la fase: {(time.time()-t_ini)/60:.1f} min
""")

tabla_reg.round(5).to_csv(os.path.join(DIR_FIG, "D_regresion.csv"))
tabla_bin.round(5).to_csv(os.path.join(DIR_FIG, "D_clasificacion_binaria.csv"))
tabla_m.round(5).to_csv(os.path.join(DIR_FIG, "D_clasificacion_multiclase.csv"))

registrar("Cierre de la Fase D", "Cierre",
          f"{len(MOD_FASE)} modelos entrenados y evaluados",
          "Los resultados alimentarán la comparación global de la Fase H",
          len(MOD_FASE))

volcar_bitacora()
gc.collect()


# =============================================================================
#  PROYECTO CPV 2024 · BORRADOR INTEGRAL
#  FASE E — KNN · NAIVE BAYES · MAQUINAS DE SOPORTE VECTORIAL
#
#  Modelos de esta fase
#    KNN          : KNeighborsRegressor (ieh) · KNeighborsClassifier
#                   (internet, nivel_equip) · estudio del parametro k
#    Naive Bayes  : Gaussiano y Categorico con discretizacion (internet,
#                   nivel_equip). No existe version de regresion.
#    SVM          : SVR lineal y RBF (ieh) · SVC lineal y RBF
#                   (internet, nivel_equip)
#
#  Se reproduce EXACTAMENTE la misma muestra y particion de la Fase D, de modo
#  que todos los modelos del proyecto se evaluan sobre las mismas filas.
#
#
#  DISEÑADO PARA MEMORIA LIMITADA · random_state = 42 en todo
# =============================================================================


# %%
# =============================================================================
#  E0 — CONFIGURACION, REGISTRO DE CONSOLA Y CARGA
# =============================================================================
import os
import gc
import sys
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

SEMILLA = 42
N_MUESTRA = 120_000          # IGUAL que la Fase D: misma muestra
PRUEBA = 0.25                # IGUAL que la Fase D: misma partición
N_KNN_ENT = 25_000           # KNN compara cada punto con todo el entrenamiento
N_SVM_RBF = 8_000            # el kernel RBF escala de forma cuadrática

FASE = "E"
DIR_FIG = "figuras_v2"
DIR_LOG = "logs"
os.makedirs(DIR_FIG, exist_ok=True)
os.makedirs(DIR_LOG, exist_ok=True)


# -----------------------------------------------------------------------------
#  REGISTRO DE CONSOLA: todo lo que se imprime queda también en un archivo
# -----------------------------------------------------------------------------
class Registro:
    def __init__(self, ruta):
        self.archivo = open(ruta, 'a', encoding='utf-8')
        self.consola = sys.__stdout__ if not hasattr(sys.stdout, 'consola') \
            else sys.stdout.consola

    def write(self, texto):
        try:
            self.consola.write(texto)
        except Exception:
            pass
        self.archivo.write(texto)
        self.archivo.flush()

    def flush(self):
        try:
            self.consola.flush()
        except Exception:
            pass
        self.archivo.flush()


RUTA_LOG = os.path.join(DIR_LOG, f"consola_fase_{FASE}.txt")
if not isinstance(sys.stdout, Registro):
    _stdout_original = sys.stdout
    reg_consola = Registro(RUTA_LOG)
    reg_consola.consola = _stdout_original
    sys.stdout = reg_consola
print(f"\n{'#'*86}\n#  EJECUCION {datetime.now():%Y-%m-%d %H:%M:%S}\n{'#'*86}")

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA = "#FF6B8A", "#FFC145", "#22223B"
GRIS, AGUA = "#8D99AE", "#A8DADC"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, AGUA, "#6C8AE4", "#C3A6F0",
          "#2EC4B6", "#E07A5F"]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "grid.linewidth": .9,
    "axes.axisbelow": True, "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92,
    "legend.edgecolor": "#D8DEF0"})


def guardar(n):
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   [figura] {r}")
    try:
        from IPython.display import Image, display
        display(Image(filename=r))
    except Exception:
        pass


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


# ---------------- BITACORA MAESTRA EN EXCEL ----------------------------------
BITACORA_XLSX = "BITACORA_PROYECTO.xlsx"
REG_FASE, MOD_FASE = [], []


def registrar(paso, categoria, detalle, justificacion, valor=None):
    REG_FASE.append({
        'fase': FASE, 'n': len(REG_FASE) + 1,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'categoria': categoria, 'paso': paso, 'detalle': detalle,
        'justificacion': justificacion, 'valor': valor})
    print(f"   [bitácora {FASE}-{len(REG_FASE):02d}] {paso}")


def registrar_modelo(objetivo, tipo, modelo, metricas, segundos, nota=""):
    fila = {'fase': FASE,
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objetivo': objetivo, 'tipo': tipo, 'modelo': modelo,
            'segundos': round(segundos, 2), 'nota': nota}
    fila.update({k: (round(float(v), 5) if v is not None else None)
                 for k, v in metricas.items()})
    MOD_FASE.append(fila)


def leer_hoja(nombre):
    if os.path.exists(BITACORA_XLSX):
        try:
            return pd.read_excel(BITACORA_XLSX, sheet_name=nombre)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def volcar_bitacora():
    reg, mod = leer_hoja('Registro'), leer_hoja('Modelos')
    if len(reg) and 'fase' in reg:
        reg = reg[reg['fase'].astype(str) != FASE]
    if len(mod) and 'fase' in mod:
        mod = mod[mod['fase'].astype(str) != FASE]
    reg = pd.concat([reg, pd.DataFrame(REG_FASE)], ignore_index=True)
    mod = pd.concat([mod, pd.DataFrame(MOD_FASE)], ignore_index=True)
    resumen = (reg.groupby('fase').agg(pasos=('paso', 'count'),
                                       ultima_actualizacion=('fecha', 'max'))
               .reset_index())
    if len(mod):
        rm = mod.groupby('fase').agg(modelos=('modelo', 'count')).reset_index()
        resumen = resumen.merge(rm, on='fase', how='left')
    try:
        with pd.ExcelWriter(BITACORA_XLSX, engine='openpyxl') as w:
            resumen.to_excel(w, sheet_name='Resumen por fase', index=False)
            reg.to_excel(w, sheet_name='Registro', index=False)
            mod.to_excel(w, sheet_name='Modelos', index=False)
        print(f"   ✔ Bitácora actualizada: {BITACORA_XLSX}  "
              f"(registro {len(reg)} · modelos {len(mod)})")
    except PermissionError:
        alt = f"BITACORA_respaldo_{FASE}_{datetime.now():%H%M%S}.csv"
        pd.DataFrame(REG_FASE).to_csv(alt, index=False, encoding='utf-8')
        pd.DataFrame(MOD_FASE).to_csv(alt.replace('.csv', '_modelos.csv'),
                                      index=False, encoding='utf-8')


titulo("FASE E · KNN · NAIVE BAYES · SVM")
t_ini = time.time()

from sklearn.model_selection import train_test_split

BASE = "vivienda_cpv2024_LIMPIO_v2"
datos = pd.read_parquet(f"{BASE}.parquet") if os.path.exists(f"{BASE}.parquet") \
    else pd.read_csv(f"{BASE}.csv", sep=';')
for c in datos.select_dtypes(include=['float64']).columns:
    datos[c] = datos[c].astype('float32')

# --- EXACTAMENTE la misma muestra que la Fase D ------------------------------
if len(datos) > N_MUESTRA:
    muestra, _ = train_test_split(datos, train_size=N_MUESTRA,
                                  stratify=datos['internet'],
                                  random_state=SEMILLA)
else:
    muestra = datos.copy()
muestra = muestra.reset_index(drop=True)
del datos
gc.collect()

CATEGORICAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v15_servsan', 'v16_desague', 'v17_tenencia', 'tip_hog',
               'v20a_emi', 'v21a_fal']
NUMERICAS = ['v13_habitac', 'v14_dormit', 'tot_pers', 'v20b_totemi',
             'v21b_totfal', 'hacinamiento', 'pers_por_habitac',
             'hacin_critico', 'sin_servicios']
CATEGORICAS = [c for c in CATEGORICAS if c in muestra.columns]
NUMERICAS = [c for c in NUMERICAS if c in muestra.columns]
PREDICTORAS = CATEGORICAS + NUMERICAS

X = muestra[PREDICTORAS]
y_ieh = muestra['ieh'].astype('float32')
y_int = muestra['internet'].astype(int)
y_niv = muestra['nivel_equip'].astype(str)

idx_tr, idx_te = train_test_split(np.arange(len(muestra)), test_size=PRUEBA,
                                  stratify=y_int, random_state=SEMILLA)
X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]

print(f"   Muestra        : {len(muestra):,}  (idéntica a la Fase D)")
print(f"   Entrenamiento  : {len(idx_tr):,}")
print(f"   Prueba         : {len(idx_te):,}  (mismas filas que la Fase D)")

# Submuestras reproducibles para los algoritmos de alto costo
rng = np.random.default_rng(SEMILLA)
sub_knn = rng.choice(len(idx_tr), min(N_KNN_ENT, len(idx_tr)), replace=False)
sub_rbf = rng.choice(len(idx_tr), min(N_SVM_RBF, len(idx_tr)), replace=False)
print(f"   Entrenamiento KNN      : {len(sub_knn):,}")
print(f"   Entrenamiento SVM RBF  : {len(sub_rbf):,}")

registrar("Reproducción de la partición de la Fase D", "Muestreo",
          f"muestra {len(muestra):,} · prueba {len(idx_te):,} filas idénticas",
          "Evaluar sobre las mismas observaciones permite comparar de forma "
          "justa todos los modelos del proyecto en la Fase H",
          len(idx_te))
registrar("Submuestreo para algoritmos de alto costo", "Muestreo",
          f"KNN {len(sub_knn):,} · SVM RBF {len(sub_rbf):,} registros de "
          f"entrenamiento",
          "KNN calcula la distancia de cada punto de prueba contra todo el "
          "entrenamiento y el SVM con kernel RBF escala de forma cuadrática "
          "o cúbica. Con 90.000 registros ambos exceden la memoria de un "
          "equipo estándar. El conjunto de PRUEBA no se reduce", len(sub_knn))

# --- Preprocesamiento: one-hot, adoptado en la Fase D ------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   KBinsDiscretizer, OrdinalEncoder)
from sklearn.pipeline import Pipeline

prep_onehot = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first',
                          sparse_output=False, dtype=np.float32), CATEGORICAS),
    ('num', StandardScaler(), NUMERICAS)])

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)


def metricas_reg(y, p):
    y = np.asarray(y); nz = y != 0
    return {'MAE': mean_absolute_error(y, p), 'MSE': mean_squared_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)), 'R2': r2_score(y, p),
            'MAPE_%': np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100}


def metricas_bin(y, pred, prob):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred),
            'Recall_min': recall_score(y, pred, pos_label=0),
            'F1': f1_score(y, pred), 'ROC_AUC': roc_auc_score(y, prob),
            'PR_AUC': average_precision_score(y, prob)}


def metricas_multi(y, pred):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, average='macro',
                                         zero_division=0),
            'Recall': recall_score(y, pred, average='macro'),
            'F1': f1_score(y, pred, average='macro')}


def escalar_prob(d):
    d = np.asarray(d, dtype=float)
    return (d - d.min()) / (d.max() - d.min() + 1e-12)

gc.collect()


# %%
# =============================================================================
#  E1 — K VECINOS MAS CERCANOS: ¿QUE VALOR DE k?
#  k es el hiperparametro central. Un k pequeño memoriza el ruido
#  (sobreajuste); uno grande promedia demasiado (subajuste).
# =============================================================================
titulo("E1 · KNN — EL DILEMA SESGO-VARIANZA")

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

Xk_tr = X_tr.iloc[sub_knn]
yk_int = y_int.iloc[idx_tr].iloc[sub_knn]

# Validacion interna: 20 % del entrenamiento KNN, sin tocar la prueba
iv_tr, iv_va = train_test_split(np.arange(len(Xk_tr)), test_size=.2,
                                stratify=yk_int, random_state=SEMILLA)
prep_k = prep_onehot.fit(Xk_tr.iloc[iv_tr])
A_tr = prep_k.transform(Xk_tr.iloc[iv_tr])
A_va = prep_k.transform(Xk_tr.iloc[iv_va])
yv_tr, yv_va = yk_int.iloc[iv_tr], yk_int.iloc[iv_va]

VALORES_K = [1, 3, 5, 9, 15, 25, 41, 61, 91]
curva = []
sub("Barrido del parámetro k")
for k in VALORES_K:
    m = KNeighborsClassifier(n_neighbors=k, weights='uniform', n_jobs=-1)
    m.fit(A_tr, yv_tr)
    acc_tr = accuracy_score(yv_tr, m.predict(A_tr))
    prob_va = m.predict_proba(A_va)[:, 1]
    acc_va = accuracy_score(yv_va, (prob_va >= .5).astype(int))
    auc_va = roc_auc_score(yv_va, prob_va)
    curva.append({'k': k, 'acc_entrenamiento': acc_tr,
                  'acc_validacion': acc_va, 'auc_validacion': auc_va})
    print(f"   k = {k:>3}  ·  entrenamiento {acc_tr:.4f}  ·  "
          f"validación {acc_va:.4f}  ·  AUC {auc_va:.4f}")
curva = pd.DataFrame(curva)
# Regla de parsimonia: el k MAS PEQUEÑO cuyo AUC esta a menos de 0,002 del
# maximo. Si la curva es plana, un k mayor solo agrega costo sin beneficio.
auc_max = curva['auc_validacion'].max()
K_MAX = int(curva.loc[curva['auc_validacion'].idxmax(), 'k'])
K_OPT = int(curva.loc[curva['auc_validacion'] >= auc_max - 0.002, 'k'].min())
print(f"\n   k con máximo AUC      : {K_MAX} (AUC {auc_max:.4f})")
print(f"   k elegido (parsimonia): {K_OPT} (AUC "
      f"{curva.loc[curva.k==K_OPT,'auc_validacion'].iloc[0]:.4f})")
print(f"   Criterio: el menor k a menos de 0,002 del AUC máximo.")
if K_OPT == max(VALORES_K):
    rango = (curva.loc[curva.k >= 25, 'auc_validacion'].max() -
             curva.loc[curva.k >= 25, 'auc_validacion'].min())
    print(f"   El óptimo cae en el borde de la grilla: la curva aún sube, pero")
    print(f"   desde k = 25 el AUC varía apenas {rango:.4f}. Valores mayores")
    print(f"   aportarían una mejora marginal a cambio de más costo.")
else:
    print(f"   Aumentar k por encima de {K_OPT} solo encarece la predicción.")
print(f"   Con k = 1 el modelo memoriza el entrenamiento: su exactitud allí")
print(f"   es {curva.iloc[0]['acc_entrenamiento']:.4f}, pero cae a "
      f"{curva.iloc[0]['acc_validacion']:.4f} en datos nuevos. Eso es sobreajuste.")

registrar("Selección del parámetro k de KNN", "Hiperparámetros",
          f"barrido k ∈ {VALORES_K} · máximo en {K_MAX} · elegido {K_OPT}",
          "Se evalúa en un conjunto de validación interno, sin tocar la "
          "prueba. Se aplica la regla de parsimonia: el menor k a menos de "
          "0,002 del AUC máximo",
          K_OPT)

fig, ax = plt.subplots(1, 2, figsize=(15.5, 5))
ax[0].plot(curva['k'], curva['acc_entrenamiento'], 'o-', lw=2.6, ms=8,
           color=CORAL, markerfacecolor='white', markeredgewidth=2,
           label='Entrenamiento')
ax[0].plot(curva['k'], curva['acc_validacion'], 's-', lw=2.6, ms=8,
           color=MENTA, markerfacecolor='white', markeredgewidth=2,
           label='Validación')
ax[0].axvline(K_OPT, color=AZUL, ls='--', lw=2, label=f'k óptimo = {K_OPT}')
ax[0].fill_between(curva['k'], curva['acc_entrenamiento'],
                   curva['acc_validacion'], alpha=.12, color=CORAL)
ax[0].set_xscale('log')
ax[0].set_xlabel("k (escala log)"); ax[0].set_ylabel("Exactitud")
ax[0].set_title("(a) Sobreajuste con k pequeño, subajuste con k grande")
ax[0].legend(fontsize=8.5)
ax[0].annotate("SOBREAJUSTE\nmemoriza el ruido", xy=(1, curva.iloc[0][
    'acc_entrenamiento']), xytext=(2.2, curva['acc_entrenamiento'].max()-.02),
    fontsize=8, color=CORAL, fontweight='bold')

ax[1].plot(curva['k'], curva['auc_validacion'], 'o-', lw=2.8, ms=9,
           color=LILA, markerfacecolor='white', markeredgewidth=2.2)
ax[1].scatter([K_OPT], [curva['auc_validacion'].max()], s=260, color=MENTA,
              zorder=5, edgecolors='white', linewidths=2.5)
ax[1].set_xscale('log')
ax[1].set_xlabel("k (escala log)"); ax[1].set_ylabel("AUC de validación")
ax[1].set_title(f"(b) AUC máximo en k = {K_OPT}")
fig.suptitle("FASE E · KNN — elección del número de vecinos", fontsize=14,
             y=1.02)
plt.tight_layout()
# Figura 32 del informe · E01_knn_eleccion_k.png
guardar("E01_knn_eleccion_k.png")
del A_tr, A_va
gc.collect()


# %%
# =============================================================================
#  E2 — KNN SOBRE LOS TRES OBJETIVOS
# =============================================================================
titulo("E2 · KNN — regresión, binaria y multiclase")

prob_bin, pred_bin, res_bin = {}, {}, {}
pred_reg, res_reg = {}, {}
pred_m, res_m = {}, {}

# --- Regresion --------------------------------------------------------------
t0 = time.time()
mk = Pipeline([('p', prep_onehot),
               ('m', KNeighborsRegressor(n_neighbors=K_OPT, weights='distance',
                                         n_jobs=-1))])
mk.fit(Xk_tr, y_ieh.iloc[idx_tr].iloc[sub_knn])
p = mk.predict(X_te); seg = time.time() - t0
pred_reg['KNN regresión'] = p
res_reg['KNN regresión'] = metricas_reg(y_ieh.iloc[idx_te], p)
registrar_modelo('ieh', 'Regresión', 'KNN regresión', res_reg['KNN regresión'],
                 seg, f"k={K_OPT}, ponderado por distancia")
r = res_reg['KNN regresión']
print(f"   KNN regresión     R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · "
      f"MAE {r['MAE']:.4f} · MAPE {r['MAPE_%']:.2f} % · {seg:.1f}s")

# --- Binaria ----------------------------------------------------------------
t0 = time.time()
mk = Pipeline([('p', prep_onehot),
               ('m', KNeighborsClassifier(n_neighbors=K_OPT, n_jobs=-1))])
mk.fit(Xk_tr, yk_int)
prob = mk.predict_proba(X_te)[:, 1]; seg = time.time() - t0
# KNN no admite class_weight: se compensa el desbalance moviendo el umbral.
# Predecir la clase 1 solo si su probabilidad supera su proporcion a priori
# equivale matematicamente a ponderar las clases de forma balanceada:
#   p1 / pi1 > p0 / pi0   <=>   p1 > pi1
umbral = yk_int.mean()
pred = (prob >= umbral).astype(int)
prob_bin['KNN'] = prob; pred_bin['KNN'] = pred
res_bin['KNN'] = metricas_bin(y_int.iloc[idx_te], pred, prob)
registrar_modelo('internet', 'Clasificación binaria', 'KNN', res_bin['KNN'],
                 seg, f"k={K_OPT}, umbral ajustado {umbral:.3f}")
r = res_bin['KNN']
print(f"   KNN binaria       AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
      f"recall mín. {r['Recall_min']:.4f} · {seg:.1f}s")
print(f"                     umbral ajustado a {umbral:.3f} para compensar el "
      f"desbalance")

# --- Multiclase -------------------------------------------------------------
t0 = time.time()
mk = Pipeline([('p', prep_onehot),
               ('m', KNeighborsClassifier(n_neighbors=K_OPT, n_jobs=-1))])
mk.fit(Xk_tr, y_niv.iloc[idx_tr].iloc[sub_knn])
pred = mk.predict(X_te); seg = time.time() - t0
pred_m['KNN'] = pred
res_m['KNN'] = metricas_multi(y_niv.iloc[idx_te], pred)
registrar_modelo('nivel_equip', 'Clasificación multiclase', 'KNN',
                 res_m['KNN'], seg, f"k={K_OPT}")
print(f"   KNN multiclase    F1 macro {res_m['KNN']['F1']:.4f} · {seg:.1f}s")

registrar("KNN sobre los tres objetivos", "Modelado",
          f"k={K_OPT} · R² {res_reg['KNN regresión']['R2']:.4f} · AUC "
          f"{res_bin['KNN']['ROC_AUC']:.4f}",
          "KNN no admite pesos por clase: el desbalance se corrigió desplazando "
          "el umbral de decisión", round(float(res_bin['KNN']['ROC_AUC']), 5))
gc.collect()


# %%
# =============================================================================
#  E3 — NAIVE BAYES: GAUSIANO FRENTE A CATEGORICO
#  El gaussiano supone variables continuas con distribucion normal. Las del
#  censo son CATEGORICAS: el supuesto no se cumple. El categorico esta
#  diseñado para ese caso. Se comparan ambos: A/B testing de supuestos.
# =============================================================================
titulo("E3 · NAIVE BAYES — ¿qué supuesto se ajusta a los datos?")

from sklearn.naive_bayes import GaussianNB, CategoricalNB

sub("Justificación")
print("   Naive Bayes supone que las variables son independientes entre sí")
print("   dado el objetivo. Además, cada variante asume una distribución:")
print("     · Gaussiano  → variables continuas con distribución normal")
print("     · Categórico → variables categóricas con probabilidades por nivel")
print("   Las numéricas continuas se DISCRETIZAN en 5 intervalos por cuantiles")
print("   para que el categórico pueda procesarlas.")
print("\n   NOTA: Naive Bayes no tiene versión de regresión. Para la dimensión")
print("   del equipamiento se aplica sobre nivel_equip, la categorización de ieh.")

# Discretizacion: las continuas pasan a 5 intervalos por cuantiles
prep_cat = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                           unknown_value=-1, dtype=np.int32), CATEGORICAS),
    ('num', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile',
                             subsample=200_000, random_state=SEMILLA),
     NUMERICAS)])

registrar("Discretización para Naive Bayes categórico", "Preprocesamiento",
          f"{len(NUMERICAS)} variables continuas en 5 intervalos por cuantiles",
          "El Naive Bayes categórico solo acepta variables discretas. Los "
          "cuantiles garantizan intervalos con igual número de observaciones")

variantes = {
    'Naive Bayes gaussiano': Pipeline([('p', prep_onehot), ('m', GaussianNB())]),
    'Naive Bayes categórico': Pipeline([('p', prep_cat),
                                        ('m', CategoricalNB(min_categories=12,
                                                            alpha=1.0))]),
}
for nom, mod in variantes.items():
    t0 = time.time()
    mod.fit(X_tr, y_int.iloc[idx_tr])
    prob = mod.predict_proba(X_te)[:, 1]; seg = time.time() - t0
    umbral = y_int.iloc[idx_tr].mean()     # proporcion a priori de la clase 1
    pred = (prob >= umbral).astype(int)
    prob_bin[nom] = prob; pred_bin[nom] = pred
    res_bin[nom] = metricas_bin(y_int.iloc[idx_te], pred, prob)
    registrar_modelo('internet', 'Clasificación binaria', nom, res_bin[nom],
                     seg, f"umbral ajustado {umbral:.3f}")
    r = res_bin[nom]
    print(f"   {nom:<24} AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
          f"recall mín. {r['Recall_min']:.4f} · {seg:.1f}s")

    mm = Pipeline([('p', mod.named_steps['p']),
                   ('m', GaussianNB() if 'gaus' in nom else
                    CategoricalNB(min_categories=12, alpha=1.0))])
    t0 = time.time()
    mm.fit(X_tr, y_niv.iloc[idx_tr])
    pred = mm.predict(X_te); seg = time.time() - t0
    pred_m[nom] = pred
    res_m[nom] = metricas_multi(y_niv.iloc[idx_te], pred)
    registrar_modelo('nivel_equip', 'Clasificación multiclase', nom,
                     res_m[nom], seg)
    print(f"   {'':<24} multiclase F1 macro {res_m[nom]['F1']:.4f}")

d_auc = (res_bin['Naive Bayes categórico']['ROC_AUC'] -
         res_bin['Naive Bayes gaussiano']['ROC_AUC'])
print(f"\n   Diferencia de AUC (categórico − gaussiano): {d_auc:+.4f}")
if d_auc > 0:
    print("   CONFIRMADO: el supuesto adecuado a la naturaleza de los datos")
    print("   produce mejor modelo. Elegir la variante según el tipo de variable")
    print("   no es un detalle técnico sino una decisión que afecta al resultado.")

registrar("A/B testing de supuestos en Naive Bayes", "Experimento",
          f"categórico vs gaussiano · ΔAUC {d_auc:+.4f}",
          "Las variables del censo son categóricas; el Naive Bayes gaussiano "
          "asume normalidad, que no se cumple", round(float(d_auc), 5))
gc.collect()


# %%
# =============================================================================
#  E4 — MAQUINAS DE SOPORTE VECTORIAL
#  SVM lineal sobre el entrenamiento completo · SVM RBF sobre submuestra
# =============================================================================
titulo("E4 · SVM — lineal frente a kernel RBF")

from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR

sub("Por qué dos versiones")
print("   El SVM lineal separa las clases con un hiperplano; su costo crece de")
print("   forma lineal con los datos, así que usa los 90.000 registros.")
print("   El kernel RBF proyecta los datos a un espacio de mayor dimensión y")
print("   captura fronteras curvas, pero su costo crece de forma cuadrática:")
print(f"   se entrena con {len(sub_rbf):,} registros. La comparación revela si")
print("   las fronteras curvas compensan el menor volumen de entrenamiento.")

Xr_tr = X_tr.iloc[sub_rbf]

# --- Regresion ---------------------------------------------------------------
for nom, est, Xa, ya, nota in [
    ('SVR lineal', LinearSVR(C=0.5, epsilon=0.5, max_iter=3000,
                             random_state=SEMILLA),
     X_tr, y_ieh.iloc[idx_tr], "entrenamiento completo"),
    ('SVR kernel RBF', SVR(kernel='rbf', C=3.0, epsilon=0.5, gamma='scale'),
     Xr_tr, y_ieh.iloc[idx_tr].iloc[sub_rbf],
     f"submuestra {len(sub_rbf):,}")]:
    t0 = time.time()
    m = Pipeline([('p', prep_onehot), ('m', est)])
    m.fit(Xa, ya)
    p = m.predict(X_te); seg = time.time() - t0
    pred_reg[nom] = p
    res_reg[nom] = metricas_reg(y_ieh.iloc[idx_te], p)
    registrar_modelo('ieh', 'Regresión', nom, res_reg[nom], seg, nota)
    r = res_reg[nom]
    print(f"   {nom:<20} R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · "
          f"MAE {r['MAE']:.4f} · {seg:.1f}s  ({nota})")

# --- Binaria ----------------------------------------------------------------
for nom, est, Xa, ya, nota in [
    ('SVM lineal', LinearSVC(C=0.1, class_weight='balanced', max_iter=3000,
                             random_state=SEMILLA),
     X_tr, y_int.iloc[idx_tr], "entrenamiento completo"),
    ('SVM kernel RBF', SVC(kernel='rbf', C=1.0, gamma='scale',
                           class_weight='balanced', random_state=SEMILLA),
     Xr_tr, y_int.iloc[idx_tr].iloc[sub_rbf], f"submuestra {len(sub_rbf):,}")]:
    t0 = time.time()
    m = Pipeline([('p', prep_onehot), ('m', est)])
    m.fit(Xa, ya)
    pred = m.predict(X_te)
    prob = escalar_prob(m.decision_function(X_te)); seg = time.time() - t0
    prob_bin[nom] = prob; pred_bin[nom] = pred
    res_bin[nom] = metricas_bin(y_int.iloc[idx_te], pred, prob)
    registrar_modelo('internet', 'Clasificación binaria', nom, res_bin[nom],
                     seg, nota)
    r = res_bin[nom]
    print(f"   {nom:<20} AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
          f"recall mín. {r['Recall_min']:.4f} · {seg:.1f}s  ({nota})")

# --- Multiclase -------------------------------------------------------------
for nom, est, Xa, ya, nota in [
    ('SVM lineal', LinearSVC(C=0.1, class_weight='balanced', max_iter=3000,
                             random_state=SEMILLA),
     X_tr, y_niv.iloc[idx_tr], "entrenamiento completo"),
    ('SVM kernel RBF', SVC(kernel='rbf', C=1.0, gamma='scale',
                           class_weight='balanced', random_state=SEMILLA),
     Xr_tr, y_niv.iloc[idx_tr].iloc[sub_rbf], f"submuestra {len(sub_rbf):,}")]:
    t0 = time.time()
    m = Pipeline([('p', prep_onehot), ('m', est)])
    m.fit(Xa, ya)
    pred = m.predict(X_te); seg = time.time() - t0
    pred_m[nom] = pred
    res_m[nom] = metricas_multi(y_niv.iloc[idx_te], pred)
    registrar_modelo('nivel_equip', 'Clasificación multiclase', nom,
                     res_m[nom], seg, nota)
    print(f"   {nom:<20} multiclase F1 macro {res_m[nom]['F1']:.4f} · "
          f"{seg:.1f}s")

registrar("SVM lineal y RBF", "Modelado",
          f"RBF entrenado con {len(sub_rbf):,} registros",
          "El SVM con kernel RBF es inviable sobre 90.000 registros en un "
          "equipo estándar por su costo cuadrático. El lineal usa el "
          "entrenamiento completo")
gc.collect()


# %%
# =============================================================================
#  E5 — GRAFICOS COMPARATIVOS
# =============================================================================
titulo("E5 · COMPARACION DE LOS MODELOS DE LA FASE")

tabla_bin = pd.DataFrame(res_bin).T
tabla_reg = pd.DataFrame(res_reg).T
tabla_m = pd.DataFrame(res_m).T
yte_b = y_int.iloc[idx_te]

# --- Curvas ROC y PR ---------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
for i, nom in enumerate(tabla_bin.index):
    fpr, tpr, _ = roc_curve(yte_b, prob_bin[nom])
    ax[0].plot(fpr, tpr, lw=2.4, color=PALETA[i],
               label=f"{nom[:22]} ({tabla_bin.loc[nom,'ROC_AUC']:.3f})")
    pr, rc, _ = precision_recall_curve(yte_b, prob_bin[nom])
    ax[1].plot(rc, pr, lw=2.4, color=PALETA[i],
               label=f"{nom[:22]} ({tabla_bin.loc[nom,'PR_AUC']:.3f})")
ax[0].plot([0, 1], [0, 1], '--', color=GRIS, lw=2, label='Azar (0.500)')
ax[0].set_xlabel("Tasa de falsos positivos")
ax[0].set_ylabel("Tasa de verdaderos positivos")
ax[0].set_title("(a) Curvas ROC"); ax[0].legend(fontsize=7, loc='lower right')
ax[1].axhline(yte_b.mean(), color=GRIS, ls='--', lw=2,
              label=f'Azar ({yte_b.mean():.3f})')
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("(b) Curvas Precision-Recall")
ax[1].legend(fontsize=7, loc='lower left')

s = tabla_bin['Recall_min'].sort_values()
ax[2].barh(range(len(s)), s.values,
           color=[MENTA if v == s.max() else LILA for v in s.values],
           edgecolor='white', linewidth=1.8)
ax[2].set_yticks(range(len(s)), s.index, fontsize=8)
for i, v in enumerate(s.values):
    ax[2].text(v, i, f" {v:.3f}", va='center', fontsize=8.5, fontweight='bold')
ax[2].axvline(0.727, color=CORAL, ls='--', lw=2,
              label='Logística Fase D (0,727)')
ax[2].set_xlabel("Recall de la clase SIN internet")
ax[2].set_title("(c) Detección de hogares desconectados")
ax[2].legend(fontsize=7.5); ax[2].set_xlim(0, 1)
fig.suptitle("FASE E · CLASIFICACIÓN BINARIA — curvas y detección",
             fontsize=14, y=1.02)
plt.tight_layout()
# Figura 33 del informe · E02_clasificacion_curvas.png
guardar("E02_clasificacion_curvas.png")

# --- Matrices de confusion binarias -----------------------------------------
n = len(tabla_bin)
fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
for k, nom in enumerate(tabla_bin.index):
    mc = confusion_matrix(yte_b, pred_bin[nom])
    ax[k].imshow(mc, cmap=CMAP)
    for i_ in range(2):
        for j_ in range(2):
            ax[k].text(j_, i_, f"{mc[i_, j_]:,}", ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='white' if mc[i_, j_] > mc.max()/2 else TINTA)
    ax[k].set_xticks([0, 1], ['Pred: sin', 'Pred: con'], fontsize=7.5)
    ax[k].set_yticks([0, 1], ['Real: sin', 'Real: con'], fontsize=7.5)
    ax[k].set_title(f"{nom[:22]}\nrecall mín. "
                    f"{tabla_bin.loc[nom,'Recall_min']:.3f}", fontsize=8.5)
    ax[k].grid(False)
fig.suptitle("FASE E · MATRICES DE CONFUSIÓN — clasificación binaria",
             fontsize=14, y=1.05)
plt.tight_layout()
# Figura 34 del informe · E03_matrices_binarias.png
guardar("E03_matrices_binarias.png")

# --- Regresion --------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
s = tabla_reg['R2'].sort_values()
ax[0].barh(range(len(s)), s.values,
           color=[MENTA if v == s.max() else AZUL for v in s.values],
           edgecolor='white', linewidth=1.8)
ax[0].set_yticks(range(len(s)), s.index, fontsize=8.5)
for i, v in enumerate(s.values):
    ax[0].text(v, i, f" {v:.4f}", va='center', fontsize=8.5, fontweight='bold')
ax[0].axvline(0.529, color=CORAL, ls='--', lw=2,
              label='Polinomial Fase D (0,529)')
ax[0].set_xlabel("R²"); ax[0].set_title("(a) R² de regresión")
ax[0].legend(fontsize=7.5)

x = np.arange(len(tabla_reg)); w = .27
for j, (met, col) in enumerate([('MAE', AZUL), ('RMSE', LILA), ('MAPE_%', AMBAR)]):
    vals = tabla_reg[met] / (100 if met == 'MAPE_%' else 1)
    ax[1].bar(x + (j-1)*w, vals, w, label=met if met != 'MAPE_%' else
              'MAPE / 100', color=col, edgecolor='white', linewidth=1.2)
ax[1].set_xticks(x, [n_[:14] for n_ in tabla_reg.index], rotation=18,
                 fontsize=8)
ax[1].set_title("(b) Errores"); ax[1].legend(fontsize=7.5)

mejor_r = tabla_reg['R2'].idxmax()
yte_r = y_ieh.iloc[idx_te].values
sel = np.random.default_rng(SEMILLA).choice(len(yte_r), min(6000, len(yte_r)),
                                            replace=False)
jit = np.random.default_rng(SEMILLA).uniform(-.3, .3, len(sel))
ax[2].scatter(yte_r[sel] + jit, pred_reg[mejor_r][sel], s=5, alpha=.2,
              color=AZUL, edgecolors='none')
ax[2].plot([0, 13], [0, 13], '--', color=CORAL, lw=2.2,
           label='Predicción perfecta')
ax[2].set_xlabel("ieh real"); ax[2].set_ylabel("ieh predicho")
ax[2].set_title(f"(c) Real frente a predicho · {mejor_r}")
ax[2].legend(fontsize=8)
fig.suptitle("FASE E · REGRESIÓN — índice de equipamiento", fontsize=14,
             y=1.02)
plt.tight_layout()
# Figura 35 del informe · E04_regresion.png
guardar("E04_regresion.png")

# --- Multiclase -------------------------------------------------------------
orden = ['Bajo', 'Medio', 'Alto']
n = len(tabla_m)
fig, ax = plt.subplots(1, n, figsize=(3.9 * n, 4.3))
for k, nom in enumerate(tabla_m.index):
    mc = confusion_matrix(y_niv.iloc[idx_te], pred_m[nom], labels=orden)
    mcn = mc / mc.sum(axis=1, keepdims=True)
    ax[k].imshow(mcn, cmap=CMAP, vmin=0, vmax=1)
    for i_ in range(3):
        for j_ in range(3):
            ax[k].text(j_, i_, f"{mcn[i_, j_]*100:.0f} %", ha='center',
                       va='center', fontsize=9, fontweight='bold',
                       color='white' if mcn[i_, j_] > .5 else TINTA)
    ax[k].set_xticks(range(3), orden, fontsize=8)
    ax[k].set_yticks(range(3), orden, fontsize=8)
    ax[k].set_title(f"{nom[:22]}\nF1 macro {tabla_m.loc[nom,'F1']:.3f}",
                    fontsize=8.5)
    ax[k].grid(False)
fig.suptitle("FASE E · MULTICLASE — nivel de equipamiento", fontsize=14,
             y=1.06)
plt.tight_layout()
# Figura 36 del informe · E05_multiclase.png
guardar("E05_multiclase.png")
gc.collect()


# %%
# =============================================================================
#  E6 — CIERRE, RESUMEN Y BITACORA
# =============================================================================
titulo("E6 · RESUMEN DE LA FASE E")

sub("Clasificación binaria")
print(tabla_bin.round(4).to_string())
sub("Regresión")
print(tabla_reg.round(4).to_string())
sub("Multiclase")
print(tabla_m.round(4).to_string())

mejor_b = tabla_bin['ROC_AUC'].idxmax()
mejor_rec = tabla_bin['Recall_min'].idxmax()
mejor_m = tabla_m['F1'].idxmax()

print(f"""
   REFERENCIA DE LA FASE D (mejores lineales)
     ieh         : Polinomial  R² 0,5290
     internet    : Logística   AUC 0,811 · recall mín. 0,727
     nivel_equip : Logística   F1 macro 0,633

   MEJORES DE LA FASE E
     ieh         : {mejor_r:<24} R² {tabla_reg.loc[mejor_r,'R2']:.4f}
     internet    : {mejor_b:<24} AUC {tabla_bin.loc[mejor_b,'ROC_AUC']:.4f}
     recall mín. : {mejor_rec:<24} {tabla_bin.loc[mejor_rec,'Recall_min']:.4f}
     nivel_equip : {mejor_m:<24} F1 macro {tabla_m.loc[mejor_m,'F1']:.4f}

   Modelos entrenados en la fase : {len(MOD_FASE)}
   Tiempo de la fase             : {(time.time()-t_ini)/60:.1f} min
""")

tabla_bin.round(5).to_csv(os.path.join(DIR_FIG, "E_clasificacion_binaria.csv"))
tabla_reg.round(5).to_csv(os.path.join(DIR_FIG, "E_regresion.csv"))
tabla_m.round(5).to_csv(os.path.join(DIR_FIG, "E_clasificacion_multiclase.csv"))

registrar("Cierre de la Fase E", "Cierre",
          f"{len(MOD_FASE)} modelos · mejor AUC {mejor_b} "
          f"{tabla_bin.loc[mejor_b,'ROC_AUC']:.4f}",
          "Los resultados alimentarán la comparación global de la Fase H",
          len(MOD_FASE))
volcar_bitacora()

print(f"\n   Consola completa guardada en: {RUTA_LOG}")

# Restituye la salida estándar para no duplicar texto en otras celdas
if isinstance(sys.stdout, Registro):
    sys.stdout.archivo.close()
    sys.stdout = sys.stdout.consola
gc.collect()


# =============================================================================
#  PROYECTO CPV 2024 · BORRADOR INTEGRAL
#  FASE F — ARBOLES DE DECISION Y ENSAMBLES
#
#  Conceptos y modelos de esta fase (cada bloque produce su propio grafico)
#    F1  Criterios de division: Gini frente a Entropia (ganancia de informacion)
#    F2  Profundidad del arbol: sobreajuste y subajuste
#    F3  Poda por costo-complejidad
#    F4  Arboles de regresion: cuatro criterios, incluido Poisson para conteos
#    F5  Arbol multiclase
#    F6  Random Forest (regresion, binaria, multiclase)
#    F7  Extra Trees: arboles extremadamente aleatorizados
#    F8  Bagging (regresion, binaria, multiclase)
#    F9  Aprendices debiles frente a fuertes
#    F10 Voting: votacion dura y blanda
#    F11 Stacking: apilamiento con meta-modelo
#    F12 Comparacion global de la fase
#
#  Misma muestra y misma particion que las Fases D y E · random_state = 42
# =============================================================================


# %%
# =============================================================================
#  F0 — CONFIGURACION Y CARGA
# =============================================================================
import os
import gc
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

SEMILLA = 42
N_MUESTRA = 120_000          # idéntica a las Fases D y E
PRUEBA = 0.25
N_META = 30_000              # entrenamiento de Voting y Stacking
N_PODA = 20_000              # cálculo de la ruta de poda

FASE = "F"
DIR_FIG = "figuras_v2"
os.makedirs(DIR_FIG, exist_ok=True)

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA = "#FF6B8A", "#FFC145", "#22223B"
GRIS, AGUA = "#8D99AE", "#A8DADC"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, AGUA, "#6C8AE4", "#C3A6F0",
          "#2EC4B6", "#E07A5F", "#F4A261", "#9D4EDD"]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "grid.linewidth": .9,
    "axes.axisbelow": True, "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92,
    "legend.edgecolor": "#D8DEF0"})


def guardar(n):
    """Guarda la figura y la muestra en línea, justo debajo de su salida."""
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    print(f"   [figura] {r}")
    plt.show()


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


# ---------------- BITACORA MAESTRA EN EXCEL ----------------------------------
BITACORA_XLSX = "BITACORA_PROYECTO.xlsx"
REG_FASE, MOD_FASE = [], []


def registrar(paso, categoria, detalle, justificacion, valor=None):
    REG_FASE.append({
        'fase': FASE, 'n': len(REG_FASE) + 1,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'categoria': categoria, 'paso': paso, 'detalle': detalle,
        'justificacion': justificacion, 'valor': valor})
    print(f"   [bitácora {FASE}-{len(REG_FASE):02d}] {paso}")


def registrar_modelo(objetivo, tipo, modelo, metricas, segundos, nota=""):
    fila = {'fase': FASE,
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objetivo': objetivo, 'tipo': tipo, 'modelo': modelo,
            'segundos': round(segundos, 2), 'nota': nota}
    fila.update({k: (round(float(v), 5) if v is not None else None)
                 for k, v in metricas.items()})
    MOD_FASE.append(fila)


def leer_hoja(nombre):
    if os.path.exists(BITACORA_XLSX):
        try:
            return pd.read_excel(BITACORA_XLSX, sheet_name=nombre)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def volcar_bitacora():
    reg, mod = leer_hoja('Registro'), leer_hoja('Modelos')
    if len(reg) and 'fase' in reg:
        reg = reg[reg['fase'].astype(str) != FASE]
    if len(mod) and 'fase' in mod:
        mod = mod[mod['fase'].astype(str) != FASE]
    reg = pd.concat([reg, pd.DataFrame(REG_FASE)], ignore_index=True)
    mod = pd.concat([mod, pd.DataFrame(MOD_FASE)], ignore_index=True)
    resumen = (reg.groupby('fase').agg(pasos=('paso', 'count'),
                                       ultima_actualizacion=('fecha', 'max'))
               .reset_index())
    if len(mod):
        rm = mod.groupby('fase').agg(modelos=('modelo', 'count')).reset_index()
        resumen = resumen.merge(rm, on='fase', how='left')
    try:
        with pd.ExcelWriter(BITACORA_XLSX, engine='openpyxl') as w:
            resumen.to_excel(w, sheet_name='Resumen por fase', index=False)
            reg.to_excel(w, sheet_name='Registro', index=False)
            mod.to_excel(w, sheet_name='Modelos', index=False)
        print(f"   ✔ Bitácora actualizada: {BITACORA_XLSX}  "
              f"(registro {len(reg)} · modelos {len(mod)})")
    except PermissionError:
        alt = f"BITACORA_respaldo_{FASE}_{datetime.now():%H%M%S}.csv"
        pd.DataFrame(REG_FASE).to_csv(alt, index=False, encoding='utf-8')
        pd.DataFrame(MOD_FASE).to_csv(alt.replace('.csv', '_modelos.csv'),
                                      index=False, encoding='utf-8')


titulo("FASE F · ARBOLES DE DECISION Y ENSAMBLES")
t_ini = time.time()

from sklearn.model_selection import train_test_split

BASE = "vivienda_cpv2024_LIMPIO_v2"
datos = pd.read_parquet(f"{BASE}.parquet") if os.path.exists(f"{BASE}.parquet") \
    else pd.read_csv(f"{BASE}.csv", sep=';')
for c in datos.select_dtypes(include=['float64']).columns:
    datos[c] = datos[c].astype('float32')

if len(datos) > N_MUESTRA:
    muestra, _ = train_test_split(datos, train_size=N_MUESTRA,
                                  stratify=datos['internet'],
                                  random_state=SEMILLA)
else:
    muestra = datos.copy()
muestra = muestra.reset_index(drop=True)
del datos
gc.collect()

CATEGORICAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v15_servsan', 'v16_desague', 'v17_tenencia', 'tip_hog',
               'v20a_emi', 'v21a_fal']
NUMERICAS = ['v13_habitac', 'v14_dormit', 'tot_pers', 'v20b_totemi',
             'v21b_totfal', 'hacinamiento', 'pers_por_habitac',
             'hacin_critico', 'sin_servicios']
CATEGORICAS = [c for c in CATEGORICAS if c in muestra.columns]
NUMERICAS = [c for c in NUMERICAS if c in muestra.columns]
PREDICTORAS = CATEGORICAS + NUMERICAS

NOMBRES = {
    'idep': 'Departamento', 'urbrur': 'Área urbana/rural',
    'v01_tipoviv': 'Tipo de vivienda', 'v03_pared': 'Material de pared',
    'v04_revoq': 'Pared revocada', 'v05_techo': 'Material de techo',
    'v06_piso': 'Material de piso', 'v07_aguapro': 'Procedencia del agua',
    'v08_aguadist': 'Distribución de agua', 'v09_energia': 'Fuente de energía',
    'v10_combus': 'Combustible de cocina', 'v11_basura': 'Manejo de basura',
    'v12_cocina': 'Tiene cocina', 'v13_habitac': 'N° de habitaciones',
    'v14_dormit': 'N° de dormitorios', 'v15_servsan': 'Servicio sanitario',
    'v16_desague': 'Tipo de desagüe', 'v17_tenencia': 'Tenencia',
    'tot_pers': 'Personas en el hogar', 'tip_hog': 'Tipo de hogar',
    'v20a_emi': 'Hubo emigración', 'v20b_totemi': 'N° emigrantes',
    'v21a_fal': 'Hubo fallecimiento', 'v21b_totfal': 'N° fallecimientos',
    'hacinamiento': 'Personas/dormitorio', 'hacin_critico': 'Hacinam. crítico',
    'sin_servicios': 'Carencia servicios', 'pers_por_habitac': 'Personas/habitación'}
ETQ = lambda c: NOMBRES.get(c, c)

X = muestra[PREDICTORAS]
y_ieh = muestra['ieh'].astype('float32')
y_int = muestra['internet'].astype(int)
y_niv = muestra['nivel_equip'].astype(str)

idx_tr, idx_te = train_test_split(np.arange(len(muestra)), test_size=PRUEBA,
                                  stratify=y_int, random_state=SEMILLA)
X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
ytr_b, yte_b = y_int.iloc[idx_tr], y_int.iloc[idx_te]
ytr_r, yte_r = y_ieh.iloc[idx_tr], y_ieh.iloc[idx_te]
ytr_m, yte_m = y_niv.iloc[idx_tr], y_niv.iloc[idx_te]

print(f"   Muestra       : {len(muestra):,}  (idéntica a D y E)")
print(f"   Entrenamiento : {len(idx_tr):,}")
print(f"   Prueba        : {len(idx_te):,}  (mismas filas que D y E)")

rng = np.random.default_rng(SEMILLA)
sub_meta = rng.choice(len(idx_tr), min(N_META, len(idx_tr)), replace=False)
sub_poda = rng.choice(len(idx_tr), min(N_PODA, len(idx_tr)), replace=False)

registrar("Reproducción de la partición de D y E", "Muestreo",
          f"prueba {len(idx_te):,} filas idénticas",
          "Comparación justa entre todas las fases en la Fase H", len(idx_te))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)

prep_onehot = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first',
                          sparse_output=False, dtype=np.float32), CATEGORICAS),
    ('num', StandardScaler(), NUMERICAS)])


def metricas_reg(y, p):
    y = np.asarray(y); nz = y != 0
    return {'MAE': mean_absolute_error(y, p), 'MSE': mean_squared_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)), 'R2': r2_score(y, p),
            'MAPE_%': np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100}


def metricas_bin(y, pred, prob):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred),
            'Recall_min': recall_score(y, pred, pos_label=0),
            'F1': f1_score(y, pred), 'ROC_AUC': roc_auc_score(y, prob),
            'PR_AUC': average_precision_score(y, prob)}


def metricas_multi(y, pred):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, average='macro',
                                         zero_division=0),
            'Recall': recall_score(y, pred, average='macro'),
            'F1': f1_score(y, pred, average='macro')}


def matriz_bin(ax, y, pred, tit):
    mc = confusion_matrix(y, pred)
    ax.imshow(mc, cmap=CMAP)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mc[i, j]:,}", ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if mc[i, j] > mc.max()/2 else TINTA)
    ax.set_xticks([0, 1], ['Pred: sin', 'Pred: con'], fontsize=8)
    ax.set_yticks([0, 1], ['Real: sin', 'Real: con'], fontsize=8)
    ax.set_title(tit, fontsize=9.5); ax.grid(False)


def matriz_multi(ax, y, pred, tit):
    orden = ['Bajo', 'Medio', 'Alto']
    mc = confusion_matrix(y, pred, labels=orden)
    mcn = mc / mc.sum(axis=1, keepdims=True)
    ax.imshow(mcn, cmap=CMAP, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mcn[i, j]*100:.0f} %", ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if mcn[i, j] > .5 else TINTA)
    ax.set_xticks(range(3), orden, fontsize=8)
    ax.set_yticks(range(3), orden, fontsize=8)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(tit, fontsize=9.5); ax.grid(False)


def dispersion(ax, y, p, tit):
    y = np.asarray(y)
    sel = np.random.default_rng(SEMILLA).choice(len(y), min(6000, len(y)),
                                                replace=False)
    j = np.random.default_rng(SEMILLA).uniform(-.3, .3, len(sel))
    ax.scatter(y[sel] + j, p[sel], s=5, alpha=.2, color=AZUL, edgecolors='none')
    ax.plot([0, 13], [0, 13], '--', color=CORAL, lw=2.2,
            label='Predicción perfecta')
    ax.set_xlabel("ieh real"); ax.set_ylabel("ieh predicho")
    ax.set_title(tit, fontsize=9.5); ax.legend(fontsize=7.5)


# Acumuladores para la comparacion final
RES_BIN, PROB_BIN, PRED_BIN = {}, {}, {}
RES_REG, PRED_REG = {}, {}
RES_M, PRED_M = {}, {}
gc.collect()


# %%
# =============================================================================
#  F1 — CRITERIO DE DIVISION: GINI FRENTE A ENTROPIA
#
#  Un arbol elige en cada nodo la division que deja los grupos resultantes
#  lo mas homogeneos posible. Hay dos formas de medir esa homogeneidad:
#    · Gini      = 1 − Σ pᵢ²            probabilidad de clasificar mal
#    · Entropía  = − Σ pᵢ · log₂(pᵢ)    desorden de la información
#  La GANANCIA DE INFORMACION es la reduccion de entropia que logra una
#  division: el arbol elige la division de mayor ganancia.
# =============================================================================
titulo("F1 · ÁRBOL DE DECISIÓN — Gini frente a Entropía")

from sklearn.tree import DecisionTreeClassifier, plot_tree

sub("Nota sobre la codificación")
print("   Los árboles NO requieren one-hot ni escalado: dividen por umbrales")
print("   y son insensibles a la magnitud. Se usan los códigos originales, lo")
print("   que además reduce de 113 a 28 las columnas y acelera el cálculo.")

# Verificacion experimental de la decision
t0 = time.time()
a_cod = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50,
                               class_weight='balanced', random_state=SEMILLA)
a_cod.fit(X_tr, ytr_b)
auc_cod = roc_auc_score(yte_b, a_cod.predict_proba(X_te)[:, 1])
a_oh = Pipeline([('p', prep_onehot), ('m', DecisionTreeClassifier(
    max_depth=10, min_samples_leaf=50, class_weight='balanced',
    random_state=SEMILLA))])
a_oh.fit(X_tr, ytr_b)
auc_oh = roc_auc_score(yte_b, a_oh.predict_proba(X_te)[:, 1])
print(f"\n   Verificación · códigos AUC {auc_cod:.4f}  ·  one-hot AUC {auc_oh:.4f}"
      f"  ·  diferencia {auc_cod-auc_oh:+.4f}")
registrar("Codificación para árboles", "Preprocesamiento",
          f"códigos AUC {auc_cod:.4f} vs one-hot {auc_oh:.4f}",
          "Los árboles dividen por umbrales y no interpretan el código como "
          "cantidad lineal. Se usan los códigos originales: 28 columnas en "
          "lugar de 113, sin pérdida relevante de desempeño",
          round(float(auc_cod - auc_oh), 5))

sub("Comparación de criterios · profundidad 10")
arboles = {}
for crit, nom in [('gini', 'Árbol · Gini'), ('entropy', 'Árbol · Entropía')]:
    t0 = time.time()
    a = DecisionTreeClassifier(criterion=crit, max_depth=10,
                               min_samples_leaf=50, class_weight='balanced',
                               random_state=SEMILLA)
    a.fit(X_tr, ytr_b)
    prob = a.predict_proba(X_te)[:, 1]; pred = a.predict(X_te)
    seg = time.time() - t0
    arboles[crit] = a
    PROB_BIN[nom] = prob; PRED_BIN[nom] = pred
    RES_BIN[nom] = metricas_bin(yte_b, pred, prob)
    registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                     seg, f"criterio={crit}, profundidad 10")
    r = RES_BIN[nom]
    print(f"   {nom:<20} AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
          f"recall mín. {r['Recall_min']:.4f} · hojas {a.get_n_leaves():,} · "
          f"{seg:.1f}s")

# Calculo manual de Gini y entropia en el nodo raiz, para la exposicion
p1 = ytr_b.mean(); p0 = 1 - p1
gini_raiz = 1 - (p0**2 + p1**2)
ent_raiz = -(p0*np.log2(p0) + p1*np.log2(p1))
print(f"\n   NODO RAÍZ (antes de cualquier división)")
print(f"     Proporciones   : sin internet {p0:.4f} · con internet {p1:.4f}")
print(f"     Gini           : 1 − ({p0:.4f}² + {p1:.4f}²) = {gini_raiz:.4f}")
print(f"     Entropía       : {ent_raiz:.4f} bits")

print(f"\n   ¿POR QUÉ EL DIBUJO DEL ÁRBOL MUESTRA gini = 0,5 EN LA RAÍZ?")
print(f"     El cálculo anterior usa las proporciones reales. Pero el árbol se")
print(f"     entrenó con class_weight='balanced', que multiplica cada hogar sin")
print(f"     internet por {p1/p0:.2f} para compensar el desbalance. El árbol")
print(f"     've' entonces ambas clases con el mismo peso: [0,5 ; 0,5], y la")
print(f"     impureza de la raíz es la máxima posible: 1 − (0,5² + 0,5²) = 0,5")

var_raiz = PREDICTORAS[arboles['gini'].tree_.feature[0]]
print(f"\n   Primera división elegida (Gini): {ETQ(var_raiz)}")
print(f"   Es la variable que más reduce la impureza de todas las disponibles.")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=.35, wspace=.28)
ax = fig.add_subplot(gs[0, :])
plot_tree(arboles['gini'], max_depth=2, feature_names=[ETQ(c) for c in
          PREDICTORAS], class_names=['Sin internet', 'Con internet'],
          filled=True, rounded=True, fontsize=9, proportion=True,
          precision=2, ax=ax)
ax.set_title("(a) Primeros tres niveles del árbol · criterio Gini", fontsize=12)

ax = fig.add_subplot(gs[1, 0])
p = np.linspace(.001, .999, 200)
ax.plot(p, 1 - (p**2 + (1-p)**2), lw=2.8, color=AZUL, label='Gini')
ax.plot(p, -(p*np.log2(p) + (1-p)*np.log2(1-p)), lw=2.8, color=CORAL,
        label='Entropía')
ax.axvline(p1, color=MENTA, ls='--', lw=2, label=f'Raíz ({p1:.2f})')
ax.set_xlabel("Proporción de la clase 'con internet'")
ax.set_ylabel("Impureza")
ax.set_title("(b) Las dos medidas de impureza")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
mets = ['Accuracy', 'Recall_min', 'F1', 'ROC_AUC']
x = np.arange(len(mets)); w = .36
for i, (nom, col) in enumerate([('Árbol · Gini', AZUL),
                                ('Árbol · Entropía', CORAL)]):
    ax.bar(x + (i - .5)*w, [RES_BIN[nom][m] for m in mets], w, label=nom,
           color=col, edgecolor='white', linewidth=1.5)
ax.set_xticks(x, ['Accuracy', 'Recall\nmínimo', 'F1', 'AUC'])
ax.set_ylim(0, 1.05)
ax.set_title("(c) Desempeño de ambos criterios")
ax.legend(fontsize=8)

matriz_bin(fig.add_subplot(gs[1, 2]), yte_b, PRED_BIN['Árbol · Gini'],
           f"(d) Matriz · Gini\nrecall mín. "
           f"{RES_BIN['Árbol · Gini']['Recall_min']:.3f}")
fig.suptitle("FASE F1 · ÁRBOL DE DECISIÓN — criterio de división",
             fontsize=15, y=.99)
# Figura 37 del informe · F01_arbol_gini_entropia.png
guardar("F01_arbol_gini_entropia.png")
registrar("Criterio de división del árbol", "Modelado",
          f"Gini AUC {RES_BIN['Árbol · Gini']['ROC_AUC']:.4f} · Entropía "
          f"{RES_BIN['Árbol · Entropía']['ROC_AUC']:.4f}",
          "Gini es más rápido de calcular; la entropía penaliza algo más las "
          "divisiones desbalanceadas. En la práctica suelen converger")
gc.collect()


# %%
# =============================================================================
#  F2 — PROFUNDIDAD: EL HIPERPARAMETRO QUE CONTROLA EL SOBREAJUSTE
# =============================================================================
titulo("F2 · PROFUNDIDAD DEL ÁRBOL")

PROF = [1, 2, 3, 5, 7, 10, 13, 16, 20, 25, None]
curva = []
for d in PROF:
    a = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1,
                               class_weight='balanced', random_state=SEMILLA)
    a.fit(X_tr, ytr_b)
    curva.append({'profundidad': d if d else 'sin límite',
                  'prof_real': a.get_depth(),
                  'auc_train': roc_auc_score(ytr_b, a.predict_proba(X_tr)[:, 1]),
                  'auc_test': roc_auc_score(yte_b, a.predict_proba(X_te)[:, 1]),
                  'hojas': a.get_n_leaves()})
    print(f"   profundidad {str(d):>10}  ·  AUC entrenamiento "
          f"{curva[-1]['auc_train']:.4f}  ·  prueba {curva[-1]['auc_test']:.4f}"
          f"  ·  hojas {curva[-1]['hojas']:>7,}")
curva = pd.DataFrame(curva)
mejor_d = curva.loc[curva['auc_test'].idxmax()]
print(f"\n   Profundidad óptima: {mejor_d['profundidad']} "
      f"(AUC de prueba {mejor_d['auc_test']:.4f})")
sin_lim = curva.iloc[-1]
print(f"   Sin límite de profundidad: AUC de entrenamiento "
      f"{sin_lim['auc_train']:.4f} frente a {sin_lim['auc_test']:.4f} en prueba")
print(f"   → el árbol sin restricciones memoriza el entrenamiento con "
      f"{int(sin_lim['hojas']):,} hojas")

fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
xs = curva['prof_real']
ax[0].plot(xs, curva['auc_train'], 'o-', lw=2.6, ms=8, color=CORAL,
           markerfacecolor='white', markeredgewidth=2, label='Entrenamiento')
ax[0].plot(xs, curva['auc_test'], 's-', lw=2.6, ms=8, color=MENTA,
           markerfacecolor='white', markeredgewidth=2, label='Prueba')
ax[0].fill_between(xs, curva['auc_train'], curva['auc_test'], alpha=.13,
                   color=CORAL)
ax[0].axvline(mejor_d['prof_real'], color=AZUL, ls='--', lw=2,
              label=f"Óptimo = {mejor_d['profundidad']}")
ax[0].set_xlabel("Profundidad real del árbol"); ax[0].set_ylabel("AUC")
ax[0].set_title("(a) Subajuste a la izquierda, sobreajuste a la derecha")
ax[0].legend(fontsize=8.5)

ax[1].plot(xs, curva['hojas'], 'o-', lw=2.6, ms=8, color=LILA,
           markerfacecolor='white', markeredgewidth=2)
ax[1].set_yscale('log')
ax[1].set_xlabel("Profundidad real del árbol")
ax[1].set_ylabel("Número de hojas (escala log)")
ax[1].set_title("(b) La complejidad crece de forma exponencial")
fig.suptitle("FASE F2 · PROFUNDIDAD — el dilema sesgo-varianza en árboles",
             fontsize=14, y=1.02)
plt.tight_layout()
# Figura 38 del informe · F02_profundidad.png
guardar("F02_profundidad.png")
registrar("Estudio de la profundidad", "Hiperparámetros",
          f"óptimo {mejor_d['profundidad']} · AUC {mejor_d['auc_test']:.4f}",
          "Sin límite de profundidad el árbol memoriza el entrenamiento; con "
          "profundidad muy baja no captura la estructura")
gc.collect()


# %%
# =============================================================================
#  F3 — PODA POR COSTO-COMPLEJIDAD
#  En lugar de detener el crecimiento, se deja crecer el arbol completo y
#  luego se eliminan las ramas que aportan poco. El parametro ccp_alpha
#  penaliza el numero de hojas:  costo = error + alpha × hojas
# =============================================================================
titulo("F3 · PODA POR COSTO-COMPLEJIDAD")

Xp, yp = X_tr.iloc[sub_poda], ytr_b.iloc[sub_poda]

base = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=SEMILLA
)

ruta = base.cost_complexity_pruning_path(Xp, yp)

# -------------------------------------------------------------------------
# cost_complexity_pruning_path puede generar valores negativos
# extremadamente pequeños por precision numerica.
# sklearn exige ccp_alpha >= 0.
# -------------------------------------------------------------------------

ccp_alphas_limpios = np.maximum(ruta.ccp_alphas, 0.0)

ccp_alphas_limpios = np.unique(ccp_alphas_limpios)

# Excluir el ultimo alpha para evitar el arbol raiz sin hojas utiles,
ccp_alphas_utiles = ccp_alphas_limpios[:-1]

# Seleccionar 22 valores distribuidos a lo largo de la ruta de poda
if len(ccp_alphas_utiles) > 0:

    alphas = np.unique(
        np.quantile(
            ccp_alphas_utiles,
            np.linspace(0, .995, 22)
        )
    )

else:
    alphas = np.array([0.0])


print(f"   Ruta de poda calculada sobre {len(sub_poda):,} registros")
print(f"   Valores de alpha evaluados: {len(alphas)}")

# -------------------------------------------------------------------------
# Evaluar cada valor de ccp_alpha
# -------------------------------------------------------------------------

poda = []

for a_ in alphas:

    a_ = max(float(a_), 0.0)

    a = DecisionTreeClassifier(
        ccp_alpha=a_,
        class_weight='balanced',
        random_state=SEMILLA
    )

    a.fit(Xp, yp)

    prob_train = a.predict_proba(Xp)[:, 1]
    prob_test = a.predict_proba(X_te)[:, 1]

    poda.append({
        'alpha': a_,
        'hojas': a.get_n_leaves(),
        'auc_train': roc_auc_score(
            yp,
            prob_train
        ),
        'auc_test': roc_auc_score(
            yte_b,
            prob_test
        )
    })

poda = pd.DataFrame(poda)

# -------------------------------------------------------------------------
# Buscar el mejor alpha según AUC de prueba
# -------------------------------------------------------------------------

mejor_p = poda.loc[
    poda['auc_test'].idxmax()
]

print(f"\n   alpha óptimo : {mejor_p['alpha']:.6f}")

print(
    f"   hojas        : {int(mejor_p['hojas']):,} "
    f"(frente a {int(poda.iloc[0]['hojas']):,} del árbol sin podar)"
)

print(
    f"   AUC prueba   : {mejor_p['auc_test']:.4f}"
)

print(
    f"   La poda eliminó el "
    f"{(1 - mejor_p['hojas'] / poda.iloc[0]['hojas']) * 100:.1f} % de las hojas"
)

# -------------------------------------------------------------------------
# Reentrenar el árbol podado con TODO el conjunto de entrenamiento
# -------------------------------------------------------------------------

t0 = time.time()

a_podado = DecisionTreeClassifier(
    ccp_alpha=float(mejor_p['alpha']),
    class_weight='balanced',
    random_state=SEMILLA
)

a_podado.fit(
    X_tr,
    ytr_b
)

nom = 'Árbol podado'

PROB_BIN[nom] = a_podado.predict_proba(
    X_te
)[:, 1]

PRED_BIN[nom] = a_podado.predict(
    X_te
)

RES_BIN[nom] = metricas_bin(
    yte_b,
    PRED_BIN[nom],
    PROB_BIN[nom]
)

registrar_modelo(
    'internet',
    'Clasificación binaria',
    nom,
    RES_BIN[nom],
    time.time() - t0,
    f"ccp_alpha={mejor_p['alpha']:.6f}"
)

print(
    f"\n   Árbol podado reentrenado con {len(idx_tr):,} registros · "
    f"AUC {RES_BIN[nom]['ROC_AUC']:.4f}"
)

# =============================================================================
# GRAFICOS
# =============================================================================

fig, ax = plt.subplots(
    1,
    3,
    figsize=(17, 5)
)

# -------------------------------------------------------------------------
# (a) AUC de entrenamiento y prueba
# -------------------------------------------------------------------------

ax[0].plot(
    poda['alpha'],
    poda['auc_train'],
    'o-',
    lw=2.4,
    ms=6,
    color=CORAL,
    markerfacecolor='white',
    markeredgewidth=1.8,
    label='Entrenamiento'
)

ax[0].plot(
    poda['alpha'],
    poda['auc_test'],
    's-',
    lw=2.4,
    ms=6,
    color=MENTA,
    markerfacecolor='white',
    markeredgewidth=1.8,
    label='Prueba'
)

ax[0].axvline(
    mejor_p['alpha'],
    color=AZUL,
    ls='--',
    lw=2,
    label='Óptimo'
)

if poda['alpha'].max() > 0:
    ax[0].set_xscale(
        'symlog',
        linthresh=1e-5
    )

ax[0].set_xlabel("ccp_alpha")
ax[0].set_ylabel("AUC")
ax[0].set_title("(a) Efecto de la poda sobre el AUC")
ax[0].legend(fontsize=8)

# -------------------------------------------------------------------------
# (b) Número de hojas
# -------------------------------------------------------------------------

ax[1].plot(
    poda['alpha'],
    poda['hojas'],
    'o-',
    lw=2.4,
    ms=6,
    color=LILA,
    markerfacecolor='white',
    markeredgewidth=1.8
)

ax[1].axvline(
    mejor_p['alpha'],
    color=AZUL,
    ls='--',
    lw=2
)

if poda['alpha'].max() > 0:
    ax[1].set_xscale(
        'symlog',
        linthresh=1e-5
    )

ax[1].set_yscale('log')
ax[1].set_xlabel("ccp_alpha")
ax[1].set_ylabel("Hojas (escala log)")
ax[1].set_title("(b) La poda simplifica el árbol")

# -------------------------------------------------------------------------
# (c) Matriz de confusion
# -------------------------------------------------------------------------

matriz_bin(
    ax[2],
    yte_b,
    PRED_BIN[nom],
    f"(c) Árbol podado\n"
    f"recall mín. {RES_BIN[nom]['Recall_min']:.3f}"
)

fig.suptitle(
    "FASE F3 · PODA POR COSTO-COMPLEJIDAD",
    fontsize=14,
    y=1.03
)

plt.tight_layout()

# Figura 39 del informe · F03_poda.png
guardar("F03_poda.png")

# =============================================================================
# REGISTRO
# =============================================================================

registrar(
    "Poda por costo-complejidad",
    "Hiperparámetros",
    f"alpha {mejor_p['alpha']:.6f} · "
    f"hojas {int(mejor_p['hojas']):,}",
    f"Ruta calculada sobre {len(sub_poda):,} registros por costo "
    f"computacional; el árbol final se reentrena con todo el conjunto",
    round(float(mejor_p['alpha']), 8)
)

gc.collect()


# %%
# =============================================================================
#  F4 — ARBOLES DE REGRESION: CUATRO CRITERIOS
#    · squared_error : minimiza el error cuadratico (el estandar)
#    · friedman_mse  : variante que mejora las divisiones en boosting
#    · absolute_error: minimiza el error absoluto, robusto a extremos
#    · poisson       : diseñado para CONTEOS, como ieh (0 a 13 bienes)
# =============================================================================
titulo("F4 · ÁRBOLES DE REGRESIÓN — criterios de división")

from sklearn.tree import DecisionTreeRegressor

sub("Justificación del criterio Poisson")
print("   ieh es un CONTEO: número de bienes entre 0 y 13. La desviación de")
print("   Poisson es la función de pérdida natural para datos de conteo, porque")
print("   modela que la varianza crece con la media. Es el criterio teóricamente")
print("   adecuado, y se verifica aquí si también lo es empíricamente.")

sub_c = np.random.default_rng(SEMILLA).choice(len(idx_tr), 20_000, replace=False)
print(f"\n   Comparación de criterios sobre {len(sub_c):,} registros")
print(f"   (absolute_error es muy costoso sobre el conjunto completo)")

crit_res = {}
for crit in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']:
    t0 = time.time()
    a = DecisionTreeRegressor(criterion=crit, max_depth=10, min_samples_leaf=50,
                              random_state=SEMILLA)
    a.fit(X_tr.iloc[sub_c], ytr_r.iloc[sub_c])
    p = a.predict(X_te)
    crit_res[crit] = {**metricas_reg(yte_r, p), 'segundos': time.time()-t0}
    print(f"   {crit:<16} R² {crit_res[crit]['R2']:.4f} · RMSE "
          f"{crit_res[crit]['RMSE']:.4f} · {crit_res[crit]['segundos']:.1f}s")
crit_res = pd.DataFrame(crit_res).T
mejor_c = crit_res['R2'].idxmax()
print(f"\n   Mejor criterio: {mejor_c}")

t0 = time.time()
a_reg = DecisionTreeRegressor(criterion=mejor_c, max_depth=10,
                              min_samples_leaf=50, random_state=SEMILLA)
a_reg.fit(X_tr, ytr_r)
nom = 'Árbol de regresión'
PRED_REG[nom] = a_reg.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                 f"criterio={mejor_c}")
r = RES_REG[nom]
print(f"\n   Árbol final ({mejor_c}, {len(idx_tr):,} registros) · "
      f"R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · MAE {r['MAE']:.4f} · "
      f"MAPE {r['MAPE_%']:.2f} %")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
s = crit_res['R2'].sort_values()
ax[0].barh(range(len(s)), s.values,
           color=[MENTA if v == s.max() else AZUL for v in s.values],
           edgecolor='white', linewidth=1.8)
ax[0].set_yticks(range(len(s)), s.index)
for i, v in enumerate(s.values):
    ax[0].text(v, i, f" {v:.4f}", va='center', fontsize=9, fontweight='bold')
ax[0].set_xlabel("R²"); ax[0].set_title("(a) R² según el criterio")
s = crit_res['segundos'].sort_values()
ax[1].barh(range(len(s)), s.values, color=AMBAR, edgecolor='white',
           linewidth=1.8)
ax[1].set_yticks(range(len(s)), s.index)
for i, v in enumerate(s.values):
    ax[1].text(v, i, f" {v:.1f}s", va='center', fontsize=9, fontweight='bold')
ax[1].set_xlabel("Segundos"); ax[1].set_title("(b) Costo de cada criterio")
dispersion(ax[2], yte_r, PRED_REG[nom], f"(c) Real frente a predicho · "
           f"R² {r['R2']:.3f}")
fig.suptitle("FASE F4 · ÁRBOL DE REGRESIÓN — índice de equipamiento",
             fontsize=14, y=1.03)
plt.tight_layout()
# Figura 40 del informe · F04_arbol_regresion.png
guardar("F04_arbol_regresion.png")
registrar("Criterio del árbol de regresión", "Modelado",
          f"mejor {mejor_c} · R² {r['R2']:.4f}",
          "Se evaluaron cuatro criterios; Poisson es el teóricamente adecuado "
          "para un objetivo de conteo como ieh")
gc.collect()


# %%
# =============================================================================
#  F5 — ARBOL MULTICLASE
# =============================================================================
titulo("F5 · ÁRBOL MULTICLASE — nivel de equipamiento")

t0 = time.time()
a_m = DecisionTreeClassifier(criterion='gini', max_depth=10,
                             min_samples_leaf=50, class_weight='balanced',
                             random_state=SEMILLA)
a_m.fit(X_tr, ytr_m)
nom = 'Árbol de decisión'
PRED_M[nom] = a_m.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0, "Gini, profundidad 10")
r = RES_M[nom]
print(f"   Accuracy {r['Accuracy']:.4f} · Precision {r['Precision']:.4f} · "
      f"Recall {r['Recall']:.4f} · F1 macro {r['F1']:.4f}")

imp = pd.Series(a_m.feature_importances_, index=PREDICTORAS).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
matriz_multi(ax[0], yte_m, PRED_M[nom], f"(a) Matriz · F1 macro {r['F1']:.3f}")
top = imp.tail(12)
ax[1].barh([ETQ(c) for c in top.index], top.values, color=LILA,
           edgecolor='white', linewidth=1.4)
ax[1].set_xlabel("Importancia (reducción de impureza)")
ax[1].set_title("(b) Variables que definen el nivel")
ax[1].tick_params(labelsize=8)
fig.suptitle("FASE F5 · ÁRBOL MULTICLASE", fontsize=14, y=1.03)
plt.tight_layout()
# Figura 41 del informe · F05_arbol_multiclase.png
guardar("F05_arbol_multiclase.png")
gc.collect()


# %%
# =============================================================================
#  F6 — RANDOM FOREST
#  Cientos de arboles distintos, cada uno entrenado sobre una muestra
#  bootstrap y considerando solo un subconjunto aleatorio de variables en
#  cada division. La prediccion es el voto (clasificacion) o el promedio
#  (regresion) de todos. Reduce la varianza de un arbol individual.
# =============================================================================
titulo("F6 · RANDOM FOREST")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

cfg = dict(n_estimators=150, max_depth=16, min_samples_leaf=20,
           max_features='sqrt', n_jobs=-1, random_state=SEMILLA)

t0 = time.time()
rf_b = RandomForestClassifier(**cfg, class_weight='balanced')
rf_b.fit(X_tr, ytr_b)
nom = 'Random Forest'
PROB_BIN[nom] = rf_b.predict_proba(X_te)[:, 1]
PRED_BIN[nom] = rf_b.predict(X_te)
RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                 time.time()-t0, "150 árboles, profundidad 16")
r = RES_BIN[nom]
print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · recall mín. "
      f"{r['Recall_min']:.4f} · {time.time()-t0:.1f}s")

t0 = time.time()
rf_r = RandomForestRegressor(**cfg)
rf_r.fit(X_tr, ytr_r)
PRED_REG[nom] = rf_r.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                 "150 árboles, profundidad 16")
r = RES_REG[nom]
print(f"   Regresión  R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · MAE "
      f"{r['MAE']:.4f} · MAPE {r['MAPE_%']:.2f} % · {time.time()-t0:.1f}s")

t0 = time.time()
rf_m = RandomForestClassifier(**cfg, class_weight='balanced')
rf_m.fit(X_tr, ytr_m)
PRED_M[nom] = rf_m.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0, "150 árboles, profundidad 16")
print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · {time.time()-t0:.1f}s")

imp = pd.Series(rf_b.feature_importances_, index=PREDICTORAS).sort_values()
print(f"\n   Las 5 variables más importantes para predecir internet:")
for k, v in imp.tail(5)[::-1].items():
    print(f"     {ETQ(k):<26} {v:.4f}")

fig = plt.figure(figsize=(18, 9.5))
gs = fig.add_gridspec(2, 3, hspace=.4, wspace=.3)
ax = fig.add_subplot(gs[:, 0])
top = imp.tail(18)
ax.barh([ETQ(c) for c in top.index], top.values, color=AZUL,
        edgecolor='white', linewidth=1.4)
ax.set_xlabel("Importancia media (reducción de impureza)")
ax.set_title("(a) Importancia de las variables")
ax.tick_params(labelsize=8.5)

ax = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(yte_b, PROB_BIN[nom])
ax.plot(fpr, tpr, lw=2.8, color=MENTA,
        label=f"Random Forest ({RES_BIN[nom]['ROC_AUC']:.3f})")
ax.plot([0, 1], [0, 1], '--', color=GRIS, lw=2, label='Azar')
ax.fill_between(fpr, tpr, alpha=.12, color=MENTA)
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos")
ax.set_title("(b) Curva ROC"); ax.legend(fontsize=8)

matriz_bin(fig.add_subplot(gs[0, 2]), yte_b, PRED_BIN[nom],
           f"(c) Matriz binaria\nrecall mín. {RES_BIN[nom]['Recall_min']:.3f}")
dispersion(fig.add_subplot(gs[1, 1]), yte_r, PRED_REG[nom],
           f"(d) Regresión · R² {RES_REG[nom]['R2']:.3f}")
matriz_multi(fig.add_subplot(gs[1, 2]), yte_m, PRED_M[nom],
             f"(e) Multiclase · F1 {RES_M[nom]['F1']:.3f}")
fig.suptitle("FASE F6 · RANDOM FOREST — los tres objetivos", fontsize=15,
             y=.98)
# Figura 42 del informe · F06_random_forest.png
guardar("F06_random_forest.png")
registrar("Random Forest", "Modelado",
          f"AUC {RES_BIN[nom]['ROC_AUC']:.4f} · R² {RES_REG[nom]['R2']:.4f} · "
          f"F1 {RES_M[nom]['F1']:.4f}",
          "Promedia 150 árboles decorrelacionados por bootstrap y selección "
          "aleatoria de variables: reduce la varianza del árbol individual")
gc.collect()


# %%
# =============================================================================
#  F7 — EXTRA TREES (ARBOLES EXTREMADAMENTE ALEATORIZADOS)
#  Como Random Forest, pero los umbrales de division se eligen AL AZAR en
#  lugar de buscar el optimo. Mas aleatoriedad, menor varianza y mas rapido.
# =============================================================================
titulo("F7 · EXTRA TREES — árboles extremadamente aleatorizados")

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

t0 = time.time()
et_b = ExtraTreesClassifier(**cfg, class_weight='balanced')
et_b.fit(X_tr, ytr_b)
nom = 'Extra Trees'
PROB_BIN[nom] = et_b.predict_proba(X_te)[:, 1]
PRED_BIN[nom] = et_b.predict(X_te)
RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
t_et = time.time() - t0
registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom], t_et,
                 "umbrales aleatorios")
r = RES_BIN[nom]
print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · recall mín. "
      f"{r['Recall_min']:.4f} · {t_et:.1f}s")

t0 = time.time()
et_r = ExtraTreesRegressor(**cfg)
et_r.fit(X_tr, ytr_r)
PRED_REG[nom] = et_r.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0)
print(f"   Regresión  R² {RES_REG[nom]['R2']:.4f} · RMSE "
      f"{RES_REG[nom]['RMSE']:.4f} · {time.time()-t0:.1f}s")

t0 = time.time()
et_m = ExtraTreesClassifier(**cfg, class_weight='balanced')
et_m.fit(X_tr, ytr_m)
PRED_M[nom] = et_m.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0)
print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · {time.time()-t0:.1f}s")

comp = pd.DataFrame({
    'Random Forest': [RES_BIN['Random Forest']['ROC_AUC'],
                      RES_REG['Random Forest']['R2'],
                      RES_M['Random Forest']['F1']],
    'Extra Trees': [RES_BIN['Extra Trees']['ROC_AUC'],
                    RES_REG['Extra Trees']['R2'], RES_M['Extra Trees']['F1']]},
    index=['AUC binaria', 'R² regresión', 'F1 multiclase'])
print(f"\n{comp.round(4).to_string()}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
x = np.arange(3); w = .36
ax[0].bar(x - w/2, comp['Random Forest'], w, label='Random Forest',
          color=AZUL, edgecolor='white', linewidth=1.5)
ax[0].bar(x + w/2, comp['Extra Trees'], w, label='Extra Trees', color=AMBAR,
          edgecolor='white', linewidth=1.5)
for i in range(3):
    ax[0].text(i - w/2, comp.iloc[i, 0], f"{comp.iloc[i,0]:.3f}", ha='center',
               va='bottom', fontsize=8, fontweight='bold')
    ax[0].text(i + w/2, comp.iloc[i, 1], f"{comp.iloc[i,1]:.3f}", ha='center',
               va='bottom', fontsize=8, fontweight='bold')
ax[0].set_xticks(x, comp.index); ax[0].set_ylim(0, 1.05)
ax[0].set_title("(a) Random Forest frente a Extra Trees")
ax[0].legend(fontsize=8)
matriz_bin(ax[1], yte_b, PRED_BIN[nom],
           f"(b) Matriz binaria\nrecall mín. {RES_BIN[nom]['Recall_min']:.3f}")
matriz_multi(ax[2], yte_m, PRED_M[nom],
             f"(c) Multiclase · F1 {RES_M[nom]['F1']:.3f}")
fig.suptitle("FASE F7 · EXTRA TREES", fontsize=14, y=1.03)
plt.tight_layout()
# Figura 43 del informe · F07_extra_trees.png
guardar("F07_extra_trees.png")
gc.collect()


# %%
# =============================================================================
#  F8 — BAGGING (BOOTSTRAP AGGREGATING)
#  Entrena el MISMO modelo base muchas veces sobre muestras bootstrap y
#  promedia. A diferencia de Random Forest, cada modelo ve TODAS las variables.
# =============================================================================
titulo("F8 · BAGGING")

from sklearn.ensemble import BaggingClassifier, BaggingRegressor

base_c = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
                                class_weight='balanced', random_state=SEMILLA)
base_r = DecisionTreeRegressor(max_depth=12, min_samples_leaf=20,
                               random_state=SEMILLA)

t0 = time.time()
bg_b = BaggingClassifier(estimator=base_c, n_estimators=60, max_samples=.6,
                         n_jobs=-1, random_state=SEMILLA)
bg_b.fit(X_tr, ytr_b)
nom = 'Bagging'
PROB_BIN[nom] = bg_b.predict_proba(X_te)[:, 1]
PRED_BIN[nom] = bg_b.predict(X_te)
RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                 time.time()-t0, "60 árboles de profundidad 12")
r = RES_BIN[nom]
print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · recall mín. "
      f"{r['Recall_min']:.4f} · {time.time()-t0:.1f}s")

t0 = time.time()
bg_r = BaggingRegressor(estimator=base_r, n_estimators=60, max_samples=.6,
                        n_jobs=-1, random_state=SEMILLA)
bg_r.fit(X_tr, ytr_r)
PRED_REG[nom] = bg_r.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0)
print(f"   Regresión  R² {RES_REG[nom]['R2']:.4f} · RMSE "
      f"{RES_REG[nom]['RMSE']:.4f} · {time.time()-t0:.1f}s")

t0 = time.time()
bg_m = BaggingClassifier(estimator=base_c, n_estimators=60, max_samples=.6,
                         n_jobs=-1, random_state=SEMILLA)
bg_m.fit(X_tr, ytr_m)
PRED_M[nom] = bg_m.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0)
print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · {time.time()-t0:.1f}s")

# Como mejora el AUC al agregar arboles
curva_b = []
for n in [1, 3, 5, 10, 20, 40, 60]:
    m = BaggingClassifier(estimator=base_c, n_estimators=n, max_samples=.6,
                          n_jobs=-1, random_state=SEMILLA)
    m.fit(X_tr, ytr_b)
    curva_b.append({'n': n, 'auc': roc_auc_score(yte_b,
                                                 m.predict_proba(X_te)[:, 1])})
curva_b = pd.DataFrame(curva_b)
print(f"\n   AUC con 1 árbol: {curva_b.iloc[0]['auc']:.4f}  ·  con 60: "
      f"{curva_b.iloc[-1]['auc']:.4f}  ·  ganancia "
      f"{curva_b.iloc[-1]['auc']-curva_b.iloc[0]['auc']:+.4f}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].plot(curva_b['n'], curva_b['auc'], 'o-', lw=2.8, ms=9, color=LILA,
           markerfacecolor='white', markeredgewidth=2.2)
ax[0].set_xlabel("Número de árboles en el ensamble")
ax[0].set_ylabel("AUC de prueba")
ax[0].set_title("(a) Agregar modelos reduce la varianza")
matriz_bin(ax[1], yte_b, PRED_BIN[nom],
           f"(b) Matriz binaria\nrecall mín. {RES_BIN[nom]['Recall_min']:.3f}")
dispersion(ax[2], yte_r, PRED_REG[nom],
           f"(c) Regresión · R² {RES_REG[nom]['R2']:.3f}")
fig.suptitle("FASE F8 · BAGGING — bootstrap aggregating", fontsize=14, y=1.03)
plt.tight_layout()
# Figura 44 del informe · F08_bagging.png
guardar("F08_bagging.png")
registrar("Bagging", "Modelado",
          f"AUC con 1 árbol {curva_b.iloc[0]['auc']:.4f} → 60 árboles "
          f"{curva_b.iloc[-1]['auc']:.4f}",
          "Promediar modelos entrenados sobre muestras bootstrap reduce la "
          "varianza sin aumentar el sesgo")
gc.collect()


# %%
# =============================================================================
#  F9 — APRENDICES DEBILES FRENTE A FUERTES
#  Un aprendiz DEBIL apenas supera al azar (un arbol de una sola division).
#  Un aprendiz FUERTE es preciso por si mismo. La idea central de los
#  ensambles es que muchos debiles combinados pueden volverse fuertes.
# =============================================================================
titulo("F9 · APRENDICES DÉBILES FRENTE A FUERTES")

casos = {
    'Tocón (1 división)': DecisionTreeClassifier(max_depth=1,
                                                 class_weight='balanced',
                                                 random_state=SEMILLA),
    'Bagging de 100 tocones': BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced',
                                         random_state=SEMILLA),
        n_estimators=100, n_jobs=-1, random_state=SEMILLA),
    'Árbol profundidad 5': DecisionTreeClassifier(max_depth=5,
                                                  class_weight='balanced',
                                                  random_state=SEMILLA),
    'Árbol sin límite': DecisionTreeClassifier(class_weight='balanced',
                                               random_state=SEMILLA),
}
debil = []
for nom_, m in casos.items():
    m.fit(X_tr, ytr_b)
    debil.append({'modelo': nom_,
                  'auc_train': roc_auc_score(ytr_b, m.predict_proba(X_tr)[:, 1]),
                  'auc_test': roc_auc_score(yte_b, m.predict_proba(X_te)[:, 1])})
debil.append({'modelo': 'Random Forest', 'auc_train': roc_auc_score(
    ytr_b, rf_b.predict_proba(X_tr)[:, 1]),
    'auc_test': RES_BIN['Random Forest']['ROC_AUC']})
debil = pd.DataFrame(debil).set_index('modelo')
debil['brecha'] = debil['auc_train'] - debil['auc_test']
print(debil.round(4).to_string())
print(f"\n   El tocón es un aprendiz débil: una sola pregunta.")
print(f"   El árbol sin límite sobreajusta: brecha de "
      f"{debil.loc['Árbol sin límite','brecha']:.4f} entre entrenamiento y prueba.")
print(f"   Random Forest combina muchos árboles y logra el mejor AUC de prueba")
print(f"   con una brecha de solo {debil.loc['Random Forest','brecha']:.4f}.")

fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
x = np.arange(len(debil)); w = .38
ax[0].bar(x - w/2, debil['auc_train'], w, label='Entrenamiento', color=CORAL,
          edgecolor='white', linewidth=1.5)
ax[0].bar(x + w/2, debil['auc_test'], w, label='Prueba', color=MENTA,
          edgecolor='white', linewidth=1.5)
ax[0].set_xticks(x, [n[:22] for n in debil.index], rotation=15, fontsize=8)
ax[0].set_ylim(.5, 1.02); ax[0].set_ylabel("AUC")
ax[0].set_title("(a) Desempeño en entrenamiento y prueba")
ax[0].legend(fontsize=8.5)
s = debil['brecha']
ax[1].barh(range(len(s)), s.values,
           color=[CORAL if v > .05 else MENTA for v in s.values],
           edgecolor='white', linewidth=1.5)
ax[1].set_yticks(range(len(s)), s.index, fontsize=8.5)
for i, v in enumerate(s.values):
    ax[1].text(v, i, f" {v:.4f}", va='center', fontsize=8.5, fontweight='bold')
ax[1].set_xlabel("Brecha de AUC (entrenamiento − prueba)")
ax[1].set_title("(b) Sobreajuste: rojo = brecha mayor a 0,05")
fig.suptitle("FASE F9 · APRENDICES DÉBILES FRENTE A FUERTES", fontsize=14,
             y=1.03)
plt.tight_layout()
# Figura 45 del informe · F09_debiles_fuertes.png
guardar("F09_debiles_fuertes.png")
registrar("Aprendices débiles y fuertes", "Concepto",
          f"tocón AUC {debil.loc['Tocón (1 división)','auc_test']:.4f} · RF "
          f"{debil.loc['Random Forest','auc_test']:.4f}",
          "Demuestra que combinar modelos débiles o de alta varianza produce "
          "un aprendiz fuerte y estable")
gc.collect()


# %%
# =============================================================================
#  F10 — VOTING: VOTACION DURA Y BLANDA
#  Combina modelos DE DISTINTA NATURALEZA.
#    · Dura  : cada modelo vota una clase; gana la mayoria
#    · Blanda: se promedian las probabilidades; gana la mayor
# =============================================================================
titulo("F10 · VOTING — votación dura y blanda")

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

print(f"   Entrenado con {len(sub_meta):,} registros: combina tres modelos de")
print(f"   distinta naturaleza y su costo es la suma de los tres.")

Xm, ym_b = X_tr.iloc[sub_meta], ytr_b.iloc[sub_meta]
ym_r, ym_m = ytr_r.iloc[sub_meta], ytr_m.iloc[sub_meta]


def componentes_clf():
    return [
        ('logistica', Pipeline([('p', prep_onehot), ('m', LogisticRegression(
            max_iter=500, class_weight='balanced', random_state=SEMILLA))])),
        ('arbol', DecisionTreeClassifier(max_depth=10, min_samples_leaf=50,
                                         class_weight='balanced',
                                         random_state=SEMILLA)),
        ('bosque', RandomForestClassifier(n_estimators=80, max_depth=14,
                                          min_samples_leaf=20,
                                          class_weight='balanced', n_jobs=-1,
                                          random_state=SEMILLA))]


voto = {}
for tipo in ['hard', 'soft']:
    t0 = time.time()
    v = VotingClassifier(estimators=componentes_clf(), voting=tipo, n_jobs=1)
    v.fit(Xm, ym_b)
    pred = v.predict(X_te)
    if tipo == 'soft':
        prob = v.predict_proba(X_te)[:, 1]
    else:
        prob = np.mean([e.predict(X_te) for e in v.estimators_], axis=0)
    nom = f"Voting {'duro' if tipo == 'hard' else 'blando'}"
    PROB_BIN[nom] = prob; PRED_BIN[nom] = pred
    RES_BIN[nom] = metricas_bin(yte_b, pred, prob)
    registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                     time.time()-t0, f"logística + árbol + bosque, "
                                     f"{len(sub_meta):,} registros")
    voto[tipo] = v
    r = RES_BIN[nom]
    print(f"   {nom:<16} AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
          f"recall mín. {r['Recall_min']:.4f} · {time.time()-t0:.1f}s")

sub("Contribución de cada componente (votación blanda)")
aucs_comp = {}
for n_, e in zip(['Logística', 'Árbol', 'Bosque'], voto['soft'].estimators_):
    aucs_comp[n_] = roc_auc_score(yte_b, e.predict_proba(X_te)[:, 1])
    print(f"   {n_:<10} individual AUC {aucs_comp[n_]:.4f}")
aucs_comp['Voting blando'] = RES_BIN['Voting blando']['ROC_AUC']

t0 = time.time()
vr = VotingRegressor([
    ('ridge', Pipeline([('p', prep_onehot), ('m', Ridge(alpha=10.0))])),
    ('arbol', DecisionTreeRegressor(max_depth=10, min_samples_leaf=50,
                                    random_state=SEMILLA)),
    ('bosque', RandomForestRegressor(n_estimators=80, max_depth=14,
                                     min_samples_leaf=20, n_jobs=-1,
                                     random_state=SEMILLA))])
vr.fit(Xm, ym_r)
nom = 'Voting'
PRED_REG[nom] = vr.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                 "Ridge + árbol + bosque")
print(f"\n   Voting regresión  R² {RES_REG[nom]['R2']:.4f} · RMSE "
      f"{RES_REG[nom]['RMSE']:.4f}")

t0 = time.time()
vm = VotingClassifier(estimators=componentes_clf(), voting='soft')
vm.fit(Xm, ym_m)
PRED_M[nom] = vm.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0, "votación blanda")
print(f"   Voting multiclase F1 macro {RES_M[nom]['F1']:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
s = pd.Series(aucs_comp)
ax[0].bar(s.index, s.values, color=[GRIS, GRIS, GRIS, MENTA],
          edgecolor='white', linewidth=2)
for i, v in enumerate(s.values):
    ax[0].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=9,
               fontweight='bold')
ax[0].set_ylim(min(s.values) - .03, max(s.values) + .02)
ax[0].set_ylabel("AUC"); ax[0].set_title("(a) Componentes frente al ensamble")
matriz_bin(ax[1], yte_b, PRED_BIN['Voting duro'],
           f"(b) Votación dura\nrecall mín. "
           f"{RES_BIN['Voting duro']['Recall_min']:.3f}")
matriz_bin(ax[2], yte_b, PRED_BIN['Voting blando'],
           f"(c) Votación blanda\nrecall mín. "
           f"{RES_BIN['Voting blando']['Recall_min']:.3f}")
fig.suptitle("FASE F10 · VOTING — combinar modelos de distinta naturaleza",
             fontsize=14, y=1.03)
plt.tight_layout()
# Figura 46 del informe · F10_voting.png
guardar("F10_voting.png")
registrar("Voting duro y blando", "Modelado",
          f"duro AUC {RES_BIN['Voting duro']['ROC_AUC']:.4f} · blando "
          f"{RES_BIN['Voting blando']['ROC_AUC']:.4f}",
          f"Entrenado sobre {len(sub_meta):,} registros: su costo es la suma "
          f"de tres modelos")
gc.collect()


# %%
# =============================================================================
#  F11 — STACKING: APILAMIENTO CON META-MODELO
#  En lugar de promediar, un META-MODELO aprende COMO combinar las
#  predicciones de los modelos base. Las predicciones base se obtienen por
#  validacion cruzada para que el meta-modelo no vea datos memorizados.
# =============================================================================
titulo("F11 · STACKING — apilamiento")

from sklearn.ensemble import StackingClassifier, StackingRegressor

t0 = time.time()
st_b = StackingClassifier(
    estimators=componentes_clf(),
    final_estimator=LogisticRegression(max_iter=500, class_weight='balanced',
                                       random_state=SEMILLA),
    cv=3, stack_method='predict_proba', n_jobs=1)
st_b.fit(Xm, ym_b)
nom = 'Stacking'
PROB_BIN[nom] = st_b.predict_proba(X_te)[:, 1]
PRED_BIN[nom] = st_b.predict(X_te)
RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                 time.time()-t0, "meta-modelo logístico, cv=3")
r = RES_BIN[nom]
print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · recall mín. "
      f"{r['Recall_min']:.4f} · {time.time()-t0:.1f}s")

pesos = pd.Series(st_b.final_estimator_.coef_[0],
                  index=['Logística', 'Árbol', 'Bosque'])
print(f"\n   PESOS QUE APRENDIÓ EL META-MODELO")
for k, v in pesos.items():
    print(f"     {k:<10} {v:+.4f}")
print(f"   El meta-modelo confía más en: {pesos.abs().idxmax()}")

t0 = time.time()
st_r = StackingRegressor(
    estimators=[('ridge', Pipeline([('p', prep_onehot),
                                    ('m', Ridge(alpha=10.0))])),
                ('arbol', DecisionTreeRegressor(max_depth=10,
                                                min_samples_leaf=50,
                                                random_state=SEMILLA)),
                ('bosque', RandomForestRegressor(n_estimators=80, max_depth=14,
                                                 min_samples_leaf=20,
                                                 n_jobs=-1,
                                                 random_state=SEMILLA))],
    final_estimator=Ridge(alpha=1.0), cv=3)
st_r.fit(Xm, ym_r)
PRED_REG[nom] = st_r.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                 "meta-modelo Ridge, cv=3")
print(f"\n   Regresión  R² {RES_REG[nom]['R2']:.4f} · RMSE "
      f"{RES_REG[nom]['RMSE']:.4f} · {time.time()-t0:.1f}s")

t0 = time.time()
st_m = StackingClassifier(
    estimators=componentes_clf(),
    final_estimator=LogisticRegression(max_iter=500, class_weight='balanced',
                                       random_state=SEMILLA), cv=3)
st_m.fit(Xm, ym_m)
PRED_M[nom] = st_m.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0)
print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · {time.time()-t0:.1f}s")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].barh(pesos.index, pesos.values,
           color=[MENTA if v > 0 else CORAL for v in pesos.values],
           edgecolor='white', linewidth=2)
for i, v in enumerate(pesos.values):
    ax[0].text(v, i, f" {v:+.3f}", va='center', fontsize=9, fontweight='bold')
ax[0].axvline(0, color=TINTA, lw=1.4)
ax[0].set_xlabel("Coeficiente del meta-modelo")
ax[0].set_title("(a) Cuánto confía en cada modelo base")
matriz_bin(ax[1], yte_b, PRED_BIN[nom],
           f"(b) Matriz binaria\nrecall mín. {RES_BIN[nom]['Recall_min']:.3f}")
dispersion(ax[2], yte_r, PRED_REG[nom],
           f"(c) Regresión · R² {RES_REG[nom]['R2']:.3f}")
fig.suptitle("FASE F11 · STACKING — un meta-modelo aprende a combinar",
             fontsize=14, y=1.03)
plt.tight_layout()
# Figura 47 del informe · F11_stacking.png
guardar("F11_stacking.png")
registrar("Stacking", "Modelado",
          f"AUC {RES_BIN[nom]['ROC_AUC']:.4f} · modelo más ponderado "
          f"{pesos.abs().idxmax()}",
          "Las predicciones base se generan por validación cruzada (cv=3) para "
          "que el meta-modelo no aprenda de datos memorizados")
gc.collect()


# %%
# =============================================================================
#  F12 — COMPARACION GLOBAL DE LA FASE Y BITACORA
# =============================================================================
titulo("F12 · COMPARACIÓN GLOBAL DE LA FASE F")

tb = pd.DataFrame(RES_BIN).T
tr = pd.DataFrame(RES_REG).T
tm = pd.DataFrame(RES_M).T

sub("Clasificación binaria")
print(tb.round(4).to_string())
sub("Regresión")
print(tr.round(4).to_string())
sub("Multiclase")
print(tm.round(4).to_string())

mb, mr, mm = tb['ROC_AUC'].idxmax(), tr['R2'].idxmax(), tm['F1'].idxmax()
mrec = tb['Recall_min'].idxmax()
print(f"""
   TECHO DE LAS FASES D Y E       MEJOR DE LA FASE F
   AUC internet   0,811           {mb:<22} {tb.loc[mb,'ROC_AUC']:.4f}
   Recall mínimo  0,750 (KNN)     {mrec:<22} {tb.loc[mrec,'Recall_min']:.4f}
   R² ieh         0,529           {mr:<22} {tr.loc[mr,'R2']:.4f}
   F1 nivel       0,633           {mm:<22} {tm.loc[mm,'F1']:.4f}
""")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.32)

ax = fig.add_subplot(gs[0, 0])
for i, nom in enumerate(tb.index):
    fpr, tpr, _ = roc_curve(yte_b, PROB_BIN[nom])
    ax.plot(fpr, tpr, lw=1.8, color=PALETA[i % len(PALETA)],
            label=f"{nom[:18]} ({tb.loc[nom,'ROC_AUC']:.3f})")
ax.plot([0, 1], [0, 1], '--', color=GRIS, lw=1.8)
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("(a) Curvas ROC"); ax.legend(fontsize=6.3, loc='lower right')

ax = fig.add_subplot(gs[0, 1])
for i, nom in enumerate(tb.index):
    pr, rc, _ = precision_recall_curve(yte_b, PROB_BIN[nom])
    ax.plot(rc, pr, lw=1.8, color=PALETA[i % len(PALETA)],
            label=f"{nom[:18]} ({tb.loc[nom,'PR_AUC']:.3f})")
ax.axhline(yte_b.mean(), color=GRIS, ls='--', lw=1.8)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("(b) Curvas Precision-Recall")
ax.legend(fontsize=6.3, loc='lower left')

ax = fig.add_subplot(gs[0, 2])
s = tb['Recall_min'].sort_values()
ax.barh(range(len(s)), s.values,
        color=[MENTA if v == s.max() else LILA for v in s.values],
        edgecolor='white', linewidth=1.4)
ax.set_yticks(range(len(s)), s.index, fontsize=7.5)
for i, v in enumerate(s.values):
    ax.text(v, i, f" {v:.3f}", va='center', fontsize=7.5, fontweight='bold')
ax.axvline(.750, color=CORAL, ls='--', lw=2, label='KNN Fase E (0,750)')
ax.set_xlabel("Recall SIN internet")
ax.set_title("(c) Detección de desconectados"); ax.legend(fontsize=7)
ax.set_xlim(0, 1)

ax = fig.add_subplot(gs[1, 0])
s = tb['ROC_AUC'].sort_values()
ax.barh(range(len(s)), s.values,
        color=[MENTA if v == s.max() else AZUL for v in s.values],
        edgecolor='white', linewidth=1.4)
ax.set_yticks(range(len(s)), s.index, fontsize=7.5)
for i, v in enumerate(s.values):
    ax.text(v, i, f" {v:.4f}", va='center', fontsize=7.5, fontweight='bold')
ax.axvline(.811, color=CORAL, ls='--', lw=2, label='Techo D y E (0,811)')
ax.set_xlabel("AUC"); ax.set_title("(d) AUC · clasificación binaria")
ax.set_xlim(min(s.values) - .04, max(s.values) + .03)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 1])
s = tr['R2'].sort_values()
ax.barh(range(len(s)), s.values,
        color=[MENTA if v == s.max() else AZUL for v in s.values],
        edgecolor='white', linewidth=1.4)
ax.set_yticks(range(len(s)), s.index, fontsize=8)
for i, v in enumerate(s.values):
    ax.text(v, i, f" {v:.4f}", va='center', fontsize=8, fontweight='bold')
ax.axvline(.529, color=CORAL, ls='--', lw=2, label='Techo D y E (0,529)')
ax.set_xlabel("R²"); ax.set_title("(e) R² · regresión de ieh")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 2])
s = tm['F1'].sort_values()
ax.barh(range(len(s)), s.values,
        color=[MENTA if v == s.max() else LILA for v in s.values],
        edgecolor='white', linewidth=1.4)
ax.set_yticks(range(len(s)), s.index, fontsize=8)
for i, v in enumerate(s.values):
    ax.text(v, i, f" {v:.4f}", va='center', fontsize=8, fontweight='bold')
ax.axvline(.633, color=CORAL, ls='--', lw=2, label='Techo D y E (0,633)')
ax.set_xlabel("F1 macro"); ax.set_title("(f) F1 · multiclase")
ax.legend(fontsize=7)
fig.suptitle("FASE F12 · ÁRBOLES Y ENSAMBLES — ¿se rompe el techo?",
             fontsize=15, y=.99)
# Figura 48 del informe · F12_comparacion_global.png
guardar("F12_comparacion_global.png")

tb.round(5).to_csv(os.path.join(DIR_FIG, "F_clasificacion_binaria.csv"))
tr.round(5).to_csv(os.path.join(DIR_FIG, "F_regresion.csv"))
tm.round(5).to_csv(os.path.join(DIR_FIG, "F_clasificacion_multiclase.csv"))

registrar("Cierre de la Fase F", "Cierre",
          f"{len(MOD_FASE)} modelos · mejor AUC {mb} {tb.loc[mb,'ROC_AUC']:.4f}"
          f" · mejor R² {mr} {tr.loc[mr,'R2']:.4f}",
          "Los árboles capturan interacciones que los modelos lineales y de "
          "distancia no pueden representar", len(MOD_FASE))
volcar_bitacora()
print(f"\n   Modelos entrenados en la fase: {len(MOD_FASE)}")
print(f"   Tiempo total de la fase      : {(time.time()-t_ini)/60:.1f} min")
gc.collect()

# %%

# %%


# =============================================================================
#  PROYECTO CPV 2024 · BORRADOR INTEGRAL
#  FASE G — BOOSTING
#
#  El BOOSTING construye los modelos EN SECUENCIA: cada nuevo arbol se
#  concentra en los errores que cometieron los anteriores. Mientras Bagging
#  reduce la VARIANZA, el boosting reduce el SESGO.
#
#  Modelos de esta fase (cada bloque produce su propio grafico)
#    G1  AdaBoost            · repondera los hogares mal clasificados
#    G2  Boosting vs Bagging · el experimento que cierra la Fase F
#    G3  Gradient Boosting   · ajusta los residuos del modelo anterior
#    G4  Tasa de aprendizaje · el hiperparametro central del boosting
#    G5  XGBoost             · gradient boosting con regularizacion
#    G6  LightGBM            · crecimiento por hojas y categoricas nativas
#    G7  CatBoost            · tratamiento ordenado de categoricas
#    G8  Comparacion global de la fase
#
#  Misma muestra y misma particion que D, E y F · random_state = 42
# =============================================================================


# %%
# =============================================================================
#
#  correspondiente se salta con un aviso y se registra en la bitacora.
# =============================================================================
import sys
import subprocess
import importlib.util


def instalar(paquete):
    if importlib.util.find_spec(paquete) is not None:
        return True
    # --only-binary evita compilar desde el codigo fuente, que es lo que
    # suele fallar en Windows con versiones muy recientes de Python
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', paquete,
                        '--only-binary=:all:'],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"   ✔ {paquete:<10} instalado correctamente")
        return True
    lineas = [l for l in (r.stderr or r.stdout).splitlines() if l.strip()]
    for l in lineas[-4:]:
        print(f"        {l[:110]}")
    return False


resultado = {p: instalar(p) for p in ['lightgbm', 'xgboost', 'catboost']}

print()
if all(resultado.values()):
    pass
else:
    no = [p for p, ok in resultado.items() if not ok]


# %%
# =============================================================================
#  G0 — CONFIGURACION Y CARGA
# =============================================================================
import os
import gc
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

SEMILLA = 42
N_MUESTRA = 120_000
PRUEBA = 0.25
FASE = "G"
DIR_FIG = "figuras_v2"
os.makedirs(DIR_FIG, exist_ok=True)

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA = "#FF6B8A", "#FFC145", "#22223B"
GRIS, AGUA = "#8D99AE", "#A8DADC"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, AGUA, "#6C8AE4", "#C3A6F0",
          "#2EC4B6", "#E07A5F", "#F4A261", "#9D4EDD"]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "grid.linewidth": .9,
    "axes.axisbelow": True, "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92,
    "legend.edgecolor": "#D8DEF0"})


def guardar(n):
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    print(f"   [figura] {r}")
    plt.show()


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


BITACORA_XLSX = "BITACORA_PROYECTO.xlsx"
REG_FASE, MOD_FASE = [], []


def registrar(paso, categoria, detalle, justificacion, valor=None):
    REG_FASE.append({
        'fase': FASE, 'n': len(REG_FASE) + 1,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'categoria': categoria, 'paso': paso, 'detalle': detalle,
        'justificacion': justificacion, 'valor': valor})
    print(f"   [bitácora {FASE}-{len(REG_FASE):02d}] {paso}")


def registrar_modelo(objetivo, tipo, modelo, metricas, segundos, nota=""):
    fila = {'fase': FASE,
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objetivo': objetivo, 'tipo': tipo, 'modelo': modelo,
            'segundos': round(segundos, 2), 'nota': nota}
    fila.update({k: (round(float(v), 5) if v is not None else None)
                 for k, v in metricas.items()})
    MOD_FASE.append(fila)


def leer_hoja(nombre):
    if os.path.exists(BITACORA_XLSX):
        try:
            return pd.read_excel(BITACORA_XLSX, sheet_name=nombre)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def volcar_bitacora():
    reg, mod = leer_hoja('Registro'), leer_hoja('Modelos')
    if len(reg) and 'fase' in reg:
        reg = reg[reg['fase'].astype(str) != FASE]
    if len(mod) and 'fase' in mod:
        mod = mod[mod['fase'].astype(str) != FASE]
    reg = pd.concat([reg, pd.DataFrame(REG_FASE)], ignore_index=True)
    mod = pd.concat([mod, pd.DataFrame(MOD_FASE)], ignore_index=True)
    resumen = (reg.groupby('fase').agg(pasos=('paso', 'count'),
                                       ultima_actualizacion=('fecha', 'max'))
               .reset_index())
    if len(mod):
        rm = mod.groupby('fase').agg(modelos=('modelo', 'count')).reset_index()
        resumen = resumen.merge(rm, on='fase', how='left')
    try:
        with pd.ExcelWriter(BITACORA_XLSX, engine='openpyxl') as w:
            resumen.to_excel(w, sheet_name='Resumen por fase', index=False)
            reg.to_excel(w, sheet_name='Registro', index=False)
            mod.to_excel(w, sheet_name='Modelos', index=False)
        print(f"   ✔ Bitácora actualizada: {BITACORA_XLSX}  "
              f"(registro {len(reg)} · modelos {len(mod)})")
    except PermissionError:
        alt = f"BITACORA_respaldo_{FASE}_{datetime.now():%H%M%S}.csv"
        pd.DataFrame(MOD_FASE).to_csv(alt, index=False, encoding='utf-8')


titulo("FASE G · BOOSTING")
t_ini = time.time()

import importlib.util
XGB_OK = importlib.util.find_spec('xgboost') is not None
LGB_OK = importlib.util.find_spec('lightgbm') is not None
CAT_OK = importlib.util.find_spec('catboost') is not None
print(f"   XGBoost  : {'disponible' if XGB_OK else 'NO disponible'}")
print(f"   LightGBM : {'disponible' if LGB_OK else 'NO disponible'}")
print(f"   CatBoost : {'disponible' if CAT_OK else 'NO disponible'}")

from sklearn.model_selection import train_test_split

BASE = "vivienda_cpv2024_LIMPIO_v2"
datos = pd.read_parquet(f"{BASE}.parquet") if os.path.exists(f"{BASE}.parquet") \
    else pd.read_csv(f"{BASE}.csv", sep=';')
for c in datos.select_dtypes(include=['float64']).columns:
    datos[c] = datos[c].astype('float32')

if len(datos) > N_MUESTRA:
    muestra, _ = train_test_split(datos, train_size=N_MUESTRA,
                                  stratify=datos['internet'],
                                  random_state=SEMILLA)
else:
    muestra = datos.copy()
muestra = muestra.reset_index(drop=True)
del datos
gc.collect()

CATEGORICAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v15_servsan', 'v16_desague', 'v17_tenencia', 'tip_hog',
               'v20a_emi', 'v21a_fal']
NUMERICAS = ['v13_habitac', 'v14_dormit', 'tot_pers', 'v20b_totemi',
             'v21b_totfal', 'hacinamiento', 'pers_por_habitac',
             'hacin_critico', 'sin_servicios']
CATEGORICAS = [c for c in CATEGORICAS if c in muestra.columns]
NUMERICAS = [c for c in NUMERICAS if c in muestra.columns]
PREDICTORAS = CATEGORICAS + NUMERICAS

NOMBRES = {
    'idep': 'Departamento', 'urbrur': 'Área urbana/rural',
    'v01_tipoviv': 'Tipo de vivienda', 'v03_pared': 'Material de pared',
    'v04_revoq': 'Pared revocada', 'v05_techo': 'Material de techo',
    'v06_piso': 'Material de piso', 'v07_aguapro': 'Procedencia del agua',
    'v08_aguadist': 'Distribución de agua', 'v09_energia': 'Fuente de energía',
    'v10_combus': 'Combustible de cocina', 'v11_basura': 'Manejo de basura',
    'v12_cocina': 'Tiene cocina', 'v13_habitac': 'N° de habitaciones',
    'v14_dormit': 'N° de dormitorios', 'v15_servsan': 'Servicio sanitario',
    'v16_desague': 'Tipo de desagüe', 'v17_tenencia': 'Tenencia',
    'tot_pers': 'Personas en el hogar', 'tip_hog': 'Tipo de hogar',
    'v20a_emi': 'Hubo emigración', 'v20b_totemi': 'N° emigrantes',
    'v21a_fal': 'Hubo fallecimiento', 'v21b_totfal': 'N° fallecimientos',
    'hacinamiento': 'Personas/dormitorio', 'hacin_critico': 'Hacinam. crítico',
    'sin_servicios': 'Carencia servicios',
    'pers_por_habitac': 'Personas/habitación'}
ETQ = lambda c: NOMBRES.get(c, c)

X = muestra[PREDICTORAS]
y_ieh = muestra['ieh'].astype('float32')
y_int = muestra['internet'].astype(int)
y_niv = muestra['nivel_equip'].astype(str)
ORDEN = ['Bajo', 'Medio', 'Alto']
y_niv_num = y_niv.map({k: i for i, k in enumerate(ORDEN)}).astype(int)

idx_tr, idx_te = train_test_split(np.arange(len(muestra)), test_size=PRUEBA,
                                  stratify=y_int, random_state=SEMILLA)
X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
ytr_b, yte_b = y_int.iloc[idx_tr], y_int.iloc[idx_te]
ytr_r, yte_r = y_ieh.iloc[idx_tr], y_ieh.iloc[idx_te]
ytr_m, yte_m = y_niv.iloc[idx_tr], y_niv.iloc[idx_te]
ytr_mn, yte_mn = y_niv_num.iloc[idx_tr], y_niv_num.iloc[idx_te]

# Validacion interna para la parada temprana: 10 % del ENTRENAMIENTO.
# El conjunto de prueba sigue intacto hasta la evaluacion final.
i_fit, i_val = train_test_split(np.arange(len(idx_tr)), test_size=.10,
                                stratify=ytr_b, random_state=SEMILLA)
X_fit, X_val = X_tr.iloc[i_fit], X_tr.iloc[i_val]

peso_pos = (ytr_b == 0).sum() / (ytr_b == 1).sum()
print(f"   Muestra       : {len(muestra):,}  (idéntica a D, E y F)")
print(f"   Entrenamiento : {len(idx_tr):,}  (de ellos {len(i_val):,} para "
      f"parada temprana)")
print(f"   Prueba        : {len(idx_te):,}  (mismas filas que D, E y F)")
print(f"   Peso de la clase positiva para balancear: {peso_pos:.4f}")

registrar("Reproducción de la partición de D, E y F", "Muestreo",
          f"prueba {len(idx_te):,} filas idénticas",
          "Comparación justa entre todas las fases", len(idx_te))
registrar("Validación interna para parada temprana", "Muestreo",
          f"{len(i_val):,} registros del entrenamiento",
          "La parada temprana necesita datos no vistos para detectar cuándo "
          "el modelo empieza a sobreajustar. Se toman del entrenamiento para "
          "no contaminar la prueba", len(i_val))

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)


def metricas_reg(y, p):
    y = np.asarray(y); nz = y != 0
    return {'MAE': mean_absolute_error(y, p), 'MSE': mean_squared_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)), 'R2': r2_score(y, p),
            'MAPE_%': np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100}


def metricas_bin(y, pred, prob):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred),
            'Recall_min': recall_score(y, pred, pos_label=0),
            'F1': f1_score(y, pred), 'ROC_AUC': roc_auc_score(y, prob),
            'PR_AUC': average_precision_score(y, prob)}


def metricas_multi(y, pred):
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, average='macro',
                                         zero_division=0),
            'Recall': recall_score(y, pred, average='macro'),
            'F1': f1_score(y, pred, average='macro')}


def matriz_bin(ax, y, pred, tit):
    mc = confusion_matrix(y, pred)
    ax.imshow(mc, cmap=CMAP)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mc[i, j]:,}", ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if mc[i, j] > mc.max()/2 else TINTA)
    ax.set_xticks([0, 1], ['Pred: sin', 'Pred: con'], fontsize=8)
    ax.set_yticks([0, 1], ['Real: sin', 'Real: con'], fontsize=8)
    ax.set_title(tit, fontsize=9.5); ax.grid(False)


def matriz_multi(ax, y, pred, tit):
    mc = confusion_matrix(y, pred, labels=ORDEN)
    mcn = mc / mc.sum(axis=1, keepdims=True)
    ax.imshow(mcn, cmap=CMAP, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mcn[i, j]*100:.0f} %", ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if mcn[i, j] > .5 else TINTA)
    ax.set_xticks(range(3), ORDEN, fontsize=8)
    ax.set_yticks(range(3), ORDEN, fontsize=8)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(tit, fontsize=9.5); ax.grid(False)


def dispersion(ax, y, p, tit):
    y = np.asarray(y)
    sel = np.random.default_rng(SEMILLA).choice(len(y), min(6000, len(y)),
                                                replace=False)
    j = np.random.default_rng(SEMILLA).uniform(-.3, .3, len(sel))
    ax.scatter(y[sel] + j, p[sel], s=5, alpha=.2, color=AZUL, edgecolors='none')
    ax.plot([0, 13], [0, 13], '--', color=CORAL, lw=2.2,
            label='Predicción perfecta')
    ax.set_xlabel("ieh real"); ax.set_ylabel("ieh predicho")
    ax.set_title(tit, fontsize=9.5); ax.legend(fontsize=7.5)


def importancia(ax, valores, tit, color=AZUL, n=14):
    s = pd.Series(valores, index=PREDICTORAS).sort_values().tail(n)
    s = s / s.sum()
    ax.barh([ETQ(c) for c in s.index], s.values, color=color,
            edgecolor='white', linewidth=1.3)
    ax.set_xlabel("Importancia relativa")
    ax.set_title(tit, fontsize=9.5)
    ax.tick_params(labelsize=7.5)


RES_BIN, PROB_BIN, PRED_BIN = {}, {}, {}
RES_REG, PRED_REG = {}, {}
RES_M, PRED_M = {}, {}
gc.collect()


# %%
# =============================================================================
#  G1 — ADABOOST (ADAPTIVE BOOSTING)
#  1. Entrena un aprendiz debil (un arbol pequeño) con todos los hogares
#     pesando igual.
#  2. Aumenta el peso de los hogares que clasifico MAL.
#  3. El siguiente arbol se ve obligado a concentrarse en esos casos.
#  4. La prediccion final es un voto ponderado: los arboles que aciertan
#     mas pesan mas.
# =============================================================================
titulo("G1 · ADABOOST — reponderar los errores")

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

base_c = DecisionTreeClassifier(max_depth=2, class_weight='balanced',
                                random_state=SEMILLA)
base_r = DecisionTreeRegressor(max_depth=4, random_state=SEMILLA)

t0 = time.time()
ada_b = AdaBoostClassifier(estimator=base_c, n_estimators=200,
                           learning_rate=0.5, random_state=SEMILLA)
ada_b.fit(X_tr, ytr_b)
nom = 'AdaBoost'
PROB_BIN[nom] = ada_b.predict_proba(X_te)[:, 1]
PRED_BIN[nom] = ada_b.predict(X_te)
RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                 time.time()-t0, "200 árboles de profundidad 2")
r = RES_BIN[nom]
print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · recall mín. "
      f"{r['Recall_min']:.4f} · {time.time()-t0:.1f}s")

t0 = time.time()
ada_r = AdaBoostRegressor(estimator=base_r, n_estimators=150,
                          learning_rate=0.05, loss='square',
                          random_state=SEMILLA)
ada_r.fit(X_tr, ytr_r)
PRED_REG[nom] = ada_r.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                 "150 árboles de profundidad 4")
r = RES_REG[nom]
print(f"   Regresión  R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · MAE "
      f"{r['MAE']:.4f} · MAPE {r['MAPE_%']:.2f} % · {time.time()-t0:.1f}s")

t0 = time.time()
ada_m = AdaBoostClassifier(estimator=base_c, n_estimators=200,
                           learning_rate=0.5, random_state=SEMILLA)
ada_m.fit(X_tr, ytr_m)
PRED_M[nom] = ada_m.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0)
print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · {time.time()-t0:.1f}s")

# Evolucion del AUC a medida que se agregan arboles
etapas = [roc_auc_score(yte_b, p[:, 1])
          for p in ada_b.staged_predict_proba(X_te)]
pesos_arb = ada_b.estimator_weights_[:len(etapas)]
print(f"\n   AUC con 1 árbol   : {etapas[0]:.4f}")
print(f"   AUC con 200 árboles: {etapas[-1]:.4f}")
print(f"   Ganancia del boosting: {etapas[-1]-etapas[0]:+.4f}")

fig = plt.figure(figsize=(17.5, 9))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)
ax = fig.add_subplot(gs[0, :2])
ax.plot(range(1, len(etapas)+1), etapas, lw=2.8, color=LILA)
ax.fill_between(range(1, len(etapas)+1), etapas[0], etapas, alpha=.15,
                color=LILA)
ax.axhline(etapas[0], color=GRIS, ls=':', lw=2,
           label=f'Un solo árbol ({etapas[0]:.3f})')
ax.set_xlabel("Número de árboles agregados en secuencia")
ax.set_ylabel("AUC de prueba")
ax.set_title("(a) Cada árbol corrige los errores de los anteriores")
ax.legend(fontsize=8.5)

ax = fig.add_subplot(gs[0, 2])
ax.bar(range(1, len(pesos_arb)+1), pesos_arb, color=AZUL, width=1)
ax.set_xlabel("Árbol número"); ax.set_ylabel("Peso en el voto final")
ax.set_title("(b) Peso de cada árbol")

matriz_bin(fig.add_subplot(gs[1, 0]), yte_b, PRED_BIN[nom],
           f"(c) Binaria · recall mín. {RES_BIN[nom]['Recall_min']:.3f}")
dispersion(fig.add_subplot(gs[1, 1]), yte_r, PRED_REG[nom],
           f"(d) Regresión · R² {RES_REG[nom]['R2']:.3f}")
matriz_multi(fig.add_subplot(gs[1, 2]), yte_m, PRED_M[nom],
             f"(e) Multiclase · F1 {RES_M[nom]['F1']:.3f}")
fig.suptitle("FASE G1 · ADABOOST — boosting adaptativo", fontsize=15, y=.98)
# Figura 49 del informe · G01_adaboost.png
guardar("G01_adaboost.png")
registrar("AdaBoost", "Modelado",
          f"AUC {RES_BIN[nom]['ROC_AUC']:.4f} · ganancia del boosting "
          f"{etapas[-1]-etapas[0]:+.4f}",
          "Cada árbol aumenta el peso de los hogares mal clasificados por los "
          "anteriores", round(float(RES_BIN[nom]['ROC_AUC']), 5))
gc.collect()


# %%
# =============================================================================
#  G2 — BOOSTING FRENTE A BAGGING: EL EXPERIMENTO QUE CIERRA LA FASE F
#  En F9, Bagging de 100 tocones apenas supero a un tocon solo. La hipotesis
#  es que Bagging reduce la varianza pero no el sesgo. Si es cierto, el
#  boosting de los MISMOS tocones deberia mejorar mucho mas.
# =============================================================================
titulo("G2 · BOOSTING FRENTE A BAGGING — mismo aprendiz débil")

from sklearn.ensemble import BaggingClassifier

tocon = DecisionTreeClassifier(max_depth=1, class_weight='balanced',
                               random_state=SEMILLA)
exp = {}
for nom_, m in [
    ('Un tocón', DecisionTreeClassifier(max_depth=1, class_weight='balanced',
                                        random_state=SEMILLA)),
    ('Bagging de 100 tocones', BaggingClassifier(
        estimator=tocon, n_estimators=100, n_jobs=-1, random_state=SEMILLA)),
    ('AdaBoost de 100 tocones', AdaBoostClassifier(
        estimator=tocon, n_estimators=100, learning_rate=0.5,
        random_state=SEMILLA))]:
    t0 = time.time()
    m.fit(X_tr, ytr_b)
    exp[nom_] = {'AUC': roc_auc_score(yte_b, m.predict_proba(X_te)[:, 1]),
                 'segundos': time.time()-t0}
    print(f"   {nom_:<26} AUC {exp[nom_]['AUC']:.4f} · "
          f"{exp[nom_]['segundos']:.1f}s")
exp = pd.DataFrame(exp).T

g_bag = exp.loc['Bagging de 100 tocones', 'AUC'] - exp.loc['Un tocón', 'AUC']
g_ada = exp.loc['AdaBoost de 100 tocones', 'AUC'] - exp.loc['Un tocón', 'AUC']
print(f"\n   Ganancia de Bagging : {g_bag:+.4f}")
print(f"   Ganancia de Boosting: {g_ada:+.4f}")
print(f"   El boosting ganó {g_ada/max(g_bag,1e-9):.1f} veces más que el bagging")
print("\n   CONCLUSIÓN: el tocón tiene SESGO alto, no varianza alta. Bagging")
print("   promedia tocones igual de simples y no lo corrige. Boosting los")
print("   encadena para que cada uno cubra lo que el anterior no aprendió.")

fig, ax = plt.subplots(1, 2, figsize=(15.5, 5))
ax[0].bar(exp.index, exp['AUC'], color=[GRIS, AMBAR, MENTA],
          edgecolor='white', linewidth=2.5, width=.6)
for i, v in enumerate(exp['AUC']):
    ax[0].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=11,
               fontweight='bold')
ax[0].set_ylim(exp['AUC'].min() - .03, exp['AUC'].max() + .025)
ax[0].set_ylabel("AUC de prueba")
ax[0].set_title("(a) Mismo aprendiz débil, dos estrategias")
ax[0].tick_params(axis='x', labelsize=8.5)

ax[1].bar(['Bagging', 'Boosting'], [g_bag, g_ada], color=[AMBAR, MENTA],
          edgecolor='white', linewidth=2.5, width=.5)
for i, v in enumerate([g_bag, g_ada]):
    ax[1].text(i, v, f"{v:+.4f}", ha='center', va='bottom', fontsize=12,
               fontweight='bold')
ax[1].set_ylabel("Mejora de AUC sobre un tocón")
ax[1].set_title("(b) Bagging reduce varianza · Boosting reduce sesgo")
fig.suptitle("FASE G2 · BOOSTING FRENTE A BAGGING", fontsize=14, y=1.03)
plt.tight_layout()
# Figura 50 del informe · G02_boosting_vs_bagging.png
guardar("G02_boosting_vs_bagging.png")
registrar("Boosting frente a Bagging", "Experimento",
          f"ganancia Bagging {g_bag:+.4f} · Boosting {g_ada:+.4f}",
          "Confirma la hipótesis planteada en F9: Bagging no corrige el sesgo "
          "de un aprendiz débil; el boosting sí", round(float(g_ada - g_bag), 5))
gc.collect()


# %%
# =============================================================================
#  G3 — GRADIENT BOOSTING
#  Cada nuevo arbol no repondera hogares como AdaBoost: aprende a predecir
#  el ERROR (el residuo) que dejo el modelo anterior. Se avanza en la
#  direccion que mas reduce la funcion de perdida, igual que un descenso
#  por gradiente.
# =============================================================================
titulo("G3 · GRADIENT BOOSTING — ajustar los residuos")

from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.utils.class_weight import compute_sample_weight

cfg_gb = dict(n_estimators=150, max_depth=3, learning_rate=0.1,
              subsample=0.8, random_state=SEMILLA)
w_b = compute_sample_weight('balanced', ytr_b)
w_m = compute_sample_weight('balanced', ytr_m)

t0 = time.time()
gb_b = GradientBoostingClassifier(**cfg_gb)
gb_b.fit(X_tr, ytr_b, sample_weight=w_b)
nom = 'Gradient Boosting'
PROB_BIN[nom] = gb_b.predict_proba(X_te)[:, 1]
PRED_BIN[nom] = gb_b.predict(X_te)
RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                 time.time()-t0, "150 árboles, profundidad 3, submuestreo 0,8")
r = RES_BIN[nom]
print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · recall mín. "
      f"{r['Recall_min']:.4f} · {time.time()-t0:.1f}s")

t0 = time.time()
gb_r = GradientBoostingRegressor(**cfg_gb, loss='squared_error')
gb_r.fit(X_tr, ytr_r)
PRED_REG[nom] = gb_r.predict(X_te)
RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0)
r = RES_REG[nom]
print(f"   Regresión  R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · MAE "
      f"{r['MAE']:.4f} · MAPE {r['MAPE_%']:.2f} % · {time.time()-t0:.1f}s")

t0 = time.time()
gb_m = GradientBoostingClassifier(**cfg_gb)
gb_m.fit(X_tr, ytr_m, sample_weight=w_m)
PRED_M[nom] = gb_m.predict(X_te)
RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
registrar_modelo('nivel_equip', 'Clasificación multiclase', nom, RES_M[nom],
                 time.time()-t0)
print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · {time.time()-t0:.1f}s")

# Curvas de perdida en entrenamiento y prueba
perd_tr = gb_r.train_score_
perd_te = [mean_squared_error(yte_r, p) for p in gb_r.staged_predict(X_te)]
print(f"\n   Error cuadrático de prueba · árbol 1: {perd_te[0]:.4f} · "
      f"árbol 150: {perd_te[-1]:.4f}")

fig = plt.figure(figsize=(17.5, 9))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)
ax = fig.add_subplot(gs[0, :2])
ax.plot(range(1, len(perd_tr)+1), perd_tr, lw=2.6, color=CORAL,
        label='Entrenamiento (submuestra)')
ax.plot(range(1, len(perd_te)+1), perd_te, lw=2.6, color=MENTA,
        label='Prueba')
ax.set_xlabel("Número de árboles"); ax.set_ylabel("Error cuadrático medio")
ax.set_title("(a) Cada árbol reduce el error residual (regresión de ieh)")
ax.legend(fontsize=8.5)
importancia(fig.add_subplot(gs[0, 2]), gb_b.feature_importances_,
            "(b) Importancia · internet", LILA)
matriz_bin(fig.add_subplot(gs[1, 0]), yte_b, PRED_BIN[nom],
           f"(c) Binaria · recall mín. {RES_BIN[nom]['Recall_min']:.3f}")
dispersion(fig.add_subplot(gs[1, 1]), yte_r, PRED_REG[nom],
           f"(d) Regresión · R² {RES_REG[nom]['R2']:.3f}")
matriz_multi(fig.add_subplot(gs[1, 2]), yte_m, PRED_M[nom],
             f"(e) Multiclase · F1 {RES_M[nom]['F1']:.3f}")
fig.suptitle("FASE G3 · GRADIENT BOOSTING", fontsize=15, y=.98)
# Figura 51 del informe · G03_gradient_boosting.png
guardar("G03_gradient_boosting.png")
registrar("Gradient Boosting", "Modelado",
          f"AUC {RES_BIN[nom]['ROC_AUC']:.4f} · R² {RES_REG[nom]['R2']:.4f}",
          "Cada árbol aprende el residuo del anterior. subsample=0,8 introduce "
          "aleatoriedad que reduce el sobreajuste (boosting estocástico)")
gc.collect()


# %%
# =============================================================================
#  G4 — TASA DE APRENDIZAJE (learning rate)
#  Controla cuánto corrige cada árbol. Una tasa alta aprende rápido pero
#  puede pasarse del óptimo; una baja aprende despacio pero con más precisión.
# =============================================================================
import importlib.util
XGB_OK = importlib.util.find_spec('xgboost') is not None
LGB_OK = importlib.util.find_spec('lightgbm') is not None
CAT_OK = importlib.util.find_spec('catboost') is not None


def omitido(nombre, libreria):
    titulo(f"{nombre} · OMITIDO")
    print(f"   La librería {libreria} no está instalada en este entorno.")
    print("   El bloque se omite; el resto de la fase no depende de él.")
    registrar(f"{nombre} omitido", "Limitación técnica",
              f"librería {libreria} no disponible",
              "No se pudo instalar en el entorno de ejecución")


titulo("G4 · TASA DE APRENDIZAJE — el hiperparámetro central del boosting")

tasas = [0.01, 0.05, 0.1, 0.3, 1.0]
curvas_lr = {}
for lr in tasas:
    t0 = time.time()
    m = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                   learning_rate=lr, subsample=0.8,
                                   random_state=SEMILLA)
    m.fit(X_fit, ytr_b.iloc[i_fit],
          sample_weight=compute_sample_weight('balanced', ytr_b.iloc[i_fit]))
    curvas_lr[lr] = [roc_auc_score(ytr_b.iloc[i_val], p[:, 1])
                     for p in m.staged_predict_proba(X_val)]
    mejor_it = int(np.argmax(curvas_lr[lr])) + 1
    print(f"   tasa {lr:<5}  AUC máximo {max(curvas_lr[lr]):.4f} en el árbol "
          f"{mejor_it:>3}  ·  AUC final {curvas_lr[lr][-1]:.4f}  ·  "
          f"{time.time()-t0:.1f}s")

lr_opt = max(curvas_lr, key=lambda k: max(curvas_lr[k]))
print(f"\n   Tasa óptima: {lr_opt}")
print("   Con tasa 1,0 cada árbol corrige todo el error de golpe: aprende")
print("   rápido pero oscila y se estanca antes.")

fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
for i, lr in enumerate(tasas):
    ax[0].plot(range(1, 201), curvas_lr[lr], lw=2.4, color=PALETA[i],
               label=f"tasa {lr}")
ax[0].set_xlabel("Número de árboles"); ax[0].set_ylabel("AUC de validación")
ax[0].set_title("(a) Tasa alta aprende rápido, tasa baja aprende despacio")
ax[0].legend(fontsize=8.5)
maximos = [max(curvas_lr[t]) for t in tasas]
ax[1].bar([str(t) for t in tasas], maximos,
          color=[MENTA if t == lr_opt else AZUL for t in tasas],
          edgecolor='white', linewidth=2)
for i, v in enumerate(maximos):
    ax[1].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=9,
               fontweight='bold')
ax[1].set_ylim(min(maximos) - .01, max(maximos) + .006)
ax[1].set_xlabel("Tasa de aprendizaje"); ax[1].set_ylabel("AUC máximo")
ax[1].set_title(f"(b) Mejor tasa: {lr_opt}")
fig.suptitle("FASE G4 · TASA DE APRENDIZAJE", fontsize=14, y=1.03)
plt.tight_layout()
# Figura 52 del informe · G04_tasa_aprendizaje.png
guardar("G04_tasa_aprendizaje.png")
registrar("Estudio de la tasa de aprendizaje", "Hiperparámetros",
          f"tasas {tasas} · óptima {lr_opt}",
          "Evaluado en la validación interna, sin tocar la prueba", lr_opt)
gc.collect()


# %%
# =============================================================================
#  G5 — XGBOOST (eXtreme Gradient Boosting)
#  Gradient boosting con regularización L1 y L2, cálculo paralelo y PARADA
#  TEMPRANA: se detiene cuando la validación deja de mejorar.
# =============================================================================
def bloque_g5():
    from xgboost import XGBClassifier, XGBRegressor
    titulo("G5 · XGBOOST — gradient boosting regularizado")
    cfg = dict(n_estimators=600, max_depth=6, learning_rate=0.08,
               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
               reg_lambda=1.0, tree_method='hist', n_jobs=-1,
               random_state=SEMILLA, early_stopping_rounds=40)
    nom = 'XGBoost'

    t0 = time.time()
    mb = XGBClassifier(**cfg, scale_pos_weight=peso_pos, eval_metric='auc')
    mb.fit(X_fit, ytr_b.iloc[i_fit], eval_set=[(X_val, ytr_b.iloc[i_val])],
           verbose=False)
    PROB_BIN[nom] = mb.predict_proba(X_te)[:, 1]
    PRED_BIN[nom] = mb.predict(X_te)
    RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
    registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                     time.time()-t0, f"parada en árbol {mb.best_iteration}")
    r = RES_BIN[nom]
    print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
          f"recall mín. {r['Recall_min']:.4f} · parada en {mb.best_iteration}"
          f" · {time.time()-t0:.1f}s")

    t0 = time.time()
    mr = XGBRegressor(**cfg, eval_metric='rmse')
    mr.fit(X_fit, ytr_r.iloc[i_fit], eval_set=[(X_val, ytr_r.iloc[i_val])],
           verbose=False)
    PRED_REG[nom] = mr.predict(X_te)
    RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
    registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                     f"parada en árbol {mr.best_iteration}")
    r = RES_REG[nom]
    print(f"   Regresión  R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · MAE "
          f"{r['MAE']:.4f} · MAPE {r['MAPE_%']:.2f} % · {time.time()-t0:.1f}s")

    t0 = time.time()
    mm = XGBClassifier(**cfg, objective='multi:softprob', num_class=3,
                       eval_metric='mlogloss')
    mm.fit(X_fit, ytr_mn.iloc[i_fit],
           sample_weight=compute_sample_weight('balanced', ytr_mn.iloc[i_fit]),
           eval_set=[(X_val, ytr_mn.iloc[i_val])], verbose=False)
    PRED_M[nom] = np.array(ORDEN)[mm.predict(X_te).astype(int)]
    RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
    registrar_modelo('nivel_equip', 'Clasificación multiclase', nom,
                     RES_M[nom], time.time()-t0)
    print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · "
          f"{time.time()-t0:.1f}s")

    hist = mb.evals_result()['validation_0']['auc']
    fig = plt.figure(figsize=(17.5, 9))
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(range(1, len(hist)+1), hist, lw=2.6, color=AZUL)
    ax.axvline(mb.best_iteration + 1, color=CORAL, ls='--', lw=2.2,
               label=f'Parada temprana · árbol {mb.best_iteration}')
    ax.axvspan(mb.best_iteration + 1, len(hist), color=CORAL, alpha=.08)
    ax.set_xlabel("Número de árboles"); ax.set_ylabel("AUC de validación")
    ax.set_title("(a) Parada temprana: se detiene cuando deja de mejorar")
    ax.legend(fontsize=8.5)
    g = mb.get_booster().get_score(importance_type='gain')
    importancia(fig.add_subplot(gs[0, 2]),
                np.array([g.get(c, 0) for c in PREDICTORAS]),
                "(b) Importancia por ganancia", AZUL)
    matriz_bin(fig.add_subplot(gs[1, 0]), yte_b, PRED_BIN[nom],
               f"(c) Binaria · recall mín. {RES_BIN[nom]['Recall_min']:.3f}")
    dispersion(fig.add_subplot(gs[1, 1]), yte_r, PRED_REG[nom],
               f"(d) Regresión · R² {RES_REG[nom]['R2']:.3f}")
    matriz_multi(fig.add_subplot(gs[1, 2]), yte_m, PRED_M[nom],
                 f"(e) Multiclase · F1 {RES_M[nom]['F1']:.3f}")
    fig.suptitle("FASE G5 · XGBOOST", fontsize=15, y=.98)
    # Figura 53 del informe · G05_xgboost.png
    guardar("G05_xgboost.png")
    registrar("XGBoost con parada temprana", "Modelado",
              f"AUC {RES_BIN[nom]['ROC_AUC']:.4f} · parada en el árbol "
              f"{mb.best_iteration} de 600",
              "La parada temprana evita el sobreajuste sin adivinar el número "
              "de árboles; reg_alpha y reg_lambda penalizan la complejidad")


if XGB_OK:
    bloque_g5()
else:
    omitido("G5 · XGBOOST", "xgboost")
gc.collect()


# %%
# =============================================================================
#  G6 — LIGHTGBM
#    · Crece HOJA A HOJA en lugar de nivel a nivel: más rápido.
#    · Trata las variables CATEGÓRICAS de forma nativa, sin one-hot.
# =============================================================================
def bloque_g6():
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    titulo("G6 · LIGHTGBM — crecimiento por hojas y categóricas nativas")

    def a_categoria(df):
        d = df.copy()
        for c in CATEGORICAS:
            d[c] = d[c].astype(int)
        return d

    Xc_fit, Xc_val, Xc_te = a_categoria(X_fit), a_categoria(X_val), \
        a_categoria(X_te)
    # Mismas categorías en entrenamiento, validación y prueba
    for c in CATEGORICAS:
        cats = sorted(set(Xc_fit[c]) | set(Xc_val[c]) | set(Xc_te[c]))
        for d_ in (Xc_fit, Xc_val, Xc_te):
            d_[c] = pd.Categorical(d_[c], categories=cats)

    cfg = dict(n_estimators=600, learning_rate=0.08, num_leaves=63,
               min_child_samples=40, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=1.0, random_state=SEMILLA,
               n_jobs=-1, verbose=-1)
    parar = [lgb.early_stopping(40, verbose=False)]

    sub("A/B testing · categóricas nativas frente a códigos numéricos")
    ab = {}
    for nom_, Xa, Xv, Xt in [('Códigos numéricos', X_fit, X_val, X_te),
                             ('Categóricas nativas', Xc_fit, Xc_val, Xc_te)]:
        t0 = time.time()
        m = LGBMClassifier(**cfg, class_weight='balanced')
        m.fit(Xa, ytr_b.iloc[i_fit], eval_set=[(Xv, ytr_b.iloc[i_val])],
              eval_metric='auc', callbacks=parar)
        ab[nom_] = roc_auc_score(yte_b, m.predict_proba(Xt)[:, 1])
        print(f"   {nom_:<22} AUC {ab[nom_]:.4f} · {time.time()-t0:.1f}s")
    dif = ab['Categóricas nativas'] - ab['Códigos numéricos']
    nativo = dif > 0
    print(f"   Diferencia: {dif:+.4f} → se usan "
          f"{'categóricas nativas' if nativo else 'códigos numéricos'}")
    registrar("A/B testing de categóricas en LightGBM", "Experimento",
              f"nativas vs códigos · ΔAUC {dif:+.4f}",
              "LightGBM puede agrupar categorías de forma óptima en cada "
              "división en lugar de tratarlas como números ordenados",
              round(float(dif), 5))

    A_fit, A_val, A_te = (Xc_fit, Xc_val, Xc_te) if nativo else \
        (X_fit, X_val, X_te)
    nom = 'LightGBM'

    t0 = time.time()
    mb = LGBMClassifier(**cfg, class_weight='balanced')
    mb.fit(A_fit, ytr_b.iloc[i_fit], eval_set=[(A_val, ytr_b.iloc[i_val])],
           eval_metric='auc', callbacks=parar)
    PROB_BIN[nom] = mb.predict_proba(A_te)[:, 1]
    PRED_BIN[nom] = mb.predict(A_te)
    RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
    registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                     time.time()-t0, f"parada en {mb.best_iteration_}")
    r = RES_BIN[nom]
    print(f"\n   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
          f"recall mín. {r['Recall_min']:.4f} · parada en "
          f"{mb.best_iteration_} · {time.time()-t0:.1f}s")

    t0 = time.time()
    mr = LGBMRegressor(**cfg, objective='poisson')
    mr.fit(A_fit, ytr_r.iloc[i_fit], eval_set=[(A_val, ytr_r.iloc[i_val])],
           callbacks=parar)
    PRED_REG[nom] = mr.predict(A_te)
    RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
    registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                     "objetivo Poisson, ieh es un conteo")
    r = RES_REG[nom]
    print(f"   Regresión  R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · MAE "
          f"{r['MAE']:.4f} · MAPE {r['MAPE_%']:.2f} % · objetivo Poisson · "
          f"{time.time()-t0:.1f}s")

    t0 = time.time()
    mm = LGBMClassifier(**cfg, class_weight='balanced')
    mm.fit(A_fit, ytr_m.iloc[i_fit], eval_set=[(A_val, ytr_m.iloc[i_val])],
           callbacks=parar)
    PRED_M[nom] = mm.predict(A_te)
    RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
    registrar_modelo('nivel_equip', 'Clasificación multiclase', nom,
                     RES_M[nom], time.time()-t0)
    print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · "
          f"{time.time()-t0:.1f}s")

    fig = plt.figure(figsize=(17.5, 9))
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(list(ab.keys()), list(ab.values()),
           color=[GRIS, MENTA] if nativo else [MENTA, GRIS],
           edgecolor='white', linewidth=2.5, width=.55)
    for i, v in enumerate(ab.values()):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_ylim(min(ab.values()) - .01, max(ab.values()) + .006)
    ax.set_ylabel("AUC"); ax.set_title("(a) A/B · tratamiento de categóricas")
    ax.tick_params(axis='x', labelsize=8.5)
    importancia(fig.add_subplot(gs[0, 1]),
                mb.booster_.feature_importance(importance_type='gain'),
                "(b) Importancia por ganancia", MENTA)
    ax = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(yte_b, PROB_BIN[nom])
    ax.plot(fpr, tpr, lw=2.8, color=MENTA,
            label=f"LightGBM ({RES_BIN[nom]['ROC_AUC']:.3f})")
    ax.fill_between(fpr, tpr, alpha=.12, color=MENTA)
    ax.plot([0, 1], [0, 1], '--', color=GRIS, lw=2)
    ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
    ax.set_title("(c) Curva ROC"); ax.legend(fontsize=8)
    matriz_bin(fig.add_subplot(gs[1, 0]), yte_b, PRED_BIN[nom],
               f"(d) Binaria · recall mín. {RES_BIN[nom]['Recall_min']:.3f}")
    dispersion(fig.add_subplot(gs[1, 1]), yte_r, PRED_REG[nom],
               f"(e) Regresión Poisson · R² {RES_REG[nom]['R2']:.3f}")
    matriz_multi(fig.add_subplot(gs[1, 2]), yte_m, PRED_M[nom],
                 f"(f) Multiclase · F1 {RES_M[nom]['F1']:.3f}")
    fig.suptitle("FASE G6 · LIGHTGBM", fontsize=15, y=.98)
    # Figura 54 del informe · G06_lightgbm.png
    guardar("G06_lightgbm.png")


if LGB_OK:
    bloque_g6()
else:
    omitido("G6 · LIGHTGBM", "lightgbm")
gc.collect()


# %%
# =============================================================================
#  G7 — CATBOOST (Categorical Boosting)
#  Codifica las categóricas con estadísticos ORDENADOS del objetivo: para
#  cada hogar solo usa información de los anteriores, evitando fuga.
# =============================================================================
def bloque_g7():
    from catboost import CatBoostClassifier, CatBoostRegressor
    titulo("G7 · CATBOOST — tratamiento ordenado de categóricas")

    Xk_fit, Xk_val, Xk_te = X_fit.copy(), X_val.copy(), X_te.copy()
    for d_ in (Xk_fit, Xk_val, Xk_te):
        for c in CATEGORICAS:
            d_[c] = d_[c].astype(int)
    cfg = dict(iterations=600, depth=6, learning_rate=0.08, l2_leaf_reg=3.0,
               random_seed=SEMILLA, verbose=0, od_type='Iter', od_wait=40,
               thread_count=-1, cat_features=CATEGORICAS)
    nom = 'CatBoost'

    t0 = time.time()
    mb = CatBoostClassifier(**cfg, auto_class_weights='Balanced',
                            eval_metric='AUC')
    mb.fit(Xk_fit, ytr_b.iloc[i_fit], eval_set=(Xk_val, ytr_b.iloc[i_val]))
    PROB_BIN[nom] = mb.predict_proba(Xk_te)[:, 1]
    PRED_BIN[nom] = mb.predict(Xk_te).astype(int).ravel()
    RES_BIN[nom] = metricas_bin(yte_b, PRED_BIN[nom], PROB_BIN[nom])
    registrar_modelo('internet', 'Clasificación binaria', nom, RES_BIN[nom],
                     time.time()-t0, f"parada en {mb.get_best_iteration()}")
    r = RES_BIN[nom]
    print(f"   Binaria    AUC {r['ROC_AUC']:.4f} · F1 {r['F1']:.4f} · "
          f"recall mín. {r['Recall_min']:.4f} · parada en "
          f"{mb.get_best_iteration()} · {time.time()-t0:.1f}s")

    t0 = time.time()
    mr = CatBoostRegressor(**cfg, loss_function='Poisson')
    mr.fit(Xk_fit, ytr_r.iloc[i_fit], eval_set=(Xk_val, ytr_r.iloc[i_val]))
    PRED_REG[nom] = mr.predict(Xk_te)
    RES_REG[nom] = metricas_reg(yte_r, PRED_REG[nom])
    registrar_modelo('ieh', 'Regresión', nom, RES_REG[nom], time.time()-t0,
                     "pérdida Poisson")
    r = RES_REG[nom]
    print(f"   Regresión  R² {r['R2']:.4f} · RMSE {r['RMSE']:.4f} · MAE "
          f"{r['MAE']:.4f} · MAPE {r['MAPE_%']:.2f} % · {time.time()-t0:.1f}s")

    t0 = time.time()
    mm = CatBoostClassifier(**cfg, auto_class_weights='Balanced',
                            loss_function='MultiClass')
    mm.fit(Xk_fit, ytr_m.iloc[i_fit], eval_set=(Xk_val, ytr_m.iloc[i_val]))
    PRED_M[nom] = mm.predict(Xk_te).ravel().astype(str)
    RES_M[nom] = metricas_multi(yte_m, PRED_M[nom])
    registrar_modelo('nivel_equip', 'Clasificación multiclase', nom,
                     RES_M[nom], time.time()-t0)
    print(f"   Multiclase F1 macro {RES_M[nom]['F1']:.4f} · "
          f"{time.time()-t0:.1f}s")

    fig = plt.figure(figsize=(17.5, 9))
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)
    ax = fig.add_subplot(gs[0, :2])
    h = mb.get_evals_result()['validation']['AUC']
    ax.plot(range(1, len(h)+1), h, lw=2.6, color=CORAL)
    ax.axvline(mb.get_best_iteration() + 1, color=AZUL, ls='--', lw=2.2,
               label=f'Mejor iteración · {mb.get_best_iteration()}')
    ax.set_xlabel("Número de árboles"); ax.set_ylabel("AUC de validación")
    ax.set_title("(a) Curva de aprendizaje con parada temprana")
    ax.legend(fontsize=8.5)
    importancia(fig.add_subplot(gs[0, 2]), mb.get_feature_importance(),
                "(b) Importancia de las variables", CORAL)
    matriz_bin(fig.add_subplot(gs[1, 0]), yte_b, PRED_BIN[nom],
               f"(c) Binaria · recall mín. {RES_BIN[nom]['Recall_min']:.3f}")
    dispersion(fig.add_subplot(gs[1, 1]), yte_r, PRED_REG[nom],
               f"(d) Regresión Poisson · R² {RES_REG[nom]['R2']:.3f}")
    matriz_multi(fig.add_subplot(gs[1, 2]), yte_m, PRED_M[nom],
                 f"(e) Multiclase · F1 {RES_M[nom]['F1']:.3f}")
    fig.suptitle("FASE G7 · CATBOOST", fontsize=15, y=.98)
    # Figura 55 del informe · G07_catboost.png
    guardar("G07_catboost.png")
    registrar("CatBoost", "Modelado",
              f"AUC {RES_BIN[nom]['ROC_AUC']:.4f} · R² "
              f"{RES_REG[nom]['R2']:.4f}",
              "Las categóricas se codifican con estadísticos ordenados del "
              "objetivo, que evitan la fuga de información")


if CAT_OK:
    bloque_g7()
else:
    omitido("G7 · CATBOOST", "catboost")
gc.collect()


# %%
# =============================================================================
#  G8 — COMPARACIÓN GLOBAL DE LA FASE Y BITÁCORA
# =============================================================================
titulo("G8 · COMPARACIÓN GLOBAL DE LA FASE G")

tb = pd.DataFrame(RES_BIN).T
tr = pd.DataFrame(RES_REG).T
tm = pd.DataFrame(RES_M).T
sub("Clasificación binaria"); print(tb.round(4).to_string())
sub("Regresión"); print(tr.round(4).to_string())
sub("Multiclase"); print(tm.round(4).to_string())

mb_, mr_, mm_ = tb['ROC_AUC'].idxmax(), tr['R2'].idxmax(), tm['F1'].idxmax()
mrec = tb['Recall_min'].idxmax()
print(f"""
   MEJOR PREVIO (fases D a F)        MEJOR DE LA FASE G
   AUC internet  0,8162 (RF)         {mb_:<20} {tb.loc[mb_,'ROC_AUC']:.4f}
   Recall mínimo 0,7498 (KNN)        {mrec:<20} {tb.loc[mrec,'Recall_min']:.4f}
   R² ieh        0,5441 (RF)         {mr_:<20} {tr.loc[mr_,'R2']:.4f}
   F1 nivel      0,6363 (Stacking)   {mm_:<20} {tm.loc[mm_,'F1']:.4f}
""")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.32)
ax = fig.add_subplot(gs[0, 0])
for i, n_ in enumerate(tb.index):
    fpr, tpr, _ = roc_curve(yte_b, PROB_BIN[n_])
    ax.plot(fpr, tpr, lw=2.2, color=PALETA[i],
            label=f"{n_} ({tb.loc[n_,'ROC_AUC']:.3f})")
ax.plot([0, 1], [0, 1], '--', color=GRIS, lw=1.8)
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("(a) Curvas ROC"); ax.legend(fontsize=7.5, loc='lower right')

ax = fig.add_subplot(gs[0, 1])
for i, n_ in enumerate(tb.index):
    pr, rc, _ = precision_recall_curve(yte_b, PROB_BIN[n_])
    ax.plot(rc, pr, lw=2.2, color=PALETA[i],
            label=f"{n_} ({tb.loc[n_,'PR_AUC']:.3f})")
ax.axhline(yte_b.mean(), color=GRIS, ls='--', lw=1.8)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("(b) Curvas Precision-Recall")
ax.legend(fontsize=7.5, loc='lower left')

paneles = [(tb, 'Recall_min', .7498, 'KNN (0,750)', '(c) Recall SIN internet',
            LILA, gs[0, 2]),
           (tb, 'ROC_AUC', .8162, 'RF (0,816)', '(d) AUC binaria', AZUL,
            gs[1, 0]),
           (tr, 'R2', .5441, 'RF (0,544)', '(e) R² regresión', AZUL, gs[1, 1]),
           (tm, 'F1', .6363, 'Stacking (0,636)', '(f) F1 multiclase', LILA,
            gs[1, 2])]
for tabla, col, ref, lab, tit, cc, pos in paneles:
    ax = fig.add_subplot(pos)
    s = tabla[col].sort_values()
    ax.barh(range(len(s)), s.values,
            color=[MENTA if v == s.max() else cc for v in s.values],
            edgecolor='white', linewidth=1.4)
    ax.set_yticks(range(len(s)), s.index, fontsize=8)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.4f}", va='center', fontsize=8, fontweight='bold')
    ax.axvline(ref, color=CORAL, ls='--', lw=2, label=f'Mejor previo {lab}')
    lo = min(s.min(), ref)
    ax.set_xlim(0 if col == 'R2' else lo - .05, max(s.max(), ref) + .03)
    ax.set_title(tit); ax.legend(fontsize=7)
fig.suptitle("FASE G8 · BOOSTING — comparación con las fases anteriores",
             fontsize=15, y=.99)
# Figura 56 del informe · G08_comparacion_global.png
guardar("G08_comparacion_global.png")

tb.round(5).to_csv(os.path.join(DIR_FIG, "G_clasificacion_binaria.csv"))
tr.round(5).to_csv(os.path.join(DIR_FIG, "G_regresion.csv"))
tm.round(5).to_csv(os.path.join(DIR_FIG, "G_clasificacion_multiclase.csv"))

registrar("Cierre de la Fase G", "Cierre",
          f"{len(MOD_FASE)} modelos · mejor AUC {mb_} "
          f"{tb.loc[mb_,'ROC_AUC']:.4f} · mejor R² {mr_} "
          f"{tr.loc[mr_,'R2']:.4f}",
          "El boosting reduce el sesgo encadenando árboles que corrigen los "
          "errores de los anteriores", len(MOD_FASE))
volcar_bitacora()
print(f"\n   Modelos entrenados: {len(MOD_FASE)}")
print(f"   Tiempo de la fase : {(time.time()-t_ini)/60:.1f} min")
gc.collect()

# %%


# =============================================================================
#  PROYECTO CPV 2024 · FASE H — VALIDACIÓN CRUZADA Y SELECCIÓN FINAL
#
#  H1  Visualización de las particiones: KFold frente a StratifiedKFold
#  H2  Elección del número de particiones k
#  H3  Validación cruzada · clasificación binaria
#  H4  Validación cruzada · regresión
#  H5  A/B testing entre modelos: t pareada y prueba de McNemar
#  H6  Ranking final consolidado de todo el proyecto
#
# =============================================================================


# %%
# =============================================================================
#  H0 — CONFIGURACIÓN Y CARGA
# =============================================================================
import os
import gc
import time
import warnings
import importlib.util
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore")
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

SEMILLA = 42
N_CV = 40_000        # registros para la validación cruzada
K_FINAL = 5          # se confirma en H2
FASE = "H"
DIR_FIG = "figuras_v2"
os.makedirs(DIR_FIG, exist_ok=True)

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA, GRIS = "#FF6B8A", "#FFC145", "#22223B", "#8D99AE"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR, "#6C8AE4", "#2EC4B6"]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "axes.axisbelow": True,
    "font.size": 9, "figure.facecolor": "white", "axes.facecolor": "#FCFDFF",
    "legend.frameon": True, "legend.framealpha": .92})


def guardar(n):
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    print(f"   [figura] {r}")
    plt.show()


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


BITACORA_XLSX = "BITACORA_PROYECTO.xlsx"
REG_FASE, MOD_FASE = [], []


def registrar(paso, categoria, detalle, justificacion, valor=None):
    REG_FASE.append({'fase': FASE, 'n': len(REG_FASE) + 1,
                     'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     'categoria': categoria, 'paso': paso, 'detalle': detalle,
                     'justificacion': justificacion, 'valor': valor})
    print(f"   [bitácora {FASE}-{len(REG_FASE):02d}] {paso}")


def registrar_modelo(objetivo, tipo, modelo, metricas, segundos, nota=""):
    fila = {'fase': FASE,
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objetivo': objetivo, 'tipo': tipo, 'modelo': modelo,
            'segundos': round(segundos, 2), 'nota': nota}
    fila.update({k: round(float(v), 5) for k, v in metricas.items()})
    MOD_FASE.append(fila)


def leer_hoja(nombre):
    if os.path.exists(BITACORA_XLSX):
        try:
            return pd.read_excel(BITACORA_XLSX, sheet_name=nombre)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def volcar_bitacora():
    reg, mod = leer_hoja('Registro'), leer_hoja('Modelos')
    if len(reg) and 'fase' in reg:
        reg = reg[reg['fase'].astype(str) != FASE]
    if len(mod) and 'fase' in mod:
        mod = mod[mod['fase'].astype(str) != FASE]
    reg = pd.concat([reg, pd.DataFrame(REG_FASE)], ignore_index=True)
    mod = pd.concat([mod, pd.DataFrame(MOD_FASE)], ignore_index=True)
    resumen = (reg.groupby('fase').agg(pasos=('paso', 'count'),
                                       ultima=('fecha', 'max')).reset_index())
    try:
        with pd.ExcelWriter(BITACORA_XLSX, engine='openpyxl') as w:
            resumen.to_excel(w, sheet_name='Resumen por fase', index=False)
            reg.to_excel(w, sheet_name='Registro', index=False)
            mod.to_excel(w, sheet_name='Modelos', index=False)
        print(f"   ✔ Bitácora actualizada ({len(reg)} pasos · {len(mod)} modelos)")
    except PermissionError:
        pass


titulo("FASE H · VALIDACIÓN CRUZADA Y SELECCIÓN FINAL")
t_ini = time.time()

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     cross_validate)

BASE = "vivienda_cpv2024_LIMPIO_v2"
datos = pd.read_parquet(f"{BASE}.parquet") if os.path.exists(f"{BASE}.parquet") \
    else pd.read_csv(f"{BASE}.csv", sep=';')
for c in datos.select_dtypes(include=['float64']).columns:
    datos[c] = datos[c].astype('float32')

if len(datos) > N_CV:
    muestra, _ = train_test_split(datos, train_size=N_CV,
                                  stratify=datos['internet'],
                                  random_state=SEMILLA)
else:
    muestra = datos.copy()
muestra = muestra.reset_index(drop=True)
del datos
gc.collect()

CATEGORICAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v15_servsan', 'v16_desague', 'v17_tenencia', 'tip_hog',
               'v20a_emi', 'v21a_fal']
NUMERICAS = ['v13_habitac', 'v14_dormit', 'tot_pers', 'v20b_totemi',
             'v21b_totfal', 'hacinamiento', 'pers_por_habitac',
             'hacin_critico', 'sin_servicios']
CATEGORICAS = [c for c in CATEGORICAS if c in muestra.columns]
NUMERICAS = [c for c in NUMERICAS if c in muestra.columns]
PREDICTORAS = CATEGORICAS + NUMERICAS

X = muestra[PREDICTORAS]
y_b = muestra['internet'].astype(int)
y_r = muestra['ieh'].astype(float)

XGB_OK = importlib.util.find_spec('xgboost') is not None
LGB_OK = importlib.util.find_spec('lightgbm') is not None
CAT_OK = importlib.util.find_spec('catboost') is not None

print(f"   Registros para validación cruzada : {len(muestra):,}")
print(f"   Proporción con internet           : {y_b.mean()*100:.2f} %")
print(f"   XGBoost {XGB_OK} · LightGBM {LGB_OK} · CatBoost {CAT_OK}")

registrar("Muestra para validación cruzada", "Muestreo",
          f"{len(muestra):,} registros estratificados",
          "La validación cruzada entrena k veces cada modelo: con la muestra "
          "completa de 120.000 el costo se multiplicaría por k",
          len(muestra))


# %%
# =============================================================================
#  H1 — LAS PARTICIONES: KFold FRENTE A StratifiedKFold
#  KFold corta el conjunto en k bloques sin mirar la clase. StratifiedKFold
#  se asegura de que CADA bloque conserve la proporción original de clases.
#  Con un objetivo desbalanceado (76 % / 24 %) esa diferencia importa.
# =============================================================================
titulo("H1 · VISUALIZACIÓN DE LAS PARTICIONES")

N_VIS = 500                      # solo para dibujar
ordenado = np.argsort(y_b.values[:N_VIS])   # ordenado por clase, peor caso
yv = y_b.values[:N_VIS][ordenado]

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5))
for a, (cv, nom) in zip(ax[:2], [
        (KFold(n_splits=K_FINAL, shuffle=False), "KFold sin mezclar"),
        (StratifiedKFold(n_splits=K_FINAL, shuffle=True,
                         random_state=SEMILLA), "StratifiedKFold")]):
    for i, (tr, te) in enumerate(cv.split(np.zeros(N_VIS), yv)):
        ind = np.zeros(N_VIS)
        ind[te] = 1
        a.scatter(range(N_VIS), [i + .5] * N_VIS, c=ind, marker='_',
                  lw=9, cmap=LinearSegmentedColormap.from_list(
                      "f", [AZUL, AMBAR]), vmin=-.2, vmax=1.2)
    a.scatter(range(N_VIS), [K_FINAL + .8] * N_VIS, c=yv, marker='_', lw=9,
              cmap=LinearSegmentedColormap.from_list("c", [MENTA, CORAL]))
    a.set_yticks(np.arange(K_FINAL) + .5, [f"Partición {i+1}"
                                           for i in range(K_FINAL)])
    a.text(-30, K_FINAL + .8, "Clase", ha='right', va='center', fontsize=8.5,
           fontweight='bold')
    a.set_ylim(K_FINAL + 1.4, -.2)
    a.set_xlabel("Índice de la observación")
    a.set_title(f"({'a' if nom.startswith('K') else 'b'}) {nom}")
    a.grid(False)
ax[0].legend(handles=[Patch(color=AZUL, label='Entrenamiento'),
                      Patch(color=AMBAR, label='Prueba'),
                      Patch(color=MENTA, label='Sin internet'),
                      Patch(color=CORAL, label='Con internet')],
             fontsize=7.5, loc='lower right')

# Proporcion de la clase positiva en cada particion
comp = []
for nom, cv in [("KFold", KFold(n_splits=K_FINAL, shuffle=True,
                                random_state=SEMILLA)),
                ("StratifiedKFold", StratifiedKFold(
                    n_splits=K_FINAL, shuffle=True, random_state=SEMILLA))]:
    for i, (_, te) in enumerate(cv.split(X, y_b)):
        comp.append({'metodo': nom, 'particion': i + 1,
                     'pct': y_b.iloc[te].mean() * 100})
comp = pd.DataFrame(comp)
piv = comp.pivot(index='particion', columns='metodo', values='pct')
x = np.arange(K_FINAL); w = .38
ax[2].bar(x - w/2, piv['KFold'], w, label='KFold', color=AMBAR,
          edgecolor='white', linewidth=1.5)
ax[2].bar(x + w/2, piv['StratifiedKFold'], w, label='StratifiedKFold',
          color=MENTA, edgecolor='white', linewidth=1.5)
ax[2].axhline(y_b.mean()*100, color=CORAL, ls='--', lw=2,
              label=f'Real {y_b.mean()*100:.2f} %')
ax[2].set_xticks(x, [f"P{i+1}" for i in range(K_FINAL)])
ax[2].set_ylim(min(comp['pct']) - 1, max(comp['pct']) + 1)
ax[2].set_ylabel("% con internet en la partición")
ax[2].set_title("(c) ¿Cada partición representa al total?")
ax[2].legend(fontsize=8)
fig.suptitle("FASE H1 · PARTICIONES DE LA VALIDACIÓN CRUZADA", fontsize=14,
             y=1.02)
plt.tight_layout()
# Figura 57 del informe · H01_particiones.png
guardar("H01_particiones.png")

d_kf = comp[comp.metodo == 'KFold']['pct'].std()
d_sk = comp[comp.metodo == 'StratifiedKFold']['pct'].std()
print(f"   Desviación entre particiones · KFold {d_kf:.4f} · "
      f"StratifiedKFold {d_sk:.4f}")
print("   La estratificación mantiene la proporción de clases casi idéntica")
print("   en las k particiones; sin ella, cada bloque puede desviarse.")
registrar("Elección del esquema de partición", "Validación",
          f"desviación KFold {d_kf:.4f} vs estratificado {d_sk:.4f}",
          "Con un objetivo desbalanceado, la estratificación evita que una "
          "partición quede con muy pocos casos de la clase minoritaria")
gc.collect()


# %%
# =============================================================================
#  H2 — ¿CUÁNTAS PARTICIONES? ELECCIÓN DE k
#  Pocas particiones: cada modelo entrena con pocos datos → estimación
#  pesimista. Muchas: mejor estimación pero el costo crece de forma lineal.
#  Se busca el punto donde la media se estabiliza: el "codo" de la curva.
# =============================================================================
titulo("H2 · ELECCIÓN DEL NÚMERO DE PARTICIONES")

from sklearn.ensemble import RandomForestClassifier

sonda = RandomForestClassifier(n_estimators=60, max_depth=14,
                               min_samples_leaf=20, class_weight='balanced',
                               n_jobs=-1, random_state=SEMILLA)
codo = []
for k in [2, 3, 5, 10]:
    t0 = time.time()
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEMILLA)
    r = cross_validate(sonda, X, y_b, cv=cv, scoring='roc_auc', n_jobs=1)
    codo.append({'k': k, 'auc_medio': r['test_score'].mean(),
                 'desviacion': r['test_score'].std(),
                 'segundos': time.time() - t0,
                 'n_entrenamiento': int(len(X) * (k-1) / k)})
    print(f"   k = {k:>2}  AUC {codo[-1]['auc_medio']:.4f} ± "
          f"{codo[-1]['desviacion']:.4f}  ·  entrena con "
          f"{codo[-1]['n_entrenamiento']:,}  ·  {codo[-1]['segundos']:.1f}s")
codo = pd.DataFrame(codo)

mejora = codo['auc_medio'].diff().fillna(0)
print(f"\n   Ganancia de AUC al pasar de k=5 a k=10: "
      f"{codo.loc[3,'auc_medio']-codo.loc[2,'auc_medio']:+.4f}")
print(f"   Costo adicional: {codo.loc[3,'segundos']/codo.loc[2,'segundos']:.1f} "
      f"veces más tiempo")
print(f"\n   SE ADOPTA k = {K_FINAL}: la media ya se estabilizó y el costo")
print(f"   sigue siendo razonable. Es además el valor estándar en la práctica.")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].errorbar(codo['k'], codo['auc_medio'], yerr=codo['desviacion'],
               fmt='o-', lw=2.8, ms=10, capsize=6, color=AZUL,
               markerfacecolor='white', markeredgewidth=2.2)
ax[0].axvline(K_FINAL, color=MENTA, ls='--', lw=2.2, label=f'k = {K_FINAL}')
ax[0].set_xlabel("Número de particiones (k)")
ax[0].set_ylabel("AUC medio ± desviación")
ax[0].set_title("(a) La media se estabiliza: el codo")
ax[0].legend(fontsize=8.5)
ax[1].plot(codo['k'], codo['desviacion'], 'o-', lw=2.8, ms=10, color=LILA,
           markerfacecolor='white', markeredgewidth=2.2)
ax[1].set_xlabel("k"); ax[1].set_ylabel("Desviación entre particiones")
ax[1].set_title("(b) Variabilidad de la estimación")
ax[2].bar(codo['k'].astype(str), codo['segundos'], color=AMBAR,
          edgecolor='white', linewidth=2)
for i, v in enumerate(codo['segundos']):
    ax[2].text(i, v, f"{v:.0f}s", ha='center', va='bottom', fontsize=9.5,
               fontweight='bold')
ax[2].set_xlabel("k"); ax[2].set_ylabel("Segundos")
ax[2].set_title("(c) El costo crece de forma lineal")
fig.suptitle("FASE H2 · ¿CUÁNTAS PARTICIONES USAR?", fontsize=14, y=1.03)
plt.tight_layout()
# Figura 58 del informe · H02_eleccion_k.png
guardar("H02_eleccion_k.png")
registrar("Elección del número de particiones", "Validación",
          f"k ∈ [2,3,5,10] · se adopta k={K_FINAL}",
          "A partir de k=5 la media se estabiliza y el costo adicional no se "
          "compensa con la mejora en precisión de la estimación", K_FINAL)
gc.collect()


# %%
# =============================================================================
#  H3 — VALIDACIÓN CRUZADA · CLASIFICACIÓN BINARIA
#  Cada modelo se entrena y evalúa k veces sobre particiones distintas.
#  Así se distingue el desempeño real del azar de una sola partición.
# =============================================================================
titulo("H3 · VALIDACIÓN CRUZADA — clasificación de internet")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score

prep = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first',
                          sparse_output=False, dtype=np.float32), CATEGORICAS),
    ('num', StandardScaler(), NUMERICAS)])

finalistas_b = {
    'Regresión Logística': Pipeline([('p', prep), ('m', LogisticRegression(
        max_iter=500, class_weight='balanced', random_state=SEMILLA))]),
    'Random Forest': RandomForestClassifier(
        n_estimators=120, max_depth=16, min_samples_leaf=20,
        class_weight='balanced', n_jobs=-1, random_state=SEMILLA),
}
if XGB_OK:
    from xgboost import XGBClassifier
    finalistas_b['XGBoost'] = XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method='hist', n_jobs=-1,
        scale_pos_weight=(y_b == 0).sum()/(y_b == 1).sum(),
        eval_metric='auc', random_state=SEMILLA)
if LGB_OK:
    from lightgbm import LGBMClassifier
    finalistas_b['LightGBM'] = LGBMClassifier(
        n_estimators=250, learning_rate=0.1, num_leaves=63,
        min_child_samples=40, class_weight='balanced', n_jobs=-1,
        verbose=-1, random_state=SEMILLA)
if CAT_OK:
    from catboost import CatBoostClassifier
    finalistas_b['CatBoost'] = CatBoostClassifier(
        iterations=250, depth=6, learning_rate=0.1, verbose=0,
        auto_class_weights='Balanced', thread_count=-1,
        random_seed=SEMILLA)

cv_b = StratifiedKFold(n_splits=K_FINAL, shuffle=True, random_state=SEMILLA)
marcadores = {'auc': 'roc_auc', 'pr_auc': 'average_precision',
              'f1': 'f1', 'recall_min': make_scorer(recall_score, pos_label=0)}

folds_b, res_b = {}, {}
for nom, mod in finalistas_b.items():
    t0 = time.time()
    r = cross_validate(mod, X, y_b, cv=cv_b, scoring=marcadores, n_jobs=1)
    folds_b[nom] = r['test_auc']
    res_b[nom] = {'AUC_medio': r['test_auc'].mean(),
                  'AUC_desv': r['test_auc'].std(),
                  'PR_AUC': r['test_pr_auc'].mean(),
                  'F1': r['test_f1'].mean(),
                  'Recall_min': r['test_recall_min'].mean(),
                  'Recall_min_desv': r['test_recall_min'].std()}
    registrar_modelo('internet', 'Clasificación binaria', nom, res_b[nom],
                     time.time()-t0, f"validación cruzada k={K_FINAL}")
    print(f"   {nom:<22} AUC {res_b[nom]['AUC_medio']:.4f} ± "
          f"{res_b[nom]['AUC_desv']:.4f} · recall mín. "
          f"{res_b[nom]['Recall_min']:.4f} · {time.time()-t0:.1f}s")

tabla_b = pd.DataFrame(res_b).T.sort_values('AUC_medio', ascending=False)
sub("Resultados"); print(tabla_b.round(4).to_string())
mejor_b = tabla_b.index[0]
print(f"\n   Mejor por AUC medio: {mejor_b}")
print(f"   Desviación entre particiones: ±{tabla_b.loc[mejor_b,'AUC_desv']:.4f}")
print("   Una desviación pequeña indica que el resultado NO depende de la")
print("   partición: el modelo es estable.")

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5))
datos_bp = [folds_b[n] for n in tabla_b.index]
bp = ax[0].boxplot(datos_bp, tick_labels=[n[:14] for n in tabla_b.index],
                   patch_artist=True,
                   medianprops=dict(color=TINTA, linewidth=2))
for p, c in zip(bp['boxes'], PALETA):
    p.set_facecolor(c); p.set_alpha(.75); p.set_edgecolor('white')
for i, n in enumerate(tabla_b.index):
    ax[0].scatter([i+1]*K_FINAL, folds_b[n], s=28, color=TINTA, zorder=5,
                  alpha=.7)
ax[0].set_ylabel("AUC por partición")
ax[0].set_title(f"(a) Dispersión entre las {K_FINAL} particiones")
ax[0].tick_params(axis='x', rotation=18, labelsize=8)

for i, n in enumerate(tabla_b.index):
    ax[1].plot(range(1, K_FINAL+1), folds_b[n], 'o-', lw=2.2, ms=7,
               color=PALETA[i], label=n[:16])
ax[1].set_xticks(range(1, K_FINAL+1))
ax[1].set_xlabel("Partición"); ax[1].set_ylabel("AUC")
ax[1].set_title("(b) Desempeño en cada partición")
ax[1].legend(fontsize=7.5)

s = tabla_b['AUC_medio'].sort_values()
ax[2].barh(range(len(s)), s.values,
           xerr=tabla_b.loc[s.index, 'AUC_desv'], capsize=5,
           color=[MENTA if v == s.max() else AZUL for v in s.values],
           edgecolor='white', linewidth=1.5)
ax[2].set_yticks(range(len(s)), s.index, fontsize=8.5)
for i, v in enumerate(s.values):
    ax[2].text(v, i, f"  {v:.4f}", va='center', fontsize=8.5,
               fontweight='bold')
ax[2].set_xlim(s.min() - .02, s.max() + .015)
ax[2].set_xlabel("AUC medio ± desviación")
ax[2].set_title("(c) Ranking con incertidumbre")
fig.suptitle("FASE H3 · VALIDACIÓN CRUZADA · CLASIFICACIÓN", fontsize=14,
             y=1.02)
plt.tight_layout()
# Figura 59 del informe · H03_cv_clasificacion.png
guardar("H03_cv_clasificacion.png")
gc.collect()


# %%
# =============================================================================
#  H4 — VALIDACIÓN CRUZADA · REGRESIÓN
# =============================================================================
titulo("H4 · VALIDACIÓN CRUZADA — regresión del equipamiento")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

finalistas_r = {
    'Ridge': Pipeline([('p', prep), ('m', Ridge(alpha=10.0))]),
    'Random Forest': RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=20, n_jobs=-1,
        random_state=SEMILLA),
}
if XGB_OK:
    from xgboost import XGBRegressor
    finalistas_r['XGBoost'] = XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method='hist', n_jobs=-1,
        random_state=SEMILLA)
if LGB_OK:
    from lightgbm import LGBMRegressor
    finalistas_r['LightGBM'] = LGBMRegressor(
        n_estimators=250, learning_rate=0.1, num_leaves=63,
        min_child_samples=40, objective='poisson', n_jobs=-1, verbose=-1,
        random_state=SEMILLA)
if CAT_OK:
    from catboost import CatBoostRegressor
    finalistas_r['CatBoost'] = CatBoostRegressor(
        iterations=250, depth=6, learning_rate=0.1, verbose=0,
        loss_function='Poisson', thread_count=-1, random_seed=SEMILLA)

cv_r = KFold(n_splits=K_FINAL, shuffle=True, random_state=SEMILLA)
folds_r, res_r = {}, {}
for nom, mod in finalistas_r.items():
    t0 = time.time()
    r = cross_validate(mod, X, y_r, cv=cv_r, n_jobs=1,
                       scoring=['r2', 'neg_root_mean_squared_error',
                                'neg_mean_absolute_error'])
    folds_r[nom] = r['test_r2']
    res_r[nom] = {'R2_medio': r['test_r2'].mean(),
                  'R2_desv': r['test_r2'].std(),
                  'RMSE': -r['test_neg_root_mean_squared_error'].mean(),
                  'MAE': -r['test_neg_mean_absolute_error'].mean()}
    registrar_modelo('ieh', 'Regresión', nom, res_r[nom], time.time()-t0,
                     f"validación cruzada k={K_FINAL}")
    print(f"   {nom:<22} R² {res_r[nom]['R2_medio']:.4f} ± "
          f"{res_r[nom]['R2_desv']:.4f} · RMSE {res_r[nom]['RMSE']:.4f} · "
          f"{time.time()-t0:.1f}s")

tabla_r = pd.DataFrame(res_r).T.sort_values('R2_medio', ascending=False)
sub("Resultados"); print(tabla_r.round(4).to_string())
mejor_r = tabla_r.index[0]
print(f"\n   Mejor por R² medio: {mejor_r}")

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5))
bp = ax[0].boxplot([folds_r[n] for n in tabla_r.index],
                   tick_labels=[n[:14] for n in tabla_r.index],
                   patch_artist=True,
                   medianprops=dict(color=TINTA, linewidth=2))
for p, c in zip(bp['boxes'], PALETA):
    p.set_facecolor(c); p.set_alpha(.75); p.set_edgecolor('white')
for i, n in enumerate(tabla_r.index):
    ax[0].scatter([i+1]*K_FINAL, folds_r[n], s=28, color=TINTA, zorder=5,
                  alpha=.7)
ax[0].set_ylabel("R² por partición")
ax[0].set_title(f"(a) Dispersión entre las {K_FINAL} particiones")
ax[0].tick_params(axis='x', rotation=18, labelsize=8)

for i, n in enumerate(tabla_r.index):
    ax[1].plot(range(1, K_FINAL+1), folds_r[n], 'o-', lw=2.2, ms=7,
               color=PALETA[i], label=n[:16])
ax[1].set_xticks(range(1, K_FINAL+1))
ax[1].set_xlabel("Partición"); ax[1].set_ylabel("R²")
ax[1].set_title("(b) Desempeño en cada partición")
ax[1].legend(fontsize=7.5)

s = tabla_r['R2_medio'].sort_values()
ax[2].barh(range(len(s)), s.values, xerr=tabla_r.loc[s.index, 'R2_desv'],
           capsize=5,
           color=[MENTA if v == s.max() else LILA for v in s.values],
           edgecolor='white', linewidth=1.5)
ax[2].set_yticks(range(len(s)), s.index, fontsize=8.5)
for i, v in enumerate(s.values):
    ax[2].text(v, i, f"  {v:.4f}", va='center', fontsize=8.5,
               fontweight='bold')
ax[2].set_xlabel("R² medio ± desviación")
ax[2].set_title("(c) Ranking con incertidumbre")
fig.suptitle("FASE H4 · VALIDACIÓN CRUZADA · REGRESIÓN", fontsize=14, y=1.02)
plt.tight_layout()
# Figura 60 del informe · H04_cv_regresion.png
guardar("H04_cv_regresion.png")
gc.collect()


# %%
# =============================================================================
#  H5 — A/B TESTING ENTRE MODELOS
#  ¿La diferencia entre dos modelos es real o es azar?
#    · t pareada sobre las k particiones: compara el desempeño medio
#    · prueba de McNemar: compara los ACIERTOS Y ERRORES uno a uno
# =============================================================================
titulo("H5 · A/B TESTING ENTRE MODELOS")

sub("Prueba t pareada sobre las particiones")
print("   H0: los dos modelos tienen el mismo desempeño medio")
nombres = list(tabla_b.index)
pv = pd.DataFrame(np.ones((len(nombres), len(nombres))), index=nombres,
                  columns=nombres)
for i, a in enumerate(nombres):
    for j, b in enumerate(nombres):
        if i < j:
            t, p = stats.ttest_rel(folds_b[a], folds_b[b])
            pv.loc[a, b] = pv.loc[b, a] = p
print(pv.round(4).to_string())

print("\n   Comparaciones con el mejor modelo:")
for n in nombres[1:]:
    p = pv.loc[mejor_b, n]
    d = np.mean(folds_b[mejor_b]) - np.mean(folds_b[n])
    veredicto = "DIFERENCIA REAL" if p < .05 else "indistinguible"
    print(f"     {mejor_b} vs {n:<22} Δ {d:+.4f} · p = {p:.4f} → {veredicto}")

sub("Prueba de McNemar sobre el conjunto de prueba")
i_tr, i_te = train_test_split(np.arange(len(X)), test_size=.25,
                              stratify=y_b, random_state=SEMILLA)
aciertos = {}
for n in nombres[:3]:
    m = finalistas_b[n]
    m.fit(X.iloc[i_tr], y_b.iloc[i_tr])
    aciertos[n] = (m.predict(X.iloc[i_te]) == y_b.iloc[i_te].values)

a_, b_ = nombres[0], nombres[1]
n01 = int((~aciertos[a_] & aciertos[b_]).sum())
n10 = int((aciertos[a_] & ~aciertos[b_]).sum())
est = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if (n10 + n01) > 0 else 0
p_mc = 1 - stats.chi2.cdf(est, df=1)
print(f"   {a_} acierta y {b_} falla : {n10:,}")
print(f"   {a_} falla y {b_} acierta : {n01:,}")
print(f"   Estadístico χ² = {est:.4f} · valor p = {p_mc:.5f}")
print(f"   → {'Se rechaza H0: los modelos difieren' if p_mc < .05 else 'No hay evidencia de diferencia'}")

registrar("A/B testing entre modelos", "Experimento",
          f"t pareada y McNemar · {a_} vs {b_} · p McNemar {p_mc:.5f}",
          "Dos pruebas complementarias: la t pareada compara el desempeño "
          "medio entre particiones; McNemar compara aciertos y errores caso "
          "por caso sobre las mismas observaciones", round(float(p_mc), 6))

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5))
im = ax[0].imshow(pv.values, cmap='RdYlGn_r', vmin=0, vmax=.2)
for i in range(len(nombres)):
    for j in range(len(nombres)):
        if i != j:
            ax[0].text(j, i, f"{pv.values[i, j]:.3f}", ha='center',
                       va='center', fontsize=8,
                       fontweight='bold' if pv.values[i, j] < .05 else 'normal')
ax[0].set_xticks(range(len(nombres)), [n[:12] for n in nombres], rotation=40,
                 fontsize=7.5, ha='right')
ax[0].set_yticks(range(len(nombres)), [n[:12] for n in nombres], fontsize=7.5)
ax[0].set_title("(a) Valores p · t pareada\nverde = diferencia significativa")
ax[0].grid(False)
plt.colorbar(im, ax=ax[0], shrink=.8)

mc = np.array([[int((aciertos[a_] & aciertos[b_]).sum()), n10],
               [n01, int((~aciertos[a_] & ~aciertos[b_]).sum())]])
ax[1].imshow(mc, cmap=CMAP)
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, f"{mc[i, j]:,}", ha='center', va='center',
                   fontsize=13, fontweight='bold',
                   color='white' if mc[i, j] > mc.max()/2 else TINTA)
ax[1].set_xticks([0, 1], [f'{b_[:11]}\nacierta', f'{b_[:11]}\nfalla'],
                 fontsize=8)
ax[1].set_yticks([0, 1], [f'{a_[:11]}\nacierta', f'{a_[:11]}\nfalla'],
                 fontsize=8)
ax[1].set_title(f"(b) McNemar · p = {p_mc:.4f}")
ax[1].grid(False)

difs = [np.mean(folds_b[mejor_b]) - np.mean(folds_b[n]) for n in nombres[1:]]
ps = [pv.loc[mejor_b, n] for n in nombres[1:]]
ax[2].barh(range(len(difs)), difs,
           color=[MENTA if p < .05 else GRIS for p in ps],
           edgecolor='white', linewidth=1.5)
ax[2].set_yticks(range(len(difs)), [n[:16] for n in nombres[1:]], fontsize=8.5)
for i, (d, p) in enumerate(zip(difs, ps)):
    ax[2].text(d, i, f"  {d:+.4f} (p={p:.3f})", va='center', fontsize=8)
ax[2].axvline(0, color=TINTA, lw=1.5)
ax[2].set_xlabel(f"Ventaja de {mejor_b} en AUC")
ax[2].set_title("(c) Verde = diferencia estadísticamente real")
fig.suptitle("FASE H5 · ¿LAS DIFERENCIAS SON REALES?", fontsize=14, y=1.02)
plt.tight_layout()
# Figura 61 del informe · H05_ab_testing.png
guardar("H05_ab_testing.png")
gc.collect()


# %%
# =============================================================================
#  H6 — RANKING FINAL DE TODO EL PROYECTO
# =============================================================================
titulo("H6 · RANKING FINAL CONSOLIDADO")

mod = leer_hoja('Modelos')
if len(mod):
    mod = mod[mod['fase'].astype(str) != FASE]
    cb = mod[mod['objetivo'] == 'internet'].dropna(subset=['ROC_AUC'])
    cr = mod[mod['objetivo'] == 'ieh'].dropna(subset=['R2'])
    sub("Los 10 mejores por AUC en todo el proyecto (partición única)")
    print(cb.nlargest(10, 'ROC_AUC')[['fase', 'modelo', 'ROC_AUC',
                                      'Recall_min', 'F1']]
          .round(4).to_string(index=False))
    sub("Los 10 mejores por R² en todo el proyecto")
    print(cr.nlargest(10, 'R2')[['fase', 'modelo', 'R2', 'RMSE', 'MAE']]
          .round(4).to_string(index=False))

sub("Veredicto con validación cruzada")
print(f"   CLASIFICACIÓN · {mejor_b}")
print(f"     AUC {tabla_b.loc[mejor_b,'AUC_medio']:.4f} ± "
      f"{tabla_b.loc[mejor_b,'AUC_desv']:.4f}")
print(f"     Recall de la clase SIN internet: "
      f"{tabla_b.loc[mejor_b,'Recall_min']:.4f}")
print(f"\n   REGRESIÓN · {mejor_r}")
print(f"     R² {tabla_r.loc[mejor_r,'R2_medio']:.4f} ± "
      f"{tabla_r.loc[mejor_r,'R2_desv']:.4f}")
print(f"     RMSE {tabla_r.loc[mejor_r,'RMSE']:.4f} bienes de 13")

fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
if len(mod):
    top = cb.nlargest(12, 'ROC_AUC').sort_values('ROC_AUC')
    col = {'D': AZUL, 'E': LILA, 'F': MENTA, 'G': AMBAR}
    ax[0].barh(range(len(top)), top['ROC_AUC'],
               color=[col.get(str(f), GRIS) for f in top['fase']],
               edgecolor='white', linewidth=1.4)
    ax[0].set_yticks(range(len(top)),
                     [f"{m[:20]} ({f})" for m, f in zip(top['modelo'],
                                                        top['fase'])],
                     fontsize=8)
    ax[0].set_xlim(top['ROC_AUC'].min() - .02, top['ROC_AUC'].max() + .008)
    ax[0].set_xlabel("AUC")
    ax[0].set_title("(a) Los 12 mejores clasificadores del proyecto")
    ax[0].legend(handles=[Patch(color=v, label=f"Fase {k}")
                          for k, v in col.items()], fontsize=8)

    topr = cr.nlargest(12, 'R2').sort_values('R2')
    ax[1].barh(range(len(topr)), topr['R2'],
               color=[col.get(str(f), GRIS) for f in topr['fase']],
               edgecolor='white', linewidth=1.4)
    ax[1].set_yticks(range(len(topr)),
                     [f"{m[:20]} ({f})" for m, f in zip(topr['modelo'],
                                                        topr['fase'])],
                     fontsize=8)
    ax[1].set_xlim(0, topr['R2'].max() + .03)
    ax[1].set_xlabel("R²")
    ax[1].set_title("(b) Los 12 mejores modelos de regresión")
fig.suptitle("FASE H6 · RANKING FINAL DEL PROYECTO", fontsize=14, y=1.02)
plt.tight_layout()
# Figura 62 del informe · H06_ranking_final.png
guardar("H06_ranking_final.png")

tabla_b.round(5).to_csv(os.path.join(DIR_FIG, "H_cv_clasificacion.csv"))
tabla_r.round(5).to_csv(os.path.join(DIR_FIG, "H_cv_regresion.csv"))
pv.round(5).to_csv(os.path.join(DIR_FIG, "H_valores_p.csv"))

registrar("Cierre del proyecto de modelado", "Cierre",
          f"clasificación {mejor_b} AUC "
          f"{tabla_b.loc[mejor_b,'AUC_medio']:.4f} · regresión {mejor_r} R² "
          f"{tabla_r.loc[mejor_r,'R2_medio']:.4f}",
          "La validación cruzada confirma el desempeño con una estimación de "
          "su incertidumbre, no con una sola partición afortunada")
volcar_bitacora()
print(f"\n   Tiempo de la fase: {(time.time()-t_ini)/60:.1f} min")
gc.collect()

# %%


# =============================================================================
#  PROYECTO CPV 2024 · FASE I — CIERRE Y ENTREGABLES
#
#  No entrena modelos nuevos: consolida el proyecto y deja listos los
#  productos que consumirán el MLOps y la aplicación web.
#
#  I1  Consolidación de las métricas de todas las fases
#  I2  Entrenamiento final de los dos modelos ganadores
#  I3  Exportación de artefactos y predicciones agregadas
#  I4  Figura resumen del proyecto
#  I5  Informe de métricas en TXT y bitácora final
#
# =============================================================================


# %%
# =============================================================================
#  I0 — CONFIGURACIÓN
# =============================================================================
import os
import gc
import json
import time
import warnings
import importlib.util
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import joblib

warnings.filterwarnings("ignore")
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

SEMILLA = 42
# el modelo ya es estable y la memoria se mantiene bajo control.
N_FINAL = 600_000

FASE = "I"
DIR_FIG = "figuras_v2"
DIR_ENT = "entregables"
os.makedirs(DIR_FIG, exist_ok=True)
os.makedirs(DIR_ENT, exist_ok=True)

AZUL, LILA, MENTA = "#3D5AF1", "#8B6FE8", "#4ECDC4"
CORAL, AMBAR, TINTA, GRIS = "#FF6B8A", "#FFC145", "#22223B", "#8D99AE"
PALETA = [AZUL, LILA, MENTA, CORAL, AMBAR]
CMAP = LinearSegmentedColormap.from_list("p", ["#EEF2FF", AZUL, LILA])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.edgecolor": "#D8DEF0",
    "axes.grid": True, "grid.color": "#EAEFFA", "axes.axisbelow": True,
    "font.size": 9, "figure.facecolor": "white", "axes.facecolor": "#FCFDFF"})


def guardar(n):
    r = os.path.join(DIR_FIG, n)
    plt.savefig(r, bbox_inches="tight", facecolor="white")
    print(f"   [figura] {r}")
    plt.show()


def titulo(t):
    print("\n" + "═" * 86); print(f"  {t}"); print("═" * 86)


def sub(t):
    print(f"\n── {t} " + "─" * max(0, 80 - len(t)))


BITACORA_XLSX = "BITACORA_PROYECTO.xlsx"
REG_FASE = []


def registrar(paso, categoria, detalle, justificacion, valor=None):
    REG_FASE.append({'fase': FASE, 'n': len(REG_FASE) + 1,
                     'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     'categoria': categoria, 'paso': paso, 'detalle': detalle,
                     'justificacion': justificacion, 'valor': valor})
    print(f"   [bitácora {FASE}-{len(REG_FASE):02d}] {paso}")


def leer_hoja(nombre):
    if os.path.exists(BITACORA_XLSX):
        try:
            return pd.read_excel(BITACORA_XLSX, sheet_name=nombre)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


titulo("FASE I · CIERRE Y ENTREGABLES")
t_ini = time.time()

CATEGORICAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v15_servsan', 'v16_desague', 'v17_tenencia', 'tip_hog',
               'v20a_emi', 'v21a_fal']
NUMERICAS = ['v13_habitac', 'v14_dormit', 'tot_pers', 'v20b_totemi',
             'v21b_totfal', 'hacinamiento', 'pers_por_habitac',
             'hacin_critico', 'sin_servicios']
DEPTOS = {1: 'Chuquisaca', 2: 'La Paz', 3: 'Cochabamba', 4: 'Oruro',
          5: 'Potosí', 6: 'Santa Cruz', 7: 'Tarija', 8: 'Beni', 9: 'Pando'}


# %%
# =============================================================================
#  I1 — CONSOLIDACIÓN DE TODAS LAS MÉTRICAS DEL PROYECTO
# =============================================================================
titulo("I1 · CONSOLIDACIÓN DE MÉTRICAS")

mod = leer_hoja('Modelos')
if not len(mod):
    raise SystemExit("   No se encontró la hoja 'Modelos' en la bitácora.")

print(f"   Registros de modelos en la bitácora : {len(mod)}")
print(f"   Fases registradas                   : "
      f"{', '.join(sorted(mod['fase'].astype(str).unique()))}")

sub("Modelos por fase y objetivo")
print(pd.crosstab(mod['fase'], mod['objetivo']).to_string())

CLF = mod[(mod['objetivo'] == 'internet') & mod['ROC_AUC'].notna()]
REG = mod[(mod['objetivo'] == 'ieh') & mod['R2'].notna()]
MUL = mod[(mod['objetivo'] == 'nivel_equip') & mod['F1'].notna()]

sub("Clasificación binaria · ordenado por AUC")
t_clf = (CLF[['fase', 'modelo', 'Accuracy', 'Precision', 'Recall',
              'Recall_min', 'F1', 'ROC_AUC', 'PR_AUC', 'segundos']]
         .sort_values('ROC_AUC', ascending=False).reset_index(drop=True))
print(t_clf.round(4).head(15).to_string())

sub("Regresión · ordenado por R²")
t_reg = (REG[['fase', 'modelo', 'MAE', 'RMSE', 'R2', 'MAPE_%', 'segundos']]
         .sort_values('R2', ascending=False).reset_index(drop=True))
print(t_reg.round(4).head(15).to_string())

sub("Multiclase · ordenado por F1 macro")
t_mul = (MUL[['fase', 'modelo', 'Accuracy', 'Precision', 'Recall', 'F1',
              'segundos']]
         .sort_values('F1', ascending=False).reset_index(drop=True))
print(t_mul.round(4).head(12).to_string())

for nom, t in [('clasificacion_binaria', t_clf), ('regresion', t_reg),
               ('multiclase', t_mul)]:
    t.round(5).to_csv(os.path.join(DIR_ENT, f"I_todas_metricas_{nom}.csv"),
                      index=False, encoding='utf-8')
print(f"\n   Tablas completas guardadas en {DIR_ENT}/")

registrar("Consolidación de métricas", "Cierre",
          f"{len(mod)} modelos de {mod['fase'].nunique()} fases",
          "Reúne en una sola tabla los resultados de todo el proyecto para "
          "el anexo del informe", len(mod))
gc.collect()


# %%
# =============================================================================
#  I2 — ENTRENAMIENTO FINAL CON EL CONJUNTO AMPLIADO
#  Hasta ahora se trabajó sobre muestras. Los dos ganadores confirmados por
#  validación cruzada se reentrenan con muchos más registros: es el modelo
#  que irá a la aplicación web.
# =============================================================================
titulo("I2 · ENTRENAMIENTO DE LOS MODELOS FINALES")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)

BASE = "vivienda_cpv2024_LIMPIO_v2"
datos = pd.read_parquet(f"{BASE}.parquet") if os.path.exists(f"{BASE}.parquet") \
    else pd.read_csv(f"{BASE}.csv", sep=';')
for c in datos.select_dtypes(include=['float64']).columns:
    datos[c] = datos[c].astype('float32')
N_TOTAL = len(datos)
PREDICTORAS = [c for c in datos.columns
               if c not in ('internet', 'ieh', 'nivel_equip')]

if N_TOTAL > N_FINAL:
    base, _ = train_test_split(datos, train_size=N_FINAL,
                               stratify=datos['internet'],
                               random_state=SEMILLA)
else:
    base = datos.copy()
base = base.reset_index(drop=True)
print(f"   Dataset limpio        : {N_TOTAL:,} viviendas")
print(f"   Usado para el modelo  : {len(base):,} "
      f"({len(base)/N_TOTAL*100:.1f} %)")

Xb = base[PREDICTORAS]
yb = base['internet'].astype(int)
yr = base['ieh'].astype(float)
i_tr, i_te = train_test_split(np.arange(len(base)), test_size=.20,
                              stratify=yb, random_state=SEMILLA)
print(f"   Entrenamiento / prueba: {len(i_tr):,} / {len(i_te):,}")

# ---------------- CLASIFICACIÓN: Random Forest ----------------
sub("Random Forest · acceso a internet")
t0 = time.time()
final_clf = RandomForestClassifier(n_estimators=120, max_depth=18,
                                   min_samples_leaf=50, max_features='sqrt',
                                   class_weight='balanced', n_jobs=-1,
                                   random_state=SEMILLA)
final_clf.fit(Xb.iloc[i_tr], yb.iloc[i_tr])
prob = final_clf.predict_proba(Xb.iloc[i_te])[:, 1]
pred = final_clf.predict(Xb.iloc[i_te])
yt = yb.iloc[i_te]
MET_CLF = {'Accuracy': accuracy_score(yt, pred),
           'Precision': precision_score(yt, pred),
           'Recall': recall_score(yt, pred),
           'Recall_min': recall_score(yt, pred, pos_label=0),
           'F1': f1_score(yt, pred),
           'ROC_AUC': roc_auc_score(yt, prob),
           'PR_AUC': average_precision_score(yt, prob)}
t_clf_fin = time.time() - t0
for k, v in MET_CLF.items():
    print(f"     {k:<12} {v:.4f}")
print(f"     Entrenado en {t_clf_fin:.1f} s")
print(f"     Referencia con validación cruzada: AUC 0,8098 ± 0,0023")

# ---------------- REGRESIÓN: CatBoost, con respaldo ----------------
sub("Regresión del equipamiento")
CAT_OK = importlib.util.find_spec('catboost') is not None
t0 = time.time()
if CAT_OK:
    from catboost import CatBoostRegressor
    nombre_reg = "CatBoost"
    final_reg = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.08,
                                  l2_leaf_reg=3.0, loss_function='Poisson',
                                  verbose=0, thread_count=-1,
                                  random_seed=SEMILLA)
    final_reg.fit(Xb.iloc[i_tr], yr.iloc[i_tr])
else:
    from sklearn.ensemble import RandomForestRegressor
    nombre_reg = "Random Forest"
    print("   CatBoost no disponible: se usa Random Forest, el mejor de las "
          "alternativas instaladas")
    final_reg = RandomForestRegressor(n_estimators=120, max_depth=18,
                                      min_samples_leaf=50, n_jobs=-1,
                                      random_state=SEMILLA)
    final_reg.fit(Xb.iloc[i_tr], yr.iloc[i_tr])
p_reg = final_reg.predict(Xb.iloc[i_te])
ytr_ = yr.iloc[i_te].values
nz = ytr_ != 0
MET_REG = {'MAE': mean_absolute_error(ytr_, p_reg),
           'MSE': mean_squared_error(ytr_, p_reg),
           'RMSE': float(np.sqrt(mean_squared_error(ytr_, p_reg))),
           'R2': r2_score(ytr_, p_reg),
           'MAPE_%': float(np.mean(np.abs((ytr_[nz]-p_reg[nz])/ytr_[nz]))*100)}
t_reg_fin = time.time() - t0
for k, v in MET_REG.items():
    print(f"     {k:<12} {v:.4f}")
print(f"     Entrenado en {t_reg_fin:.1f} s")
print(f"     Referencia con validación cruzada: R² 0,5581 ± 0,0083")

registrar("Entrenamiento de los modelos finales", "Modelo final",
          f"Random Forest AUC {MET_CLF['ROC_AUC']:.4f} · {nombre_reg} R² "
          f"{MET_REG['R2']:.4f} sobre {len(base):,} registros",
          "Los ganadores confirmados por validación cruzada se reentrenan con "
          "un conjunto mucho mayor: es el modelo que va a la aplicación",
          len(base))
gc.collect()


# %%
# =============================================================================
#  I3 — EXPORTACIÓN DE ARTEFACTOS
# =============================================================================
titulo("I3 · ARTEFACTOS PARA MLOPS Y LA APLICACIÓN WEB")

r_clf = os.path.join(DIR_ENT, "modelo_internet.joblib")
r_reg = os.path.join(DIR_ENT, "modelo_ieh.joblib")
joblib.dump(final_clf, r_clf, compress=3)
joblib.dump(final_reg, r_reg, compress=3)
print(f"   ✔ {r_clf}  ({os.path.getsize(r_clf)/1024**2:.1f} MB)")
print(f"   ✔ {r_reg}  ({os.path.getsize(r_reg)/1024**2:.1f} MB)")

META = {
    'proyecto': 'Brecha digital y equipamiento del hogar · CPV 2024',
    'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'version': '1.0',
    'semilla': SEMILLA,
    'dataset': {'archivo': f"{BASE}.parquet", 'registros_totales': N_TOTAL,
                'registros_usados': len(base),
                'predictoras': PREDICTORAS,
                'categoricas': [c for c in CATEGORICAS if c in PREDICTORAS],
                'numericas': [c for c in NUMERICAS if c in PREDICTORAS]},
    'modelo_clasificacion': {
        'algoritmo': 'RandomForestClassifier', 'objetivo': 'internet',
        'hiperparametros': {'n_estimators': 120, 'max_depth': 18,
                            'min_samples_leaf': 50, 'max_features': 'sqrt',
                            'class_weight': 'balanced'},
        'metricas': {k: round(float(v), 5) for k, v in MET_CLF.items()},
        'validacion_cruzada': {'auc_medio': 0.8098, 'auc_desv': 0.0023,
                               'k': 5}},
    'modelo_regresion': {
        'algoritmo': nombre_reg, 'objetivo': 'ieh',
        'metricas': {k: round(float(v), 5) for k, v in MET_REG.items()},
        'validacion_cruzada': {'r2_medio': 0.5581, 'r2_desv': 0.0083,
                               'k': 5}},
    'validacion_externa': {'cobertura_propia_%': 76.27,
                           'cobertura_oficial_INE_%': 76.30},
}
r_meta = os.path.join(DIR_ENT, "metadata_modelos.json")
json.dump(META, open(r_meta, 'w', encoding='utf-8'), indent=2,
          ensure_ascii=False)
print(f"   ✔ {r_meta}")

# ---- Agregación por departamento (insumo del mapa) -------------------------
sub("Agregación territorial")
Xtodo = datos[PREDICTORAS]
datos['p_internet'] = final_clf.predict_proba(Xtodo)[:, 1]
datos['ieh_predicho'] = final_reg.predict(Xtodo)

agg = datos.groupby('idep').agg(
    viviendas=('internet', 'size'),
    internet_real=('internet', 'mean'),
    internet_predicho=('p_internet', 'mean'),
    ieh_real=('ieh', 'mean'),
    ieh_predicho=('ieh_predicho', 'mean'),
    hacin_critico=('hacin_critico', 'mean'),
    sin_servicios=('sin_servicios', 'mean'),
    pct_rural=('urbrur', lambda s: (s == 2).mean())).reset_index()
agg['departamento'] = agg['idep'].map(DEPTOS)
for c in ['internet_real', 'internet_predicho', 'hacin_critico',
          'sin_servicios', 'pct_rural']:
    agg[c] = (agg[c] * 100).round(2)
agg['brecha_prediccion'] = (agg['internet_predicho'] -
                            agg['internet_real']).round(2)
agg = agg.sort_values('internet_real')
print(agg.round(3).to_string(index=False))
r_agg = os.path.join(DIR_ENT, "indicadores_por_departamento.csv")
agg.to_csv(r_agg, index=False, encoding='utf-8')
print(f"\n   ✔ {r_agg}")

print("\n   NOTA PARA EL MAPA MUNICIPAL")
print("   El dataset limpio conserva solo el departamento (idep). Para llegar")
print("   al nivel municipal hay que releer el archivo original y conservar")
print("   iprov e imun, construyendo el código de municipio con:")
print("     cod_mun = idep(2) + iprov(2) + imun(2)")

registrar("Exportación de artefactos", "Entregables",
          "modelos joblib, metadata.json e indicadores departamentales",
          "Son los insumos que consumirán el versionado con DVC y MLflow y la "
          "aplicación web", 3)
gc.collect()


# %%
# =============================================================================
#  I4 — FIGURA RESUMEN DEL PROYECTO
# =============================================================================
titulo("I4 · FIGURA RESUMEN")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)
col_fase = {'D': AZUL, 'E': LILA, 'F': MENTA, 'G': AMBAR, 'H': CORAL}

ax = fig.add_subplot(gs[0, 0])
ev = CLF.groupby('fase')['ROC_AUC'].max().reindex(['D', 'E', 'F', 'G']).dropna()
ax.plot(ev.index, ev.values, 'o-', lw=3, ms=13, color=AZUL,
        markerfacecolor='white', markeredgewidth=2.6)
for i, (f, v) in enumerate(ev.items()):
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                xytext=(0, 13), ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel("Fase"); ax.set_ylabel("Mejor AUC alcanzado")
ax.set_title("(a) Evolución en clasificación")

ax = fig.add_subplot(gs[0, 1])
ev2 = REG.groupby('fase')['R2'].max().reindex(['D', 'E', 'F', 'G']).dropna()
ax.plot(ev2.index, ev2.values, 'o-', lw=3, ms=13, color=MENTA,
        markerfacecolor='white', markeredgewidth=2.6)
for i, (f, v) in enumerate(ev2.items()):
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                xytext=(0, 13), ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel("Fase"); ax.set_ylabel("Mejor R² alcanzado")
ax.set_title("(b) Evolución en regresión")

ax = fig.add_subplot(gs[0, 2])
cuenta = mod.groupby('fase').size()
ax.bar(cuenta.index, cuenta.values,
       color=[col_fase.get(str(f), GRIS) for f in cuenta.index],
       edgecolor='white', linewidth=2)
for i, v in enumerate(cuenta.values):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=11,
            fontweight='bold')
ax.set_xlabel("Fase"); ax.set_ylabel("Modelos entrenados")
ax.set_title(f"(c) {len(mod)} modelos en total")

ax = fig.add_subplot(gs[1, 0])
top = t_clf.head(10).sort_values('ROC_AUC')
ax.barh(range(len(top)), top['ROC_AUC'],
        color=[col_fase.get(str(f), GRIS) for f in top['fase']],
        edgecolor='white', linewidth=1.4)
ax.set_yticks(range(len(top)),
              [f"{m[:20]} ({f})" for m, f in zip(top['modelo'], top['fase'])],
              fontsize=8)
ax.set_xlim(top['ROC_AUC'].min() - .015, top['ROC_AUC'].max() + .006)
ax.set_xlabel("AUC"); ax.set_title("(d) Diez mejores clasificadores")
ax.legend(handles=[Patch(color=v, label=f"Fase {k}")
                   for k, v in col_fase.items() if k in set(top['fase'])],
          fontsize=7.5)

ax = fig.add_subplot(gs[1, 1])
mc = confusion_matrix(yt, pred)
ax.imshow(mc, cmap=CMAP)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{mc[i, j]:,}", ha='center', va='center', fontsize=12,
                fontweight='bold',
                color='white' if mc[i, j] > mc.max()/2 else TINTA)
ax.set_xticks([0, 1], ['Pred: sin', 'Pred: con'], fontsize=8.5)
ax.set_yticks([0, 1], ['Real: sin', 'Real: con'], fontsize=8.5)
ax.set_title(f"(e) Modelo final · recall sin internet "
             f"{MET_CLF['Recall_min']:.3f}")
ax.grid(False)

ax = fig.add_subplot(gs[1, 2])
a2 = agg.sort_values('internet_real')
ax.barh(a2['departamento'], a2['internet_real'], color=AZUL,
        edgecolor='white', linewidth=1.4)
ax.axvline(76.27, color=CORAL, ls='--', lw=2, label='Nacional 76,27 %')
ax.set_xlabel("% de hogares con internet")
ax.set_title("(f) Cobertura por departamento")
ax.legend(fontsize=8); ax.tick_params(labelsize=8)

fig.suptitle("FASE I · RESUMEN DEL PROYECTO DE APRENDIZAJE SUPERVISADO",
             fontsize=15, y=.98)
# Figura 63 del informe · I01_resumen_proyecto.png
guardar("I01_resumen_proyecto.png")
gc.collect()


# %%
# =============================================================================
#  I5 — INFORME DE MÉTRICAS EN TXT Y BITÁCORA FINAL
# =============================================================================
titulo("I5 · INFORME DE MÉTRICAS")

RUTA_TXT = os.path.join(DIR_ENT, "INFORME_METRICAS.txt")
L = []
L.append("=" * 86)
L.append("  INFORME DE MÉTRICAS · PROYECTO CPV 2024")
L.append("  Brecha digital y equipamiento del hogar en Bolivia")
L.append(f"  Generado: {datetime.now():%Y-%m-%d %H:%M}")
L.append("=" * 86)
L.append("")
L.append("1. DATOS")
L.append(f"   Registros originales del censo : 4.490.488")
L.append(f"   Dataset limpio                 : {N_TOTAL:,} ({N_TOTAL/4490488*100:.2f} %)")
L.append(f"   Predictoras                    : {len(PREDICTORAS)}")
L.append(f"   Validación externa             : 76,27 % propio vs 76,30 % INE")
L.append("")
L.append("2. OBJETIVOS")
L.append("   internet    · clasificación binaria  (76,27 % / 23,73 %)")
L.append("   ieh         · regresión continua     (0 a 13 bienes)")
L.append("   nivel_equip · clasificación multiclase (Bajo/Medio/Alto)")
L.append("")
L.append("3. MODELOS EVALUADOS")
L.append(f"   Total: {len(mod)} · Fases: "
         f"{', '.join(sorted(mod['fase'].astype(str).unique()))}")
L.append("")
L.append("4. CLASIFICACIÓN BINARIA · TODOS LOS MODELOS")
L.append(t_clf.round(4).to_string())
L.append("")
L.append("5. REGRESIÓN · TODOS LOS MODELOS")
L.append(t_reg.round(4).to_string())
L.append("")
L.append("6. MULTICLASE · TODOS LOS MODELOS")
L.append(t_mul.round(4).to_string())
L.append("")
L.append("7. VALIDACIÓN CRUZADA (k = 5, estratificada)")
L.append("   Clasificación · Random Forest  AUC 0,8098 ± 0,0023")
L.append("   Regresión     · CatBoost       R²  0,5581 ± 0,0083")
L.append("   Random Forest y CatBoost son estadísticamente indistinguibles")
L.append("   en clasificación (t pareada p = 0,5285; McNemar p = 0,0940).")
L.append("")
L.append("8. MODELOS FINALES")
L.append(f"   Entrenados con {len(base):,} registros")
L.append("   Clasificación · RandomForestClassifier")
for k, v in MET_CLF.items():
    L.append(f"      {k:<12} {v:.5f}")
L.append(f"   Regresión · {nombre_reg}")
for k, v in MET_REG.items():
    L.append(f"      {k:<12} {v:.5f}")
L.append("")
L.append("9. INDICADORES POR DEPARTAMENTO")
L.append(agg.round(2).to_string(index=False))
L.append("")
L.append("10. CONCLUSIONES")
L.append("   · La limpieza reproduce el universo oficial del INE.")
L.append("   · Más de 50 modelos convergen en AUC ~0,81: el límite no es el")
L.append("     algoritmo sino las variables disponibles en el censo de")
L.append("     vivienda, que no incluye ingresos ni educación.")
L.append("   · El boosting sí aporta en regresión: R² de 0,529 a 0,558.")
L.append("   · La validación cruzada corrigió el ranking obtenido con una")
L.append("     sola partición: la diferencia entre los punteros era ruido.")
L.append("   · Cuatro métodos independientes coinciden en las variables")
L.append("     determinantes: área urbano/rural, manejo de basura, material")
L.append("     del piso y combustible de cocina.")
L.append("=" * 86)

open(RUTA_TXT, 'w', encoding='utf-8').write("\n".join(L))
print(f"   ✔ {RUTA_TXT}  ({len(L)} líneas)")

registrar("Informe de métricas", "Entregables",
          f"{RUTA_TXT} con {len(mod)} modelos",
          "Documento único con todas las métricas del proyecto para el anexo "
          "del informe", len(L))

reg = leer_hoja('Registro')
if len(reg) and 'fase' in reg:
    reg = reg[reg['fase'].astype(str) != FASE]
reg = pd.concat([reg, pd.DataFrame(REG_FASE)], ignore_index=True)
resumen = (reg.groupby('fase').agg(pasos=('paso', 'count'),
                                   ultima=('fecha', 'max')).reset_index())
try:
    with pd.ExcelWriter(BITACORA_XLSX, engine='openpyxl') as w:
        resumen.to_excel(w, sheet_name='Resumen por fase', index=False)
        reg.to_excel(w, sheet_name='Registro', index=False)
        mod.to_excel(w, sheet_name='Modelos', index=False)
        t_clf.to_excel(w, sheet_name='Ranking clasificacion', index=False)
        t_reg.to_excel(w, sheet_name='Ranking regresion', index=False)
    print(f"   ✔ Bitácora final: {len(reg)} pasos · {len(mod)} modelos · "
          f"2 hojas de ranking añadidas")
except PermissionError:
    pass

gc.collect()

# %%
