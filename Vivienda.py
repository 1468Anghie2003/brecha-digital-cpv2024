# =============================================================================
#  LIMPIEZA Y PREPROCESAMIENTO - CPV 2024 VIVIENDA
#  Proyecto: Brecha digital en Bolivia
#
#  EJECUTAR CON Ctrl+Enter SOBRE CADA BLOQUE "# %%"  (nunca con el boton ▶)
#  Los graficos salen EN LINEA y quedan visibles, como en Colab.
# =============================================================================

# %%
# =============================================================================
#  CELDA 1 - CONFIGURACION Y CARGA
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Graficos EN LINEA dentro de la Interactive Window de VS Code (como Colab).
# Si el archivo se ejecuta como script normal, la instruccion se ignora.
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

# Cada figura se guarda ademas en disco para poder reutilizarla en el informe
CARPETA_FIG = "figuras"
os.makedirs(CARPETA_FIG, exist_ok=True)


def guardar(nombre):
    """Guarda la figura actual y la muestra en linea."""
    ruta = os.path.join(CARPETA_FIG, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"  [figura guardada] {ruta}")
    plt.show()


pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"

ARCHIVO = "Vivienda_CPV-2024.csv"
RUTA = ARCHIVO if os.path.exists(ARCHIVO) else os.path.join(
    r"C:\Users\HP\Desktop\ProyApSup255", ARCHIVO)

print("=" * 78)
print("  LIMPIEZA DEL DATASET DE VIVIENDA - CPV 2024")
print("=" * 78)
print(f"  Ruta: {RUTA}")


def leer_censo(ruta):
    """El INE publica con separador ';' y codificacion latina."""
    for sep in [';', ',', '\t', '|']:
        for enc in ['latin-1', 'utf-8', 'cp1252', 'utf-8-sig']:
            try:
                d = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)
                if d.shape[1] > 20:
                    print(f"  Lectura correcta con sep='{sep}' encoding='{enc}'")
                    return d
            except Exception:
                continue
    raise ValueError("No se pudo leer el archivo")


df = leer_censo(RUTA)
df.columns = df.columns.str.strip().str.lower()
crudo = df.copy()

print(f"\n  Registros : {len(df):,}")
print(f"  Variables : {df.shape[1]}")
print(f"  Memoria   : {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")


# %%
# =============================================================================
#  CELDA 2 - DIAGNOSTICO ANTES DE LIMPIAR
# =============================================================================
print("\n" + "=" * 78)
print("  DIAGNOSTICO INICIAL")
print("=" * 78)

COLS_SI_NO = [c for c in df.columns if c.startswith(('v18', 'v19'))]

diag = pd.DataFrame({'nulos': df.isna().sum(),
                     'pct_nulos': df.isna().mean() * 100})
diag['codigo_9'] = [(df[c] == 9).sum() if c in COLS_SI_NO else 0
                    for c in df.columns]
diag['pct_9'] = diag['codigo_9'] / len(df) * 100
diag['faltante_real'] = diag['nulos'] + diag['codigo_9']
diag['pct_faltante'] = diag['faltante_real'] / len(df) * 100

print("\n  TOP 15 VARIABLES CON MAS DATOS FALTANTES")
print(f"  {'Variable':<20}{'Nulos':>12}{'Codigo 9':>11}{'% total':>10}")
for k, r in diag.sort_values('pct_faltante', ascending=False).head(15).iterrows():
    if r['pct_faltante'] > 0:
        print(f"  {k:<20}{int(r['nulos']):>12,}{int(r['codigo_9']):>11,}"
              f"{r['pct_faltante']:>9.2f}%")

n_dup = df.duplicated().sum()
n_desoc = df['v02_condocup'].isin([3, 4, 5]).sum()
n_colect = df['v01_tipoviv'].between(7, 16).sum()
n_sinpers = (df['tot_pers'].fillna(0) < 1).sum()

print("\n  PROBLEMAS ESTRUCTURALES DETECTADOS")
print(f"    Filas duplicadas completas          : {n_dup:,}")
print(f"    Viviendas DESOCUPADAS (v02=3,4,5)   : {n_desoc:,} ({n_desoc/len(df)*100:.2f} %)")
print(f"    Viviendas COLECTIVAS (v01=7..16)    : {n_colect:,} ({n_colect/len(df)*100:.2f} %)")
print(f"    Registros sin personas (tot_pers<1) : {n_sinpers:,}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))

vis = diag[diag['pct_faltante'] > 0].sort_values('pct_faltante').tail(18)
if len(vis):
    ax[0].barh(vis.index, vis['pct_nulos'], color='#D1495B', label='Nulos')
    ax[0].barh(vis.index, vis['pct_9'], left=vis['pct_nulos'],
               color='#EDAE49', label='Código 9 (sin especificar)')
    ax[0].legend(fontsize=8)
ax[0].set_xlabel("% de registros")
ax[0].set_title("(a) Datos faltantes ANTES")
ax[0].tick_params(labelsize=7)

probs = pd.Series({'Duplicados': n_dup, 'Desocupadas': n_desoc,
                   'Colectivas': n_colect, 'Sin personas': n_sinpers})
ax[1].bar(probs.index, probs.values,
          color=['#8D99AE', '#D1495B', '#EDAE49', '#00798C'])
for i, v in enumerate(probs.values):
    ax[1].text(i, v, f"{v:,}", ha='center', va='bottom', fontsize=9,
               fontweight='bold')
ax[1].set_ylabel("Registros afectados")
ax[1].set_title("(b) Registros a descartar")
ax[1].tick_params(axis='x', rotation=20, labelsize=8)

# Escala logaritmica: las viviendas colectivas (cuarteles, carceles) registran
# miles de personas y aplastarian la escala lineal, dejando el grafico ilegible.
num = [c for c in ['tot_pers', 'v13_habitac', 'v14_dormit'] if c in df.columns]
ax[2].boxplot([df[c].dropna().clip(lower=0.5) for c in num], tick_labels=num)
ax[2].set_yscale('log')
ax[2].set_ylabel("Valor (escala logarítmica)")
ax[2].set_title("(c) Valores atípicos ANTES")
ax[2].tick_params(axis='x', rotation=15, labelsize=8)
ax[2].text(.5, .95, f"máx. tot_pers = {int(df['tot_pers'].max()):,}",
           transform=ax[2].transAxes, ha='center', fontsize=8, color='#D1495B')

plt.suptitle("DIAGNÓSTICO INICIAL DEL DATASET", fontsize=14, y=1.02)
plt.tight_layout()
guardar("01_diagnostico_inicial.png")


# %%
# =============================================================================
#  CELDA 3 - LIMPIEZA PASO A PASO
# =============================================================================
print("\n" + "=" * 78)
print("  PROCESO DE LIMPIEZA")
print("=" * 78)

bitacora = []


def registrar(paso, motivo, antes, despues):
    bitacora.append({'paso': paso, 'motivo': motivo,
                     'eliminados': antes - despues, 'quedan': despues})
    print(f"\n  [{len(bitacora)}] {paso}")
    print(f"      Motivo    : {motivo}")
    print(f"      Eliminados: {antes-despues:,}   Quedan: {despues:,}")


a = len(df)
df = df.drop_duplicates()
registrar("Eliminacion de duplicados",
          "Un mismo registro repetido sesga cualquier estadistica", a, len(df))

# v01_tipoviv 7 a 16 son hoteles, hospitales, cuarteles, carceles, personas
# en la calle y transito. No son hogares.
a = len(df)
df = df[df['v01_tipoviv'].between(1, 6)]
registrar("Filtro de viviendas particulares",
          "Los codigos 7-16 son viviendas colectivas, no hogares", a, len(df))

# v02_condocup: 0,1,2 = ocupada;  3,4,5 = desocupada.
a = len(df)
df = df[df['v02_condocup'].isin([0, 1, 2])]
registrar("Filtro de viviendas ocupadas",
          "Las desocupadas no tienen hogar que pueda tener internet", a, len(df))

a = len(df)
df = df[df['tot_pers'].fillna(0) >= 1]
registrar("Viviendas con al menos una persona",
          "Una vivienda ocupada debe registrar habitantes", a, len(df))

# El codigo 9 = "Sin especificar" segun el diccionario del INE
COLS_SI_NO = [c for c in df.columns if c.startswith(('v18', 'v19'))]
n9 = sum((df[c] == 9).sum() for c in COLS_SI_NO)
for c in COLS_SI_NO:
    df[c] = df[c].replace(9, np.nan)
print(f"\n  [{len(bitacora)+1}] Recodificacion del codigo 9")
print(f"      Motivo    : El diccionario define 9 = 'Sin especificar' (faltante)")
print(f"      Convertidos a nulo: {n9:,} valores en {len(COLS_SI_NO)} variables")
bitacora.append({'paso': 'Codigo 9 a nulo', 'motivo': 'Faltante codificado',
                 'eliminados': 0, 'quedan': len(df)})

print(f"\n  [{len(bitacora)+1}] Revision de rangos segun el diccionario")
for col, (lo, hi) in {'v13_habitac': (1, 8), 'v14_dormit': (0, 8),
                      'urbrur': (1, 2), 'tot_pers': (1, 30)}.items():
    if col in df.columns:
        fuera = (~df[col].between(lo, hi)) & df[col].notna()
        if fuera.sum():
            print(f"      {col}: {fuera.sum():,} valores fuera de [{lo},{hi}] -> nulo")
            df.loc[fuera, col] = np.nan
        else:
            print(f"      {col}: sin valores fuera de rango")
bitacora.append({'paso': 'Rangos validados', 'motivo': 'Coherencia con diccionario',
                 'eliminados': 0, 'quedan': len(df)})

if {'v13_habitac', 'v14_dormit'}.issubset(df.columns):
    incoh = (df['v14_dormit'] > df['v13_habitac'])
    print(f"\n  [{len(bitacora)+1}] Coherencia dormitorios vs habitaciones")
    print(f"      Registros con mas dormitorios que habitaciones: {incoh.sum():,}")
    if incoh.sum():
        df.loc[incoh, 'v14_dormit'] = np.nan
    bitacora.append({'paso': 'Coherencia dormitorios', 'motivo': 'Regla logica',
                     'eliminados': 0, 'quedan': len(df)})


# %%
# =============================================================================
#  CELDA 4 - VARIABLES DERIVADAS Y CONTROL DE FUGA DE INFORMACION
# =============================================================================
print("\n" + "=" * 78)
print("  CONSTRUCCION DE VARIABLES")
print("=" * 78)

df['internet'] = df['v19e_f'].map({1: 1, 2: 0})
print(f"\n  internet (objetivo de clasificacion)")
print(f"    Con internet : {(df['internet']==1).sum():,}")
print(f"    Sin internet : {(df['internet']==0).sum():,}")
print(f"    Sin dato     : {df['internet'].isna().sum():,}")

# Indice de Equipamiento del Hogar: el censo no registra ingresos, por lo que
# se emplea el criterio de "indice de riqueza" de las encuestas DHS.
BIENES = ['v18a_bici', 'v18b_moto', 'v18c_auto', 'v18f_refri', 'v18g_micro',
          'v18h_calefon', 'v18i_aire', 'v18j_lavadora', 'v19a_radio',
          'v19b_tv', 'v19c_compu', 'v19d_celular', 'v19g_tvcable']
BIENES = [c for c in BIENES if c in df.columns]
df['ieh'] = sum((df[c] == 1).astype(float) for c in BIENES)
print(f"\n  ieh (objetivo de regresion) con {len(BIENES)} bienes")
print(f"    Promedio : {df['ieh'].mean():.2f}")
print(f"    Urbana   : {df.loc[df.urbrur==1,'ieh'].mean():.2f}")
print(f"    Rural    : {df.loc[df.urbrur==2,'ieh'].mean():.2f}")

# v14_dormit = 0 es respuesta valida ("Cero"), no error: se usa clip(lower=1)
# para no dividir entre cero. Se recalcula con detalle en la celda 4B.
df['hacinamiento'] = df['tot_pers'] / df['v14_dormit'].clip(lower=1)
print(f"\n  hacinamiento = personas / dormitorios")
print(f"    Mediana  : {df['hacinamiento'].median():.2f} personas por dormitorio")

# v19e_inetfijo y v19f_inetmovil son los componentes con los que el INE
# construyo v19e_f. Usarlos como predictoras seria fuga de informacion.
print(f"\n  Variables excluidas por fuga de informacion:")
for c in ['v19e_inetfijo', 'v19f_inetmovil', 'v19e_f']:
    if c in df.columns:
        print(f"    {c}  (componente directo de la variable objetivo)")
print(f"  Identificadores excluidos: i00")

PREDICTORAS = ['urbrur', 'v01_tipoviv', 'v03_pared', 'v04_revoq', 'v05_techo',
               'v06_piso', 'v07_aguapro', 'v08_aguadist', 'v09_energia',
               'v10_combus', 'v11_basura', 'v12_cocina', 'v13_habitac',
               'v14_dormit', 'v15_servsan', 'v16_desague', 'v17_tenencia',
               'tot_pers', 'tip_hog', 'hacinamiento', 'idep']
PREDICTORAS = [c for c in PREDICTORAS if c in df.columns]
print(f"\n  Predictoras seleccionadas: {len(PREDICTORAS)}")


# %%
# =============================================================================
#  CELDA 4B - NULOS ESTRUCTURALES
#  Un nulo estructural es una pregunta que NO correspondia responder.
#  Imputarlo con la moda inventaria informacion.
# =============================================================================
print("\n" + "=" * 78)
print("  TRATAMIENTO DE NULOS ESTRUCTURALES")
print("=" * 78)

# La pregunta 16 ("El bano o letrina tiene desague") solo se aplica a quienes
# declararon tener bano en la pregunta 15. Si v15_servsan = 3 ("No tiene"),
# el vacio no es un dato perdido sino una respuesta no aplicable.
sin_bano = (df['v15_servsan'] == 3)
nulos_des = df['v16_desague'].isna()

print(f"\n  v16_desague")
print(f"    Nulos totales              : {nulos_des.sum():,}")
print(f"    De ellos, hogares SIN bano : {(nulos_des & sin_bano).sum():,}")
print(f"    Nulos reales (con bano)    : {(nulos_des & ~sin_bano).sum():,}")

df.loc[nulos_des & sin_bano, 'v16_desague'] = 0     # nueva categoria
if df['v16_desague'].isna().sum():
    df['v16_desague'] = df['v16_desague'].fillna(df['v16_desague'].mode()[0])
print(f"    Categoria 0 creada = 'No tiene bano / no aplica'")

df['hacinamiento'] = df['tot_pers'] / df['v14_dormit'].clip(lower=1)
print(f"\n  hacinamiento recalculado")
print(f"    Mediana : {df['hacinamiento'].median():.2f} personas por dormitorio")
print(f"    Maximo  : {df['hacinamiento'].max():.2f}")
print(f"    Nulos   : {df['hacinamiento'].isna().sum():,}")
crit = (df['hacinamiento'] > 3)
print(f"    Hacinamiento critico (>3 pers/dormitorio): {crit.sum():,} "
      f"({crit.mean()*100:.2f} %)")

prop = df['internet'].value_counts(normalize=True) * 100
print(f"\n  BALANCE DE LA VARIABLE OBJETIVO")
print(f"    Con internet : {prop.get(1.0, 0):.2f} %")
print(f"    Sin internet : {prop.get(0.0, 0):.2f} %")
print(f"    Cifra oficial INE (CPV 2024): 76.3 %")
if prop.get(1.0, 0) > 65:
    print(f"    [!] Clases desbalanceadas -> usar class_weight='balanced'")


# %%
# =============================================================================
#  CELDA 5 - TRATAMIENTO DE FALTANTES RESTANTES
# =============================================================================
print("\n" + "=" * 78)
print("  IMPUTACION")
print("=" * 78)

a = len(df)
df = df.dropna(subset=['internet'])
print(f"\n  Registros sin objetivo eliminados: {a-len(df):,}")

# Se imputa con la MODA porque casi todas las predictoras son categoricas
# codificadas: en v07_aguapro el codigo 3 no vale el triple que el codigo 1.
imputadas = []
for c in PREDICTORAS:
    n_na = df[c].isna().sum()
    if n_na:
        if c in ('tot_pers', 'hacinamiento'):
            val, metodo = df[c].median(), 'mediana'
        else:
            val, metodo = df[c].mode()[0], 'moda'
        df[c] = df[c].fillna(val)
        imputadas.append((c, n_na, metodo, val))

if imputadas:
    print(f"\n  {'Variable':<18}{'Faltantes':>12}{'Metodo':>10}{'Valor':>10}")
    for c, n_na, m, v in imputadas:
        print(f"  {c:<18}{n_na:>12,}{m:>10}{v:>10.2f}")
else:
    print("  No quedaron faltantes en las predictoras.")

limpio = df[PREDICTORAS + ['internet', 'ieh']].copy()
print(f"\n  DATASET FINAL: {len(limpio):,} registros x {limpio.shape[1]} variables")
print(f"  Nulos restantes: {limpio.isna().sum().sum()}")


# %%
# =============================================================================
#  CELDA 6 - DIAGNOSTICO DESPUES Y GUARDADO
# =============================================================================
print("\n" + "=" * 78)
print("  RESULTADO DE LA LIMPIEZA")
print("=" * 78)

print(f"\n  {'Etapa':<40}{'Registros':>14}{'% del original':>16}")
print(f"  {'Dataset original':<40}{len(crudo):>14,}{100:>15.2f}%")
for b in bitacora:
    if b['eliminados'] > 0:
        print(f"  {'- ' + b['paso']:<40}{-b['eliminados']:>14,}"
              f"{b['eliminados']/len(crudo)*100:>15.2f}%")
print(f"  {'Dataset limpio':<40}{len(limpio):>14,}"
      f"{len(limpio)/len(crudo)*100:>15.2f}%")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))

falt_desp = limpio.isna().mean() * 100
ax[0].barh(falt_desp.index, falt_desp.values, color='#00798C')
ax[0].set_xlim(0, max(1, falt_desp.max() * 1.2))
ax[0].set_xlabel("% de registros")
ax[0].set_title("(a) Datos faltantes DESPUÉS")
ax[0].tick_params(labelsize=7)
if falt_desp.sum() == 0:
    ax[0].text(.5, .5, "Sin datos faltantes", transform=ax[0].transAxes,
               ha='center', fontsize=13, fontweight='bold', color='#00798C')

comp = pd.Series({'Original': len(crudo), 'Limpio': len(limpio)})
ax[1].bar(comp.index, comp.values, color=['#8D99AE', '#00798C'])
for i, v in enumerate(comp.values):
    ax[1].text(i, v, f"{v:,}", ha='center', va='bottom', fontweight='bold')
ax[1].set_ylabel("Registros")
ax[1].set_title(f"(b) Retención: {len(limpio)/len(crudo)*100:.1f} %")

num2 = [c for c in ['tot_pers', 'v13_habitac', 'v14_dormit', 'hacinamiento']
        if c in limpio.columns]
ax[2].boxplot([limpio[c].dropna() for c in num2], tick_labels=num2)
ax[2].set_title("(c) Valores atípicos DESPUÉS")
ax[2].tick_params(axis='x', rotation=15, labelsize=8)

plt.suptitle("DATASET LIMPIO — Resultado del preprocesamiento",
             fontsize=14, y=1.02)
plt.tight_layout()
guardar("02_dataset_limpio.png")

limpio.to_csv("vivienda_cpv2024_limpio.csv", sep=';', index=False,
              encoding='utf-8')
print("\n  Guardado: vivienda_cpv2024_limpio.csv")
try:
    limpio.to_parquet("vivienda_cpv2024_limpio.parquet", index=False)
    print("  Guardado: vivienda_cpv2024_limpio.parquet (carga mas rapida)")
except Exception as e:
    print(f"  (Parquet no disponible: instalar con  pip install pyarrow)")

pd.DataFrame(bitacora).to_csv("bitacora_limpieza.csv", index=False,
                              encoding='utf-8')
print("  Guardado: bitacora_limpieza.csv (trazabilidad de cada decision)")


# %%
# =============================================================================
#  CELDA 7 - REPARACION DE v16_desague Y CARGA DEL DATASET LIMPIO
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

CARPETA_FIG = "figuras"
os.makedirs(CARPETA_FIG, exist_ok=True)


def guardar(nombre):
    ruta = os.path.join(CARPETA_FIG, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"  [figura guardada] {ruta}")
    plt.show()


plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"
COL = {'a': '#2E5EAA', 'b': '#D1495B', 'c': '#00798C',
       'd': '#EDAE49', 'g': '#8D99AE'}

datos = pd.read_csv("vivienda_cpv2024_limpio.csv", sep=';')
print("=" * 78)
print("  DATASET LIMPIO CARGADO")
print("=" * 78)
print(f"  Registros: {len(datos):,}   Variables: {datos.shape[1]}")

# --- REPARACION -------------------------------------------------------------
# La pregunta 16 ("El bano o letrina tiene desague") solo se aplica a quienes
# declararon tener bano. Los hogares con v15_servsan = 3 ("No tiene") fueron
# imputados con la moda (1 = alcantarillado), lo cual es imposible.
# Como esos registros no podian tener otro valor, la correccion es exacta.
mal = (datos['v15_servsan'] == 3)
print(f"\n  REPARACION DE v16_desague")
print(f"    Hogares sin bano (v15_servsan=3): {mal.sum():,}")
print(f"    Valor que tenian asignado       : {datos.loc[mal,'v16_desague'].unique()}")
datos.loc[mal, 'v16_desague'] = 0
print(f"    Corregidos a categoria 0 = 'No tiene bano / no aplica'")
print(f"\n  Distribucion corregida de v16_desague:")
etiq_des = {0: 'No tiene bano', 1: 'Alcantarillado', 2: 'Camara septica',
            3: 'Pozo ciego', 4: 'Pozo de absorcion', 5: 'A la superficie',
            6: 'Bano ecologico'}
for k, v in datos['v16_desague'].value_counts().sort_index().items():
    print(f"    {int(k)} = {etiq_des.get(int(k),'?'):<20} {v:>10,}  "
          f"({v/len(datos)*100:5.2f} %)")

datos.to_csv("vivienda_cpv2024_limpio.csv", sep=';', index=False,
             encoding='utf-8')
print(f"\n  Dataset corregido y regrabado.")

# --- Nombres legibles para los graficos --------------------------------------
NOMBRES = {
    'urbrur': 'Área urbana/rural', 'v01_tipoviv': 'Tipo de vivienda',
    'v03_pared': 'Material de pared', 'v04_revoq': 'Pared revocada',
    'v05_techo': 'Material de techo', 'v06_piso': 'Material de piso',
    'v07_aguapro': 'Procedencia del agua', 'v08_aguadist': 'Distribución de agua',
    'v09_energia': 'Fuente de energía', 'v10_combus': 'Combustible de cocina',
    'v11_basura': 'Manejo de basura', 'v12_cocina': 'Tiene cocina',
    'v13_habitac': 'N° de habitaciones', 'v14_dormit': 'N° de dormitorios',
    'v15_servsan': 'Servicio sanitario', 'v16_desague': 'Tipo de desagüe',
    'v17_tenencia': 'Tenencia de vivienda', 'tot_pers': 'Personas en el hogar',
    'tip_hog': 'Tipo de hogar', 'hacinamiento': 'Hacinamiento',
    'idep': 'Departamento',
}
PRED = [c for c in datos.columns if c not in ('internet', 'ieh')]
print(f"\n  Predictoras disponibles: {len(PRED)}")


# %%
import sys
!"{sys.executable}" -m pip install scikit-learn pyarrow
# %%
# =============================================================================
#  CELDA 8 - MUESTREO Y PARTICION
# =============================================================================
from sklearn.model_selection import train_test_split

print("=" * 78)
print("  PREPARACION PARA EL MODELADO")
print("=" * 78)

# Con 3.5 millones de registros, Gradient Boosting tardaria horas. Se toma una
# muestra ALEATORIA ESTRATIFICADA que conserva la proporcion de la variable
# objetivo, procedimiento estandar cuando el volumen excede la capacidad de
# computo disponible. El error de muestreo con n=500.000 es despreciable.
N_MUESTRA = min(500_000, len(datos))
muestra = datos.sample(n=N_MUESTRA, random_state=42)

print(f"\n  Poblacion total : {len(datos):,} viviendas")
print(f"  Muestra usada   : {len(muestra):,} ({len(muestra)/len(datos)*100:.1f} %)")
print(f"  Proporcion con internet -> poblacion: "
      f"{datos['internet'].mean()*100:.2f} %  muestra: "
      f"{muestra['internet'].mean()*100:.2f} %")

X = muestra[PRED]
y_clf = muestra['internet'].astype(int)
y_reg = muestra['ieh']

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X, y_reg, test_size=0.25, random_state=42)

print(f"\n  CLASIFICACION  entrenamiento: {len(Xc_tr):,}  prueba: {len(Xc_te):,}")
print(f"  REGRESION      entrenamiento: {len(Xr_tr):,}  prueba: {len(Xr_te):,}")

des = y_clf.value_counts(normalize=True) * 100
print(f"\n  Balance de clases: {des.get(1,0):.1f} % con internet / "
      f"{des.get(0,0):.1f} % sin internet")
print(f"  Se aplicara class_weight='balanced' para compensar el desbalance.")
# %%
# %%
# =============================================================================
#  CELDA 9 - CLASIFICACION: ARBOL, RANDOM FOREST Y GRADIENT BOOSTING
# =============================================================================
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

print("=" * 78)
print("  MODELOS DE CLASIFICACION - ¿La vivienda tiene internet?")
print("=" * 78)

modelos_clf = {
    'Árbol de decisión': DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=100, class_weight='balanced',
        random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=50,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingClassifier(
        max_iter=200, max_depth=8, learning_rate=0.1,
        class_weight='balanced', random_state=42),
}

res_clf, probas = {}, {}
for nombre, mod in modelos_clf.items():
    t0 = time.time()
    mod.fit(Xc_tr, yc_tr)
    seg = time.time() - t0
    pred = mod.predict(Xc_te)
    prob = mod.predict_proba(Xc_te)[:, 1]
    probas[nombre] = prob
    res_clf[nombre] = {
        'Exactitud': accuracy_score(yc_te, pred),
        'Precisión': precision_score(yc_te, pred),
        'Recall (con internet)': recall_score(yc_te, pred),
        'Recall (SIN internet)': recall_score(yc_te, pred, pos_label=0),
        'F1': f1_score(yc_te, pred),
        'AUC-ROC': roc_auc_score(yc_te, prob),
        'Segundos': seg,
        'pred': pred,
    }
    print(f"\n  {nombre}  (entrenado en {seg:.1f} s)")
    print(f"    Exactitud             : {res_clf[nombre]['Exactitud']:.4f}")
    print(f"    AUC-ROC               : {res_clf[nombre]['AUC-ROC']:.4f}")
    print(f"    Recall clase SIN internet: {res_clf[nombre]['Recall (SIN internet)']:.4f}")

tabla = pd.DataFrame({k: {m: v for m, v in d.items() if m != 'pred'}
                      for k, d in res_clf.items()}).T
print("\n" + "=" * 78)
print("  COMPARACION DE MODELOS DE CLASIFICACION")
print("=" * 78)
print(tabla.round(4).to_string())

mejor_clf = tabla['AUC-ROC'].idxmax()
print(f"\n  Mejor modelo por AUC: {mejor_clf} ({tabla.loc[mejor_clf,'AUC-ROC']:.4f})")

print(f"\n  INFORME DETALLADO DEL MEJOR MODELO ({mejor_clf})")
print(classification_report(yc_te, res_clf[mejor_clf]['pred'],
                            target_names=['Sin internet', 'Con internet']))
# %%
# %%
# =============================================================================
#  CELDA 10 - GRAFICOS DE CLASIFICACION
# =============================================================================

# ---------- G1: curvas ROC comparadas ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

for (nombre, prob), c in zip(probas.items(), [COL['a'], COL['b'], COL['c']]):
    fpr, tpr, _ = roc_curve(yc_te, prob)
    ax[0].plot(fpr, tpr, lw=2.5, color=c,
               label=f"{nombre} (AUC={res_clf[nombre]['AUC-ROC']:.3f})")
ax[0].plot([0, 1], [0, 1], '--', color='gray', lw=2, label='Azar (0.500)')
ax[0].set_xlabel("Tasa de falsos positivos")
ax[0].set_ylabel("Tasa de verdaderos positivos")
ax[0].set_title("(a) Curvas ROC comparadas")
ax[0].legend(fontsize=8, loc='lower right')
ax[0].grid(alpha=.3)

metricas = ['Exactitud', 'Precisión', 'Recall (con internet)',
            'Recall (SIN internet)', 'F1', 'AUC-ROC']
x = np.arange(len(metricas))
w = 0.26
for i, (nombre, c) in enumerate(zip(res_clf, [COL['a'], COL['b'], COL['c']])):
    vals = [res_clf[nombre][m] for m in metricas]
    ax[1].bar(x + (i - 1) * w, vals, w, label=nombre, color=c)
ax[1].set_xticks(x)
ax[1].set_xticklabels([m.replace(' (', '\n(') for m in metricas],
                      fontsize=7, rotation=20)
ax[1].set_ylim(0, 1.05)
ax[1].set_ylabel("Valor")
ax[1].set_title("(b) Métricas por modelo")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3, axis='y')

tiempos = [res_clf[n]['Segundos'] for n in res_clf]
ax[2].barh(list(res_clf.keys()), tiempos,
           color=[COL['a'], COL['b'], COL['c']])
for i, v in enumerate(tiempos):
    ax[2].text(v, i, f" {v:.1f}s", va='center', fontsize=9, fontweight='bold')
ax[2].set_xlabel("Segundos de entrenamiento")
ax[2].set_title("(c) Costo computacional")
ax[2].grid(alpha=.3, axis='x')

plt.suptitle("CLASIFICACIÓN — Comparación de los tres algoritmos",
             fontsize=14, y=1.02)
plt.tight_layout()
guardar("03_clasificacion_comparacion.png")

# ---------- G2: matrices de confusion ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for i, (nombre, c) in enumerate(zip(res_clf, ['Blues', 'Reds', 'GnBu'])):
    mc = confusion_matrix(yc_te, res_clf[nombre]['pred'])
    ax[i].imshow(mc, cmap=c)
    ax[i].set_xticks([0, 1], ['Pred: sin', 'Pred: con'])
    ax[i].set_yticks([0, 1], ['Real: sin', 'Real: con'])
    for a in range(2):
        for b in range(2):
            ax[i].text(b, a, f"{mc[a,b]:,}", ha='center', va='center',
                       fontsize=13,
                       color='white' if mc[a, b] > mc.max()/2 else 'black')
    ax[i].set_title(f"{nombre}\nExactitud {res_clf[nombre]['Exactitud']:.3f}",
                    fontsize=10)
    ax[i].grid(False)
plt.suptitle("CLASIFICACIÓN — Matrices de confusión", fontsize=14, y=1.03)
plt.tight_layout()
guardar("04_clasificacion_matrices.png")

# ---------- G3: importancia de variables ----------
rf = modelos_clf['Random Forest']
imp = pd.Series(rf.feature_importances_, index=PRED).sort_values()
imp.index = [NOMBRES.get(i, i) for i in imp.index]

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].barh(imp.index, imp.values, color=COL['a'])
ax[0].set_xlabel("Importancia relativa")
ax[0].set_title("(a) Variables que determinan el acceso a internet")
ax[0].tick_params(labelsize=8)
ax[0].grid(alpha=.3, axis='x')

arbol = modelos_clf['Árbol de decisión']
plot_tree(arbol, max_depth=2, filled=True, rounded=True, fontsize=8,
          feature_names=[NOMBRES.get(c, c) for c in PRED],
          class_names=['Sin internet', 'Con internet'],
          proportion=True, precision=2, ax=ax[1])
ax[1].set_title("(b) Primeros niveles del árbol de decisión")

plt.suptitle("CLASIFICACIÓN — Interpretación del modelo", fontsize=14, y=1.00)
plt.tight_layout()
guardar("05_clasificacion_importancia.png")

print("\n  IMPORTANCIA DE VARIABLES (Random Forest)")
for k, v in imp.sort_values(ascending=False).items():
    print(f"    {k:<26} {v:.4f}")
# %%
# %%
# =============================================================================
#  CELDA 11 - REGRESION: ARBOL, RANDOM FOREST Y GRADIENT BOOSTING
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 78)
print("  MODELOS DE REGRESION - Indice de Equipamiento del Hogar (0 a 13)")
print("=" * 78)

modelos_reg = {
    'Árbol de regresión': DecisionTreeRegressor(
        max_depth=10, min_samples_leaf=100, random_state=42),
    'Random Forest': RandomForestRegressor(
        n_estimators=150, max_depth=18, min_samples_leaf=50,
        random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingRegressor(
        max_iter=200, max_depth=8, learning_rate=0.1, random_state=42),
}

res_reg, preds_reg = {}, {}
for nombre, mod in modelos_reg.items():
    t0 = time.time()
    mod.fit(Xr_tr, yr_tr)
    seg = time.time() - t0
    p = mod.predict(Xr_te)
    preds_reg[nombre] = p
    res_reg[nombre] = {
        'RMSE': np.sqrt(mean_squared_error(yr_te, p)),
        'MAE': mean_absolute_error(yr_te, p),
        'R2': r2_score(yr_te, p),
        'Segundos': seg,
    }
    print(f"\n  {nombre}  (entrenado en {seg:.1f} s)")
    print(f"    RMSE : {res_reg[nombre]['RMSE']:.4f} bienes")
    print(f"    MAE  : {res_reg[nombre]['MAE']:.4f} bienes")
    print(f"    R2   : {res_reg[nombre]['R2']:.4f}")

tabla_reg = pd.DataFrame(res_reg).T
print("\n" + "=" * 78)
print("  COMPARACION DE MODELOS DE REGRESION")
print("=" * 78)
print(tabla_reg.round(4).to_string())

mejor_reg = tabla_reg['R2'].idxmax()
print(f"\n  Mejor modelo por R2: {mejor_reg} ({tabla_reg.loc[mejor_reg,'R2']:.4f})")
print(f"  Desviacion estandar del indice: {yr_te.std():.4f}")
print(f"  Error de predecir siempre la media: {yr_te.std():.4f} bienes")
print(f"  Mejora del modelo sobre esa referencia: "
      f"{(1 - tabla_reg.loc[mejor_reg,'RMSE']/yr_te.std())*100:.1f} %")
# %%
# %%
# =============================================================================
# INSTALACIÓN DE LIGHTGBM
# =============================================================================

import sys

!"{sys.executable}" -m pip install lightgbm -q

print("LightGBM instalado correctamente.")
# %%
from lightgbm import LGBMClassifier, LGBMRegressor
# %%
Xc_tr, Xc_te, yc_tr, yc_te
# %%
# %%
# =============================================================================
# CELDA 9B - BOOSTING: ADABOOST Y LIGHTGBM
# CLASIFICACIÓN: ¿LA VIVIENDA TIENE INTERNET?
# =============================================================================

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from lightgbm import LGBMClassifier

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 78)
print("  BOOSTING - CLASIFICACIÓN")
print("  Objetivo: ¿La vivienda tiene acceso a Internet?")
print("=" * 78)


# -----------------------------------------------------------------------------
# 1. BALANCE DE CLASES
# -----------------------------------------------------------------------------

n_sin = (yc_tr == 0).sum()
n_con = (yc_tr == 1).sum()

peso_positivo = n_sin / n_con

print(f"\n  Casos SIN internet : {n_sin:,}")
print(f"  Casos CON internet : {n_con:,}")
print(f"  scale_pos_weight   : {peso_positivo:.3f}")


# -----------------------------------------------------------------------------
# 2. MODELOS BOOSTING
# -----------------------------------------------------------------------------

# AdaBoost utiliza árboles pequeños como modelos débiles.
modelo_adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=2,
        random_state=42,
        class_weight='balanced'
    ),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)


# LightGBM utiliza Gradient Boosting basado en árboles.
modelo_lightgbm = LGBMClassifier(
    objective='binary',
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=peso_positivo,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


modelos_boost_clf = {
    'AdaBoost': modelo_adaboost,
    'LightGBM': modelo_lightgbm
}


# -----------------------------------------------------------------------------
# 3. ENTRENAMIENTO Y EVALUACIÓN
# -----------------------------------------------------------------------------

res_boost_clf = {}
probas_boost_clf = {}
preds_boost_clf = {}

for nombre, modelo in modelos_boost_clf.items():

    print(f"\n{'-' * 65}")
    print(f"  Entrenando: {nombre}")
    print(f"{'-' * 65}")

    inicio = time.time()

    modelo.fit(Xc_tr, yc_tr)

    segundos = time.time() - inicio

    pred = modelo.predict(Xc_te)
    prob = modelo.predict_proba(Xc_te)[:, 1]

    preds_boost_clf[nombre] = pred
    probas_boost_clf[nombre] = prob

    res_boost_clf[nombre] = {
        'Exactitud': accuracy_score(yc_te, pred),
        'Precisión': precision_score(yc_te, pred, zero_division=0),
        'Recall (con internet)': recall_score(
            yc_te, pred, zero_division=0
        ),
        'Recall (SIN internet)': recall_score(
            yc_te, pred, pos_label=0, zero_division=0
        ),
        'F1': f1_score(yc_te, pred, zero_division=0),
        'AUC-ROC': roc_auc_score(yc_te, prob),
        'Segundos': segundos
    }

    print(f"  Tiempo             : {segundos:.2f} s")
    print(f"  Exactitud          : {res_boost_clf[nombre]['Exactitud']:.4f}")
    print(f"  Precisión          : {res_boost_clf[nombre]['Precisión']:.4f}")
    print(f"  Recall CON internet: "
          f"{res_boost_clf[nombre]['Recall (con internet)']:.4f}")
    print(f"  Recall SIN internet: "
          f"{res_boost_clf[nombre]['Recall (SIN internet)']:.4f}")
    print(f"  F1                 : {res_boost_clf[nombre]['F1']:.4f}")
    print(f"  AUC-ROC            : {res_boost_clf[nombre]['AUC-ROC']:.4f}")


# -----------------------------------------------------------------------------
# 4. TABLA DE RESULTADOS
# -----------------------------------------------------------------------------

tabla_boost_clf = pd.DataFrame(res_boost_clf).T

print("\n" + "=" * 78)
print("  COMPARACIÓN: ADABOOST VS LIGHTGBM")
print("=" * 78)

print(tabla_boost_clf.round(4).to_string())

mejor_boost_clf = tabla_boost_clf['AUC-ROC'].idxmax()

print(f"\n  Mejor Boosting por AUC-ROC:")
print(f"  {mejor_boost_clf} -> "
      f"{tabla_boost_clf.loc[mejor_boost_clf, 'AUC-ROC']:.4f}")


# -----------------------------------------------------------------------------
# 5. INFORME DEL MEJOR BOOSTING
# -----------------------------------------------------------------------------

print("\n" + "=" * 78)
print(f"  INFORME DETALLADO: {mejor_boost_clf}")
print("=" * 78)

print(
    classification_report(
        yc_te,
        preds_boost_clf[mejor_boost_clf],
        target_names=['Sin internet', 'Con internet'],
        zero_division=0
    )
)
# %%
# %%
# =============================================================================
# CELDA 9C - GRÁFICAS DE BOOSTING PARA CLASIFICACIÓN
# =============================================================================

# -----------------------------------------------------------------------------
# GRÁFICA 1 - CURVAS ROC
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

for nombre, prob in probas_boost_clf.items():

    fpr, tpr, _ = roc_curve(yc_te, prob)

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{nombre} "
              f"(AUC = {res_boost_clf[nombre]['AUC-ROC']:.3f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    '--',
    linewidth=1.5,
    label='Azar (AUC = 0.500)'
)

ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Boosting — Curvas ROC para acceso a Internet")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
guardar("06_boosting_clasificacion_roc.png")


# -----------------------------------------------------------------------------
# GRÁFICA 2 - COMPARACIÓN DE MÉTRICAS
# -----------------------------------------------------------------------------

metricas = [
    'Exactitud',
    'Precisión',
    'Recall (con internet)',
    'Recall (SIN internet)',
    'F1',
    'AUC-ROC'
]

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(metricas))
ancho = 0.35

nombres = list(res_boost_clf.keys())

for i, nombre in enumerate(nombres):

    valores = [
        res_boost_clf[nombre][m]
        for m in metricas
    ]

    desplazamiento = (i - 0.5) * ancho

    ax.bar(
        x + desplazamiento,
        valores,
        ancho,
        label=nombre
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [
        'Exactitud',
        'Precisión',
        'Recall\ncon Internet',
        'Recall\nsin Internet',
        'F1',
        'AUC-ROC'
    ]
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Valor")
ax.set_title("Comparación de AdaBoost y LightGBM")
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
guardar("07_boosting_clasificacion_metricas.png")


# -----------------------------------------------------------------------------
# GRÁFICA 3 - MATRICES DE CONFUSIÓN
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, nombre in zip(axes, modelos_boost_clf.keys()):

    mc = confusion_matrix(
        yc_te,
        preds_boost_clf[nombre]
    )

    im = ax.imshow(mc)

    ax.set_xticks(
        [0, 1],
        ['Pred: sin', 'Pred: con']
    )

    ax.set_yticks(
        [0, 1],
        ['Real: sin', 'Real: con']
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                f"{mc[i, j]:,}",
                ha='center',
                va='center',
                fontsize=13
            )

    ax.set_title(
        f"{nombre}\n"
        f"AUC = {res_boost_clf[nombre]['AUC-ROC']:.3f}"
    )

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

plt.suptitle(
    "Matrices de confusión — Boosting",
    fontsize=14
)

plt.tight_layout()
guardar("08_boosting_clasificacion_confusion.png")


# -----------------------------------------------------------------------------
# GRÁFICA 4 - IMPORTANCIA DE VARIABLES
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6))


for ax, nombre in zip(
    axes,
    modelos_boost_clf.keys()
):

    modelo = modelos_boost_clf[nombre]

    importancia = pd.Series(
        modelo.feature_importances_,
        index=PRED
    ).sort_values(ascending=True)

    # Mostrar solamente las 10 más importantes
    importancia = importancia.tail(10)

    importancia.index = [
        NOMBRES.get(c, c)
        for c in importancia.index
    ]

    ax.barh(
        importancia.index,
        importancia.values
    )

    ax.set_title(
        f"{nombre} — 10 variables más importantes"
    )

    ax.set_xlabel("Importancia")


plt.suptitle(
    "Importancia de variables — Modelos Boosting",
    fontsize=14
)

plt.tight_layout()
guardar("09_boosting_clasificacion_importancia.png")
# %%
y_reg = muestra['ieh']
# %%
# %%
# =============================================================================
# CELDA 11B - BOOSTING: ADABOOST Y LIGHTGBM
# REGRESIÓN: ÍNDICE DE EQUIPAMIENTO DEL HOGAR
# =============================================================================

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from lightgbm import LGBMRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


print("=" * 78)
print("  BOOSTING - REGRESIÓN")
print("  Objetivo: predecir el índice de equipamiento del hogar")
print("  Rango esperado: 0 a 13 bienes")
print("=" * 78)


# -----------------------------------------------------------------------------
# 1. MODELOS
# -----------------------------------------------------------------------------

modelo_adaboost_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=50,
        random_state=42
    ),
    n_estimators=100,
    learning_rate=0.05,
    loss='square',
    random_state=42
)


modelo_lightgbm_reg = LGBMRegressor(
    objective='regression',
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


modelos_boost_reg = {
    'AdaBoost': modelo_adaboost_reg,
    'LightGBM': modelo_lightgbm_reg
}


# -----------------------------------------------------------------------------
# 2. ENTRENAMIENTO Y EVALUACIÓN
# -----------------------------------------------------------------------------

res_boost_reg = {}
preds_boost_reg = {}

for nombre, modelo in modelos_boost_reg.items():

    print(f"\n{'-' * 65}")
    print(f"  Entrenando: {nombre}")
    print(f"{'-' * 65}")

    inicio = time.time()

    modelo.fit(Xr_tr, yr_tr)

    segundos = time.time() - inicio

    pred = modelo.predict(Xr_te)

    # Como el objetivo real está entre 0 y 13,
    # guardamos también una versión acotada.
    pred_acotada = np.clip(pred, 0, 13)

    preds_boost_reg[nombre] = pred_acotada

    rmse = np.sqrt(
        mean_squared_error(yr_te, pred_acotada)
    )

    mae = mean_absolute_error(
        yr_te,
        pred_acotada
    )

    r2 = r2_score(
        yr_te,
        pred_acotada
    )

    res_boost_reg[nombre] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Segundos': segundos
    }

    print(f"  Tiempo : {segundos:.2f} s")
    print(f"  RMSE   : {rmse:.4f} bienes")
    print(f"  MAE    : {mae:.4f} bienes")
    print(f"  R²     : {r2:.4f}")


# -----------------------------------------------------------------------------
# 3. TABLA COMPARATIVA
# -----------------------------------------------------------------------------

tabla_boost_reg = pd.DataFrame(
    res_boost_reg
).T

print("\n" + "=" * 78)
print("  COMPARACIÓN: ADABOOST VS LIGHTGBM")
print("=" * 78)

print(
    tabla_boost_reg.round(4).to_string()
)


mejor_boost_reg = tabla_boost_reg['R2'].idxmax()

print(f"\n  Mejor Boosting por R²:")
print(
    f"  {mejor_boost_reg} -> "
    f"{tabla_boost_reg.loc[mejor_boost_reg, 'R2']:.4f}"
)
# %%
# %%
# =============================================================================
# CELDA 11C - GRÁFICAS DE BOOSTING PARA REGRESIÓN
# =============================================================================


# -----------------------------------------------------------------------------
# GRÁFICA 1 - MÉTRICAS
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

nombres = list(res_boost_reg.keys())

rmse = [
    res_boost_reg[n]['RMSE']
    for n in nombres
]

mae = [
    res_boost_reg[n]['MAE']
    for n in nombres
]

r2 = [
    res_boost_reg[n]['R2']
    for n in nombres
]


axes[0].bar(nombres, rmse)
axes[0].set_title("RMSE")
axes[0].set_ylabel("Error en bienes")
axes[0].grid(alpha=0.3, axis='y')


axes[1].bar(nombres, mae)
axes[1].set_title("MAE")
axes[1].set_ylabel("Error medio en bienes")
axes[1].grid(alpha=0.3, axis='y')


axes[2].bar(nombres, r2)
axes[2].set_title("R²")
axes[2].set_ylabel("Coeficiente R²")
axes[2].grid(alpha=0.3, axis='y')


plt.suptitle(
    "Boosting — Comparación de modelos de regresión",
    fontsize=14
)

plt.tight_layout()
guardar("10_boosting_regresion_metricas.png")


# -----------------------------------------------------------------------------
# GRÁFICA 2 - REAL VS PREDICHO
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))


for ax, nombre in zip(
    axes,
    modelos_boost_reg.keys()
):

    reales = yr_te
    predichos = preds_boost_reg[nombre]

    ax.scatter(
        reales,
        predichos,
        alpha=0.15,
        s=8
    )

    ax.plot(
        [0, 13],
        [0, 13],
        '--',
        linewidth=2
    )

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)

    ax.set_xlabel("Bienes reales")
    ax.set_ylabel("Bienes predichos")

    ax.set_title(
        f"{nombre}\n"
        f"R² = {res_boost_reg[nombre]['R2']:.3f}"
    )

    ax.grid(alpha=0.3)


plt.suptitle(
    "Regresión — Valores reales vs. predichos",
    fontsize=14
)

plt.tight_layout()
guardar("11_boosting_regresion_real_predicho.png")


# -----------------------------------------------------------------------------
# GRÁFICA 3 - RESIDUOS
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))


for ax, nombre in zip(
    axes,
    modelos_boost_reg.keys()
):

    reales = yr_te
    predichos = preds_boost_reg[nombre]

    residuos = reales - predichos

    ax.scatter(
        predichos,
        residuos,
        alpha=0.15,
        s=8
    )

    ax.axhline(
        0,
        linestyle='--',
        linewidth=2
    )

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Residuo")

    ax.set_title(
        f"{nombre} — Residuos"
    )

    ax.grid(alpha=0.3)


plt.suptitle(
    "Regresión — Análisis de residuos",
    fontsize=14
)

plt.tight_layout()
guardar("12_boosting_regresion_residuos.png")


# -----------------------------------------------------------------------------
# GRÁFICA 4 - IMPORTANCIA DE VARIABLES
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6))


for ax, nombre in zip(
    axes,
    modelos_boost_reg.keys()
):

    modelo = modelos_boost_reg[nombre]

    importancia = pd.Series(
        modelo.feature_importances_,
        index=PRED
    ).sort_values(ascending=True)

    importancia = importancia.tail(10)

    importancia.index = [
        NOMBRES.get(c, c)
        for c in importancia.index
    ]

    ax.barh(
        importancia.index,
        importancia.values
    )

    ax.set_title(
        f"{nombre} — 10 variables más importantes"
    )

    ax.set_xlabel("Importancia")


plt.suptitle(
    "Regresión — Importancia de variables",
    fontsize=14
)

plt.tight_layout()
guardar("13_boosting_regresion_importancia.png")
# %%
