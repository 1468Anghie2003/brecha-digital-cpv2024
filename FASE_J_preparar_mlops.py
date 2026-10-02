# =============================================================================
#  FASE J · PREPARACIÓN DE ARTEFACTOS PARA EL SISTEMA MLOPS
#  Entrena las versiones del modelo, mide la deriva geográfica y deja todo
#  listo en la carpeta mlops/ para que la aplicación web la consuma.
#  EJECUTAR UNA SOLA VEZ · random_state = 42
# =============================================================================

# %%
# =============================================================================
#  J0 — CONFIGURACIÓN
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
import joblib

warnings.filterwarnings("ignore")
SEMILLA = 42
DIR = "mlops"
os.makedirs(os.path.join(DIR, "modelos"), exist_ok=True)

XGB = importlib.util.find_spec("xgboost") is not None
LGB = importlib.util.find_spec("lightgbm") is not None
CAT = importlib.util.find_spec("catboost") is not None

VERSIONES = {"v1.0": 80_000, "v2.0": 200_000}

CATEGORICAS = ['urbrur', 'idep', 'v01_tipoviv', 'v03_pared', 'v04_revoq',
               'v05_techo', 'v06_piso', 'v07_aguapro', 'v08_aguadist',
               'v09_energia', 'v10_combus', 'v11_basura', 'v12_cocina',
               'v15_servsan', 'v16_desague', 'v17_tenencia', 'tip_hog',
               'v20a_emi', 'v21a_fal']
NUMERICAS = ['v13_habitac', 'v14_dormit', 'tot_pers', 'v20b_totemi',
             'v21b_totfal', 'hacinamiento', 'pers_por_habitac',
             'hacin_critico', 'sin_servicios']

# Orden oficial del INE
DEPTOS = {1: 'Chuquisaca', 2: 'La Paz', 3: 'Cochabamba', 4: 'Oruro',
          5: 'Potosí', 6: 'Tarija', 7: 'Santa Cruz', 8: 'Beni', 9: 'Pando'}
COORD = {1: (-19.05, -65.26), 2: (-16.50, -68.15), 3: (-17.39, -66.16),
         4: (-17.97, -67.11), 5: (-19.58, -65.75), 6: (-21.53, -64.73),
         7: (-17.78, -63.18), 8: (-14.83, -64.90), 9: (-11.02, -68.77)}

print("═" * 78)
print("  FASE J · PREPARACIÓN DE ARTEFACTOS MLOPS")
print("═" * 78)
print(f"  XGBoost {XGB} · LightGBM {LGB} · CatBoost {CAT}")

BASE = "vivienda_cpv2024_LIMPIO_v2"
datos = pd.read_parquet(f"{BASE}.parquet") if os.path.exists(f"{BASE}.parquet") \
    else pd.read_csv(f"{BASE}.csv", sep=';')
for c in datos.select_dtypes(include=['float64']).columns:
    datos[c] = datos[c].astype('float32')
PRED = [c for c in datos.columns if c not in ('internet', 'ieh', 'nivel_equip')]
CATEGORICAS = [c for c in CATEGORICAS if c in PRED]
NUMERICAS = [c for c in NUMERICAS if c in PRED]
N_TOTAL = len(datos)
print(f"  Dataset: {N_TOTAL:,} viviendas · {len(PRED)} predictoras")


# %%
# =============================================================================
#  J1 — PANORAMA DEL DATASET (insumo de la primera página)
# =============================================================================
print("\n── J1 · Panorama ──")
NOMBRES = {
    'urbrur': 'Área urbana/rural', 'idep': 'Departamento',
    'v01_tipoviv': 'Tipo de vivienda', 'v03_pared': 'Material de pared',
    'v04_revoq': 'Pared revocada', 'v05_techo': 'Material de techo',
    'v06_piso': 'Material de piso', 'v07_aguapro': 'Procedencia del agua',
    'v08_aguadist': 'Distribución de agua', 'v09_energia': 'Fuente de energía',
    'v10_combus': 'Combustible de cocina', 'v11_basura': 'Manejo de basura',
    'v12_cocina': 'Tiene cocina', 'v13_habitac': 'N° de habitaciones',
    'v14_dormit': 'N° de dormitorios', 'v15_servsan': 'Servicio sanitario',
    'v16_desague': 'Tipo de desagüe', 'v17_tenencia': 'Tenencia',
    'tot_pers': 'Personas en el hogar', 'tip_hog': 'Tipo de hogar',
    'v20a_emi': 'Hubo emigración', 'v20b_totemi': 'N° de emigrantes',
    'v21a_fal': 'Hubo fallecimiento', 'v21b_totfal': 'N° fallecimientos',
    'hacinamiento': 'Personas por dormitorio',
    'hacin_critico': 'Hacinamiento crítico',
    'sin_servicios': 'Carencia de servicios',
    'pers_por_habitac': 'Personas por habitación',
    'internet': 'Acceso a internet', 'ieh': 'Índice de equipamiento'}

ETIQ = {
    'urbrur': {1: 'Urbana', 2: 'Rural'},
    'v03_pared': {1: 'Ladrillo/bloque', 2: 'Adobe/tapial', 3: 'Tabique',
                  4: 'Piedra', 5: 'Madera', 6: 'Caña/palma', 7: 'Otro'},
    'v06_piso': {1: 'Tierra', 2: 'Tablón', 3: 'Machimbre', 4: 'Cerámica',
                 5: 'Cemento', 6: 'Mosaico', 7: 'Ladrillo', 8: 'Flotante',
                 9: 'Otro'},
    'v07_aguapro': {1: 'Cañería de red', 2: 'Pileta pública', 3: 'Lluvia',
                    4: 'Pozo con bomba', 5: 'Pozo sin bomba',
                    6: 'Manantial', 7: 'Río/vertiente', 8: 'Carro aguatero',
                    9: 'Otro'},
    'v09_energia': {1: 'Red pública', 2: 'Generador', 3: 'Panel solar',
                    4: 'Otro', 5: 'No tiene'},
    'v10_combus': {1: 'Gas en garrafa', 2: 'Gas por cañería', 3: 'Leña',
                   4: 'Guano/taquia', 5: 'Electricidad', 6: 'Solar',
                   7: 'Otro', 8: 'No cocina'},
    'v11_basura': {1: 'Contenedor', 2: 'Carro basurero', 3: 'Terreno baldío',
                   4: 'Río', 5: 'La queman', 6: 'La entierran', 7: 'Otra'},
    'v15_servsan': {1: 'Sí, exclusivo', 2: 'Sí, compartido', 3: 'No tiene'},
    'v16_desague': {0: 'Sin baño', 1: 'Alcantarillado', 2: 'Cámara séptica',
                    3: 'Pozo ciego', 4: 'Pozo absorción', 5: 'Superficie',
                    6: 'Baño ecológico'},
    'tip_hog': {1: 'Unipersonal', 2: 'Pareja sin hijos', 3: 'Monoparental',
                4: 'Nuclear', 5: 'Extendido', 6: 'Compuesto', 7: 'Otro',
                8: 'Sin jefe'},
    'v04_revoq': {1: 'Sí', 2: 'No'}, 'v12_cocina': {1: 'Sí', 2: 'No'},
    'v20a_emi': {1: 'Sí', 2: 'No'}, 'v21a_fal': {1: 'Sí', 2: 'No'},
    'v05_techo': {1: 'Calamina', 2: 'Teja', 3: 'Losa', 4: 'Paja', 5: 'Otro'},
    'v08_aguadist': {1: 'Dentro de la vivienda', 2: 'Fuera, en el lote',
                     3: 'Fuera del lote'},
    'v01_tipoviv': {1: 'Casa', 2: 'Choza', 3: 'Departamento',
                    4: 'Cuarto suelto', 5: 'Vivienda improvisada',
                    6: 'No destinada a vivienda'},
    'v17_tenencia': {1: 'Propia', 2: 'En proceso de pago', 3: 'Alquilada',
                     4: 'Cedida', 5: 'Anticrético', 6: 'Mixta',
                     7: 'Contrato', 8: 'Otra'},
    'hacin_critico': {0: 'No', 1: 'Sí'}, 'sin_servicios': {0: 'No', 1: 'Sí'},
    'idep': DEPTOS}

pan = {
    'n_total': int(N_TOTAL), 'n_predictoras': len(PRED),
    'pct_internet': round(float(datos['internet'].mean()*100), 2),
    'ieh_medio': round(float(datos['ieh'].mean()), 3),
    'pct_rural': round(float((datos['urbrur'] == 2).mean()*100), 2),
    'pct_hacin': round(float(datos['hacin_critico'].mean()*100), 2),
    'pct_sin_serv': round(float(datos['sin_servicios'].mean()*100), 2),
    'nombres': NOMBRES,
    'etiquetas': {k: {str(a): b for a, b in v.items()}
                  for k, v in ETIQ.items()},
    'columnas': PRED, 'categoricas': CATEGORICAS, 'numericas': NUMERICAS}

# Distribuciones y tasas por categoría de cada variable
dist = {}
for c in PRED:
    vc = datos[c].value_counts().sort_index()
    if len(vc) <= 30:
        g = datos.groupby(c, observed=True).agg(
            n=('internet', 'size'), internet=('internet', 'mean'),
            ieh=('ieh', 'mean'), hacin=('hacin_critico', 'mean'),
            serv=('sin_servicios', 'mean'))
        dist[c] = {'valores': [float(v) for v in g.index],
                   'n': [int(v) for v in g['n']],
                   'internet': [round(float(v)*100, 2) for v in g['internet']],
                   'ieh': [round(float(v), 3) for v in g['ieh']],
                   'hacin': [round(float(v)*100, 2) for v in g['hacin']],
                   'serv': [round(float(v)*100, 2) for v in g['serv']]}
    else:
        h, b = np.histogram(datos[c], bins=30)
        dist[c] = {'hist': [int(v) for v in h],
                   'bins': [round(float(v), 3) for v in b]}
pan['distribuciones'] = dist

num = NUMERICAS + ['internet', 'ieh']
cm = datos[num].corr().round(4)
pan['correlacion'] = {'vars': num, 'matriz': cm.values.tolist()}
json.dump(pan, open(os.path.join(DIR, "panorama.json"), 'w',
                    encoding='utf-8'), ensure_ascii=False)
print(f"   panorama.json · {len(dist)} variables")

# Mapa departamental
mapa = datos.groupby('idep').agg(
    viviendas=('internet', 'size'), internet=('internet', 'mean'),
    ieh=('ieh', 'mean'), hacin_critico=('hacin_critico', 'mean'),
    sin_servicios=('sin_servicios', 'mean'),
    rural=('urbrur', lambda s: (s == 2).mean()),
    personas=('tot_pers', 'mean')).reset_index()
mapa['departamento'] = mapa['idep'].map(DEPTOS)
mapa['lat'] = mapa['idep'].map(lambda d: COORD[int(d)][0])
mapa['lon'] = mapa['idep'].map(lambda d: COORD[int(d)][1])
for c in ['internet', 'hacin_critico', 'sin_servicios', 'rural']:
    mapa[c] = (mapa[c] * 100).round(2)
mapa['sin_internet'] = (100 - mapa['internet']).round(2)
mapa['ieh'] = mapa['ieh'].round(3)
mapa['personas'] = mapa['personas'].round(2)
mapa.to_csv(os.path.join(DIR, "mapa_departamentos.csv"), index=False)
print(f"   mapa_departamentos.csv · {len(mapa)} departamentos")
print(mapa[['departamento', 'viviendas', 'internet', 'ieh']].to_string(index=False))


# --- Cruces para gráficos jerárquicos y comparativos -------------------------
cruces = {}
for v in ['urbrur', 'v06_piso', 'v09_energia', 'v15_servsan', 'v10_combus',
          'v11_basura', 'v03_pared', 'tip_hog']:
    if v not in datos.columns:
        continue
    g = datos.groupby(['idep', v], observed=True).agg(
        n=('internet', 'size'), internet=('internet', 'mean'),
        ieh=('ieh', 'mean')).reset_index()
    cruces[v] = {'idep': [int(x) for x in g['idep']],
                 'cat': [float(x) for x in g[v]],
                 'n': [int(x) for x in g['n']],
                 'internet': [round(float(x)*100, 2) for x in g['internet']],
                 'ieh': [round(float(x), 3) for x in g['ieh']]}
json.dump(cruces, open(os.path.join(DIR, "cruces.json"), 'w', encoding='utf-8'))
print(f"   cruces.json · {len(cruces)} variables")

# --- Nivel de equipamiento y trazabilidad de la limpieza (Fase B) -----------
ne = datos['nivel_equip'].value_counts()
pan['nivel_equip'] = {str(k): int(v) for k, v in ne.items()}
pan['limpieza'] = [
    {'etapa': 'Registros del censo', 'n': 4490488, 'quita': 0},
    {'etapa': 'Viviendas colectivas', 'n': 4463773, 'quita': 26715},
    {'etapa': 'Viviendas desocupadas', 'n': 4125627, 'quita': 338146},
    {'etapa': 'Sin habitantes', 'n': 3623711, 'quita': 501916},
    {'etapa': 'Sin variable objetivo', 'n': 3492019, 'quita': 131692}]
pan['validacion'] = {'propio': 76.27, 'oficial_ine': 76.30}
pan['cv'] = {'clasificacion': {'modelo': 'Random Forest', 'metrica': 'AUC',
                               'valor': 0.8098, 'desv': 0.0023,
                               'recall_min': 0.7273},
             'regresion': {'modelo': 'CatBoost', 'metrica': 'R2',
                           'valor': 0.5581, 'desv': 0.0083, 'rmse': 1.8788},
             'k': 5, 'n_modelos_proyecto': 58,
             'mcnemar_p': 0.0940, 't_pareada_p': 0.5285}
json.dump(pan, open(os.path.join(DIR, "panorama.json"), 'w',
                    encoding='utf-8'), ensure_ascii=False)


# --- Ranking real del proyecto (métricas que figuran en el informe) --------
# Se leen los consolidados de la Fase I. Son las cifras del informe y NO se
# recalculan aquí: la aplicación debe mostrar exactamente lo mismo que el
# documento. Las métricas que entrena esta fase son de las versiones del
# sistema y se presentan siempre junto a su referencia.
rank = {}
for clave, arch in [('clasificacion', 'I_todas_metricas_clasificacion_binaria.csv'),
                    ('regresion', 'I_todas_metricas_regresion.csv'),
                    ('multiclase', 'I_todas_metricas_multiclase.csv')]:
    ruta = os.path.join('entregables', arch)
    if os.path.exists(ruta):
        d = pd.read_csv(ruta)
        rank[clave] = d.to_dict('list')
        print(f"   ranking {clave}: {len(d)} modelos")
    else:
        print(f"   (falta {ruta}: la sección de ranking quedará vacía)")
json.dump(rank, open(os.path.join(DIR, "ranking.json"), 'w',
                     encoding='utf-8'), ensure_ascii=False)

# --- Modelos ganadores confirmados por validación cruzada (Fase H) ---------
GANADORES = {
    'clasificacion': {
        'modelo': 'Random Forest', 'metrica': 'AUC', 'valor': 0.8098,
        'desv': 0.0023, 'extra': 'recall de la clase sin internet 0,7273',
        'final': 'Reentrenado con 600.000 registros: AUC 0,8196 · recall 0,7500',
        'criterio': 'Mayor AUC medio en validación cruzada de 5 particiones. '
                    'Empate estadístico con CatBoost (t pareada p = 0,5285; '
                    'McNemar p = 0,0940) resuelto por parsimonia a favor del '
                    'modelo más simple y rápido.'},
    'regresion': {
        'modelo': 'CatBoost', 'metrica': 'R2', 'valor': 0.5581,
        'desv': 0.0083, 'extra': 'RMSE 1,8788 bienes de 13',
        'final': 'Reentrenado con 600.000 registros: R² 0,5692 · RMSE 1,8610',
        'criterio': 'Mayor R² medio en validación cruzada, con ventaja '
                    'consistente en las cinco particiones.'},
    'multiclase': {
        'modelo': 'CatBoost', 'metrica': 'F1 macro', 'valor': 0.6409,
        'desv': None, 'extra': 'sobre la muestra común de 120.000 registros',
        'final': 'No reentrenado: objetivo secundario del proyecto',
        'criterio': 'Mayor F1 macro entre los modelos multiclase evaluados '
                    'sobre la misma partición de prueba.'}}
pan['ganadores'] = GANADORES
json.dump(pan, open(os.path.join(DIR, "panorama.json"), 'w',
                    encoding='utf-8'), ensure_ascii=False)

# Muestra para la vista previa
datos.sample(3000, random_state=SEMILLA).to_csv(
    os.path.join(DIR, "muestra.csv"), index=False)
gc.collect()


# %%
# =============================================================================
#  J2 — ENTRENAMIENTO DE LAS VERSIONES
#  Dos versiones del sistema, cada una con varios algoritmos. Cambiar de
#  versión en la aplicación cambia todas las métricas y los gráficos.
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)

print("\n── J2 · Entrenamiento de versiones ──")
prep = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first',
                          sparse_output=False, dtype=np.float32), CATEGORICAS),
    ('num', StandardScaler(), NUMERICAS)])


def modelos_clf():
    m = {'Regresión Logística': Pipeline([('p', prep), ('m', LogisticRegression(
        max_iter=400, class_weight='balanced', random_state=SEMILLA))]),
        'Random Forest': RandomForestClassifier(
            n_estimators=80, max_depth=16, min_samples_leaf=40,
            max_features='sqrt', class_weight='balanced', n_jobs=-1,
            random_state=SEMILLA)}
    if XGB:
        from xgboost import XGBClassifier
        m['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6,
                                     learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, tree_method='hist',
                                     n_jobs=-1, eval_metric='auc',
                                     random_state=SEMILLA)
    if LGB:
        from lightgbm import LGBMClassifier
        m['LightGBM'] = LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                       num_leaves=63, min_child_samples=40,
                                       class_weight='balanced', n_jobs=-1,
                                       verbose=-1, random_state=SEMILLA)
    if CAT:
        from catboost import CatBoostClassifier
        m['CatBoost'] = CatBoostClassifier(iterations=150, depth=6,
                                           learning_rate=0.1, verbose=0,
                                           auto_class_weights='Balanced',
                                           thread_count=-1,
                                           random_seed=SEMILLA)
    return m


def modelos_reg():
    m = {'Ridge': Pipeline([('p', prep), ('m', Ridge(alpha=10.0))]),
         'Random Forest': RandomForestRegressor(
             n_estimators=80, max_depth=16, min_samples_leaf=40, n_jobs=-1,
             random_state=SEMILLA)}
    if XGB:
        from xgboost import XGBRegressor
        m['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=6,
                                    learning_rate=0.1, subsample=0.8,
                                    colsample_bytree=0.8, tree_method='hist',
                                    n_jobs=-1, random_state=SEMILLA)
    if LGB:
        from lightgbm import LGBMRegressor
        m['LightGBM'] = LGBMRegressor(n_estimators=200, learning_rate=0.1,
                                      num_leaves=63, min_child_samples=40,
                                      objective='poisson', n_jobs=-1,
                                      verbose=-1, random_state=SEMILLA)
    if CAT:
        from catboost import CatBoostRegressor
        m['CatBoost'] = CatBoostRegressor(iterations=150, depth=6,
                                          learning_rate=0.1, verbose=0,
                                          loss_function='Poisson',
                                          thread_count=-1,
                                          random_seed=SEMILLA)
    return m


def importancias(mod):
    try:
        if hasattr(mod, 'feature_importances_'):
            v = np.asarray(mod.feature_importances_, dtype=float)
            if len(v) == len(PRED):
                return dict(zip(PRED, np.round(v / max(v.sum(), 1e-9), 5)))
        if hasattr(mod, 'named_steps'):
            m2 = mod.named_steps['m']
            nom = mod.named_steps['p'].get_feature_names_out()
            co = np.abs(m2.coef_[0] if m2.coef_.ndim > 1 else m2.coef_)
            s = pd.Series(co, index=[n.split('__', 1)[-1] for n in nom])
            agg = {}
            for c in PRED:
                agg[c] = float(s[[i for i in s.index
                                  if i == c or i.startswith(c + '_')]].sum())
            t = sum(agg.values()) or 1
            return {k: round(v / t, 5) for k, v in agg.items()}
    except Exception:
        pass
    return {}


REG = {'generado': datetime.now().strftime('%Y-%m-%d %H:%M'),
       'semilla': SEMILLA, 'versiones': {}}
CURVAS, MATRICES, IMPORT, DISP = {}, {}, {}, {}

for ver, n in VERSIONES.items():
    n = min(n, N_TOTAL)
    if n < N_TOTAL:
        m_, _ = train_test_split(datos, train_size=n,
                                 stratify=datos['internet'],
                                 random_state=SEMILLA)
    else:
        m_ = datos.copy()
    m_ = m_.reset_index(drop=True)
    X, yb, yr = m_[PRED], m_['internet'].astype(int), m_['ieh'].astype(float)
    itr, ite = train_test_split(np.arange(n), test_size=.25, stratify=yb,
                                random_state=SEMILLA)
    Xtr, Xte = X.iloc[itr], X.iloc[ite]
    REG['versiones'][ver] = {'n_total': int(n), 'n_entrenamiento': len(itr),
                             'n_prueba': len(ite), 'clasificacion': {},
                             'regresion': {}}
    print(f"\n  {ver} · {n:,} registros")

    for nom, mod in modelos_clf().items():
        t0 = time.time()
        mod.fit(Xtr, yb.iloc[itr])
        prob = mod.predict_proba(Xte)[:, 1]
        pred = mod.predict(Xte)
        yt = yb.iloc[ite]
        seg = time.time() - t0
        met = {'Accuracy': accuracy_score(yt, pred),
               'Precision': precision_score(yt, pred, zero_division=0),
               'Recall': recall_score(yt, pred),
               'Recall_min': recall_score(yt, pred, pos_label=0),
               'F1': f1_score(yt, pred), 'ROC_AUC': roc_auc_score(yt, prob),
               'PR_AUC': average_precision_score(yt, prob)}
        REG['versiones'][ver]['clasificacion'][nom] = {
            'metricas': {k: round(float(v), 5) for k, v in met.items()},
            'segundos': round(seg, 2)}
        fpr, tpr, _ = roc_curve(yt, prob)
        pr, rc, _ = precision_recall_curve(yt, prob)
        p1 = max(1, len(fpr)//250); p2 = max(1, len(pr)//250)
        CURVAS[f"{ver}|clf|{nom}"] = {
            'fpr': np.round(fpr[::p1], 4).tolist(),
            'tpr': np.round(tpr[::p1], 4).tolist(),
            'prec': np.round(pr[::p2], 4).tolist(),
            'rec': np.round(rc[::p2], 4).tolist()}
        MATRICES[f"{ver}|clf|{nom}"] = confusion_matrix(yt, pred).tolist()
        base = mod.named_steps['m'] if hasattr(mod, 'named_steps') else mod
        IMPORT[f"{ver}|clf|{nom}"] = importancias(mod if hasattr(
            mod, 'named_steps') else base)
        joblib.dump(mod, os.path.join(DIR, "modelos",
                                      f"{ver}_clf_{nom}.joblib".replace(
                                          " ", "_")), compress=3)
        print(f"     clf {nom:<22} AUC {met['ROC_AUC']:.4f} · {seg:5.1f}s")

    for nom, mod in modelos_reg().items():
        t0 = time.time()
        mod.fit(Xtr, yr.iloc[itr])
        p = mod.predict(Xte)
        yt = yr.iloc[ite].values
        nz = yt != 0
        seg = time.time() - t0
        met = {'MAE': mean_absolute_error(yt, p),
               'MSE': mean_squared_error(yt, p),
               'RMSE': float(np.sqrt(mean_squared_error(yt, p))),
               'R2': r2_score(yt, p),
               'MAPE': float(np.mean(np.abs((yt[nz]-p[nz])/yt[nz]))*100)}
        REG['versiones'][ver]['regresion'][nom] = {
            'metricas': {k: round(float(v), 5) for k, v in met.items()},
            'segundos': round(seg, 2)}
        s = np.random.default_rng(SEMILLA).choice(len(yt),
                                                  min(2500, len(yt)),
                                                  replace=False)
        DISP[f"{ver}|reg|{nom}"] = {
            'real': yt[s].round(2).tolist(),
            'pred': np.round(p[s], 3).tolist()}
        base = mod.named_steps['m'] if hasattr(mod, 'named_steps') else mod
        IMPORT[f"{ver}|reg|{nom}"] = importancias(mod if hasattr(
            mod, 'named_steps') else base)
        joblib.dump(mod, os.path.join(DIR, "modelos",
                                      f"{ver}_reg_{nom}.joblib".replace(
                                          " ", "_")), compress=3)
        print(f"     reg {nom:<22} R²  {met['R2']:.4f} · {seg:5.1f}s")
    del m_, X, Xtr, Xte
    gc.collect()


# %%
# =============================================================================
#  J3 — DERIVA GEOGRÁFICA
#  Entrena en el eje central y evalúa en la periferia: ¿el modelo generaliza
#  igual en todo el territorio?
# =============================================================================
print("\n── J3 · Deriva geográfica ──")
CENTRO = [2, 3, 7]        # La Paz, Cochabamba, Santa Cruz
PERIF = [1, 4, 5, 6, 8, 9]

sub = datos.sample(min(250_000, N_TOTAL), random_state=SEMILLA)
ent = sub[sub['idep'].isin(CENTRO)]
if LGB:
    from lightgbm import LGBMClassifier
    base_m = LGBMClassifier(n_estimators=200, learning_rate=0.1,
                            num_leaves=63, class_weight='balanced',
                            n_jobs=-1, verbose=-1, random_state=SEMILLA)
else:
    base_m = RandomForestClassifier(n_estimators=80, max_depth=16,
                                    min_samples_leaf=40,
                                    class_weight='balanced', n_jobs=-1,
                                    random_state=SEMILLA)
base_m.fit(ent[PRED], ent['internet'].astype(int))

filas = []
for d in sorted(DEPTOS):
    g = sub[sub['idep'] == d]
    if len(g) < 500:
        continue
    pr = base_m.predict_proba(g[PRED])[:, 1]
    yy = g['internet'].astype(int)
    if yy.nunique() < 2:
        continue
    filas.append({'idep': int(d), 'departamento': DEPTOS[d],
                  'grupo': 'Entrenamiento' if d in CENTRO else 'No visto',
                  'n': int(len(g)),
                  'auc': round(float(roc_auc_score(yy, pr)), 4),
                  'pct_internet': round(float(yy.mean()*100), 2)})
deriva = pd.DataFrame(filas)
deriva.to_csv(os.path.join(DIR, "deriva.csv"), index=False)
a1 = deriva[deriva.grupo == 'Entrenamiento']['auc'].mean()
a2 = deriva[deriva.grupo == 'No visto']['auc'].mean()
print(deriva.to_string(index=False))
print(f"   AUC medio entrenamiento {a1:.4f} · no visto {a2:.4f} · "
      f"caída {a1-a2:+.4f}")
REG['deriva'] = {'auc_entrenamiento': round(float(a1), 4),
                 'auc_no_visto': round(float(a2), 4),
                 'caida': round(float(a1-a2), 4),
                 'departamentos_entrenamiento': [DEPTOS[d] for d in CENTRO]}
del sub, ent
gc.collect()


# %%
# =============================================================================
#  J4 — OBJETIVOS ADICIONALES
#  Dos variables binarias que el dataset ya contiene y no se habían modelado.
# =============================================================================
print("\n── J4 · Objetivos adicionales ──")
# Cada objetivo se deriva de otras columnas: hay que excluir sus componentes
# o el modelo alcanzaría AUC 1,0 por fuga de información, no por aprender.
EXTRA = {'hacin_critico': 'Hacinamiento crítico (>3 personas por dormitorio)',
         'sin_servicios': 'Carencia de agua, energía o saneamiento'}
FUGA_EXTRA = {
    'hacin_critico': ['hacin_critico', 'hacinamiento', 'pers_por_habitac',
                      'tot_pers', 'v14_dormit', 'v13_habitac'],
    'sin_servicios': ['sin_servicios', 'v07_aguapro', 'v09_energia',
                      'v15_servsan', 'v16_desague', 'v08_aguadist']}
m_ = datos.sample(min(120_000, N_TOTAL), random_state=SEMILLA) \
    .reset_index(drop=True)
REG['extra'] = {}
for obj, desc in EXTRA.items():
    cols = [c for c in PRED if c not in FUGA_EXTRA[obj]]
    Xe, ye = m_[cols], m_[obj].astype(int)
    itr, ite = train_test_split(np.arange(len(m_)), test_size=.25,
                                stratify=ye, random_state=SEMILLA)
    mo = RandomForestClassifier(n_estimators=80, max_depth=16,
                                min_samples_leaf=40, class_weight='balanced',
                                n_jobs=-1, random_state=SEMILLA)
    t0 = time.time()
    mo.fit(Xe.iloc[itr], ye.iloc[itr])
    prob = mo.predict_proba(Xe.iloc[ite])[:, 1]
    pred = mo.predict(Xe.iloc[ite])
    yt = ye.iloc[ite]
    REG['extra'][obj] = {
        'descripcion': desc, 'prevalencia': round(float(ye.mean()*100), 2),
        'metricas': {'Accuracy': round(float(accuracy_score(yt, pred)), 5),
                     'Recall_min': round(float(recall_score(yt, pred,
                                                            pos_label=0)), 5),
                     'F1': round(float(f1_score(yt, pred)), 5),
                     'ROC_AUC': round(float(roc_auc_score(yt, prob)), 5)},
        'segundos': round(time.time()-t0, 2),
        'excluidas': FUGA_EXTRA[obj], 'n_predictoras': len(cols),
        'importancias': {k: float(v) for k, v in sorted(
            dict(zip(cols, np.round(mo.feature_importances_, 5))).items(),
            key=lambda x: -x[1])[:12]}}
    fpr, tpr, _ = roc_curve(yt, prob)
    p1 = max(1, len(fpr)//250)
    CURVAS[f"extra|{obj}"] = {'fpr': np.round(fpr[::p1], 4).tolist(),
                              'tpr': np.round(tpr[::p1], 4).tolist()}
    MATRICES[f"extra|{obj}"] = confusion_matrix(yt, pred).tolist()
    joblib.dump(mo, os.path.join(DIR, "modelos", f"extra_{obj}.joblib"),
                compress=3)
    print(f"   {obj:<16} AUC {REG['extra'][obj]['metricas']['ROC_AUC']:.4f} "
          f"· prevalencia {REG['extra'][obj]['prevalencia']:.2f} % · "
          f"{len(cols)} predictoras (excluidas "
          f"{len(FUGA_EXTRA[obj])} por fuga)")


# %%
# =============================================================================
#  J5 — GUARDADO Y REGISTRO EN MLFLOW (si está disponible)
# =============================================================================
print("\n── J5 · Guardado ──")
REG['valores_tipicos'] = {
    **{c: float(datos[c].mode().iloc[0]) for c in CATEGORICAS},
    **{c: float(datos[c].median()) for c in NUMERICAS}}
json.dump(REG, open(os.path.join(DIR, "registro.json"), 'w',
                    encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(CURVAS, open(os.path.join(DIR, "curvas.json"), 'w', encoding='utf-8'))
json.dump(MATRICES, open(os.path.join(DIR, "matrices.json"), 'w', encoding='utf-8'))
json.dump(IMPORT, open(os.path.join(DIR, "importancias.json"), 'w', encoding='utf-8'))
json.dump(DISP, open(os.path.join(DIR, "dispersion.json"), 'w', encoding='utf-8'))
print("   registro.json · curvas.json · matrices.json · importancias.json")

try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("CPV2024_Brecha_Digital")
    for ver, d in REG['versiones'].items():
        for tipo in ['clasificacion', 'regresion']:
            for nom, info in d[tipo].items():
                with mlflow.start_run(run_name=f"{ver}_{tipo}_{nom}"):
                    mlflow.log_params({'version': ver, 'modelo': nom,
                                       'tipo': tipo,
                                       'n_entrenamiento': d['n_entrenamiento'],
                                       'semilla': SEMILLA})
                    mlflow.log_metrics(info['metricas'])
                    mlflow.log_metric('segundos', info['segundos'])
    print("   ✔ Registrado en MLflow (sqlite:///mlflow.db)")
except Exception as e:
    print(f"   (MLflow omitido: {type(e).__name__})")

print(f"\n  ✔ LISTO. Ahora ejecuta en la terminal:")
print(f"     streamlit run app_mlops.py")
gc.collect()
