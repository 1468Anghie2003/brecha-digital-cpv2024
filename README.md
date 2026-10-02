# Brecha digital y equipamiento del hogar en Bolivia

Sistema predictivo de exclusión digital y material construido sobre la tabla
de vivienda del **Censo de Población y Vivienda 2024** del Instituto Nacional
de Estadística de Bolivia.

**Universidad Mayor de San Andrés** · Facultad de Ciencias Puras y Naturales
Carrera de Informática · Machine Learning · Aprendizaje supervisado

Ethan Alba Orellana · Jhamil López Salgueiro · Angela Luz Mamani Castro

---

## Resumen

A partir de 4.490.488 registros censales se construyó un universo depurado de
**3.492.019 viviendas particulares ocupadas**, cuya cobertura de internet
—76,27 %— reproduce la cifra oficial del INE con una diferencia de **0,03
puntos porcentuales**.

Sobre ese conjunto se entrenaron y compararon modelos de todas las familias
del aprendizaje supervisado para tres variables objetivo.

| Tarea | Modelo seleccionado | Desempeño verificado |
|---|---|---|
| Clasificación · acceso a internet | Random Forest | AUC 0,8098 ± 0,0023 · recall de la clase minoritaria 0,7273 |
| Regresión · equipamiento del hogar | CatBoost | R² 0,5581 ± 0,0083 · RMSE 1,8788 bienes de 13 |
| Clasificación multiclase · nivel de equipamiento | CatBoost | F1 macro 0,6409 |

La selección se realizó con validación cruzada estratificada de cinco
particiones y pruebas de significancia estadística (t pareada y McNemar).

---

## Ejecutar la aplicación web

No es necesario reproducir el procesamiento: los artefactos ya calculados
viajan con el repositorio.

```bash
pip install -r requirements.txt
streamlit run app_mlops.py
```

La aplicación se abre en `http://localhost:8501`.

En Windows también puede ejecutarse con doble clic sobre `ejecutar_app.bat`.

Para comprobar que el entorno está completo antes de lanzarla:

```bash
python verificar_mlops.py
```

---

## Contenido del repositorio

| Elemento | Descripción |
|---|---|
| `Vivienda2.py` | Código completo del proyecto, fases A a I |
| `Vivienda.py` | Versión preliminar previa al rediseño en fases |
| `FASE_J_preparar_mlops.py` | Genera los artefactos del sistema MLOps |
| `app_mlops.py` | Aplicación web con once secciones |
| `verificar_mlops.py` | Diagnóstico del entorno y de los artefactos |
| `mlops/` | Modelos serializados, métricas, curvas y datos agregados |
| `entregables/` | Modelos finales, metadatos y ranking completo de modelos |
| `figuras_v2/` | Las 48 figuras del informe |
| `BITACORA_PROYECTO.xlsx` | Registro de decisiones metodológicas y de modelos |
| `FASES_RESULTADOS_SALIDAS_PYTHON_VIVIENDA.txt` | Salida de consola completa de todas las fases |

Los conjuntos de datos no se publican por superar el límite de tamaño del
repositorio. El archivo original es de acceso público en el portal del INE.

---

## Secciones de la aplicación

**Inicio** · modelos ganadores e indicadores generales
**Panorama del dataset** · explorador de las 28 variables predictoras
**Explorador territorial** · cruces entre departamento y características
**Mapa de Bolivia** · seis indicadores en tres vistas
**Clasificación** y **Regresión** · comparación entre algoritmos
**Ranking del proyecto** · métricas publicadas en el informe
**Predicción en vivo** · estimación de riesgo para una vivienda
**Objetivos adicionales** · hacinamiento y carencia de servicios
**MLOps y versionado** · registro de versiones y deriva territorial

---

## Versionado y seguimiento

El sistema registra dos versiones que difieren en el volumen de entrenamiento
(80.000 y 200.000 registros) y mide la **deriva geográfica**: entrenado solo
con La Paz, Cochabamba y Santa Cruz, su AUC cae de 0,8428 a 0,7892 al
evaluarse sobre los seis departamentos restantes.

El registro de experimentos también está disponible en MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Nota sobre el uso de los modelos

Los clasificadores se entrenaron con ponderación de clases balanceada, lo que
desplaza sus probabilidades hacia abajo. **Son adecuados para ordenar hogares
por nivel de riesgo, no para estimar proporciones poblacionales.** El mapa
territorial utiliza por ello los valores observados en el censo.

---

## Fuente de los datos

Instituto Nacional de Estadística de Bolivia (2024). *Censo de Población y
Vivienda 2024*. https://www.ine.gob.bo
