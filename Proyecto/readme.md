# Proyecto Final de Machine Learning – Pipeline MLOps

Realizado por **Jesús Andrés Vargas Zerpa** <br>
Curso: **Machine Learning** <br>
Profesor: **Juan Sebastian Parra** <br>

**Universidad Católica Luis Amigó**

## 1. Descripción general
Este repositorio implementa un flujo completo de MLOps para predecir el estado académico de estudiantes universitarios (`Dropout`, `Enrolled`, `Graduate`). El dataset proviene de Kaggle (“Predict Students' Dropout and Academic Success”) y contiene 4.424 registros y 37 atributos que describen información demográfica, socioeconómica y académica.

El objetivo del proyecto es:
- Construir un pipeline reproducible (ingesta → EDA → feature engineering → entrenamiento → evaluación → despliegue → monitoreo).
- Documentar hallazgos con métricas concretas y lineamientos accionables.
- Disponibilizar modelo y pipeline como servicio API y preparar artefactos para monitoreo y análisis continuo.

## 2. Estructura principal del repositorio
```
Proyecto/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb          # Verificación de orígenes y calidad básica de la data
│       ├── comprension_eda.ipynb       # Análisis exploratorio completo (EDA)
│       ├── ft_engineering.py           # Generación de features y particiones (entrenamiento/prueba)
│       ├── model_training.ipynb        # Entrenamiento interactivo y visualizaciones
│       ├── model_training_evaluation.py# Script orquestador de entrenamiento y métricas
│       ├── model_evaluation.ipynb      # Reporte consolidado para auditoría y despliegue
│       ├── model_deploy.ipynb          # Preparación de API FastAPI y artefactos Docker
│       └── model_monitoring.ipynb      # Métricas de drift y visualizaciones de monitoreo
├── Base_de_datos.csv                   # Dataset fuente (no productivo)
├── config.json                         # Parámetros globales del pipeline
├── requirements.txt                    # Dependencias (Python 3.13)
├── Dockerfile / .dockerignore          # Imagen para servir el modelo vía FastAPI
├── readme.md                           # Documento guía (este archivo)
└── set_up.bat                          # Script opcional para crear el entorno virtual
```

## 3. Preparación del entorno
```bash
# Clonar el repo y ubicarse en la raíz
cd "Proyecto Final ML/Proyecto"

# Crear/activar el entorno virtual (opcional: usar set_up.bat)
python -m venv -venv
.\-venv\Scripts\Activate.ps1      # PowerShell

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Flujo de ejecución recomendado
1. **Feature engineering** – `python mlops_pipeline/src/ft_engineering.py`
   - Limpia nombres de columnas, clasifica variables (numéricas, dicotómicas, ordinales), aplica imputación (`SimpleImputer`) y genera el `ColumnTransformer` con escalado, one-hot y codificación ordinal.
   - Guarda artefactos en `mlops_pipeline/outputs/` (`feature_pipeline.joblib`, `feature_metadata.joblib`, `x_train.npy`, `x_test.npy`, etc.).
2. **Entrenamiento masivo** – `python mlops_pipeline/src/model_training_evaluation.py`
   - Entrena Logistic Regression, RandomForest y Gradient Boosting con validación cruzada estratificada (5 folds).
   - Calcula métricas (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, tiempos) y selecciona el mejor modelo.
   - Persiste el modelo campeón (`mlops_pipeline/models/best_model_*.joblib`) y un reporte de métricas (`model_cv_summary.csv`, `classification_report_test.csv`).
3. **Notebooks interactivos** (opcional pero recomendado para auditoría):
   - `Cargar_datos.ipynb` → validar fuentes.
   - `comprension_eda.ipynb` → revisar hallazgos exploratorios.
   - `model_training.ipynb` y `model_evaluation.ipynb` → visualizar matrices de confusión, curvas ROC y resúmenes.
   - `model_deploy.ipynb` → generar `app/main.py` y archivos Docker.
   - `model_monitoring.ipynb` → ejecutar métricas de drift y generar `drift_report.parquet`.
4. **Despliegue local (opcional)**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   # Luego visitar http://127.0.0.1:8000/docs o consumir /health y /predict
   ```
5. **Monitoreo** – ejecutar `model_monitoring.ipynb` periódicamente (o programarlo) para recalcular KS, PSI, Jensen-Shannon y alertas categóricas.

## 5. Exploratory Data Analysis (EDA)
Principales hallazgos (ver `comprension_eda.ipynb`):
- **Distribución de Target:** Graduate 49.93 %, Dropout 32.12 %, Enrolled 17.95 % (4.424 registros en total).
- **Tipificación:**
  - 19 variables numéricas continuas (aprobaciones, créditos, indicadores macroeconómicos, edad).
  - 7 variables dicotómicas (`Gender`, `Scholarship holder`, `Debtor`, etc.).
  - 12 variables politómicas/ordinales (modo de aplicación, curso, asistencia diurna/nocturna, antecedentes familiares).
- **Edad y rendimiento:** edad promedio 23.27 años (Dropout 26.07 vs Graduate 21.78). Promedio de asignaturas aprobadas: 1.º semestre 4.71, 2.º semestre 4.44.
- **Finanzas y retención:**
  - Becarios: 76 % se gradúa; no becarios: 41.3 % gradúa y 38.7 % abandona.
  - Estudiantes al día en matrículas (`Tuition fees up to date`) y sin deudas muestran mayor retención.
- **Jornada:** modalidad nocturna tiene 42.9 % de deserción y 41.6 % de graduación; diurna sube a 51 % de graduación.
- **Correlaciones:** `Target_id` correlaciona fuertemente con aprobaciones y notas (0.59–0.65); factores financieros aportan correlaciones moderadas (0.30). Variables macroeconómicas y antecedentes familiares carecen de señal ⇒ se descartan en la etapa de features.
- **Outliers:** <5 % en la mayoría de indicadores académicos; se gestionan con escalado robusto o winsorización en futuros retrainings.

## 6. Ingeniería de características (`ft_engineering.py`)
- **Limpieza:** normalización de nombres, homologación de nulos, clasificación de variables (numéricas vs. categóricas dicotómicas/ordinales).
- **Transformaciones:**
  - Numéricas → `SimpleImputer(strategy='median')` + `StandardScaler`.
  - Categóricas nominales → `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Categóricas ordinales/dicotómicas → `SimpleImputer` + `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.
- **Partición estratificada:** 80 % entrenamiento / 20 % prueba (`random_state=42`).
- **Persistencia:** pipeline (joblib), metadata de features y datasets transformados (`.npy` / `.joblib`).

## 7. Entrenamiento y evaluación (`model_training_evaluation.py`)
### Resultados en validación cruzada (5 folds)
| Modelo               | Accuracy | Precision_macro | Recall_macro | F1_macro | fit_time (s) |
|----------------------|---------:|----------------:|-------------:|---------:|-------------:|
| Logistic Regression  | 0.7429   | 0.7061          | 0.7112       | 0.7030   | 0.55         |
| Gradient Boosting    | 0.7674   | 0.7186          | 0.6796       | 0.6901   | 4.30         |
| Random Forest        | 0.7748   | 0.7345          | 0.6772       | 0.6889   | 1.53         |

### Desempeño en conjunto de prueba (885 registros)
| Clase     | Precision | Recall | F1-score | Support |
|-----------|----------:|-------:|---------:|--------:|
| Dropout   | 0.82      | 0.68   | 0.74     | 284     |
| Enrolled  | 0.41      | 0.65   | 0.50     | 159     |
| Graduate  | 0.87      | 0.79   | 0.83     | 442     |
| **Macro avg** | **0.70** | **0.71** | **0.69** | **885** |
| **Accuracy**  | **0.73** |         |         |         |

- **Modelo seleccionado:** `logistic_regression` balanceada (mejor `f1_macro` ≈ 0.70 con complejidad baja y tiempos reducidos).
- **Riesgo principal:** clase `Enrolled` con precision moderada (0.41) ⇒ monitoreo específico y considerar oversampling/umbral adaptativo.
- **Artefactos guardados:** `best_model_logistic_regression.joblib`, `model_metadata.json`, `model_cv_summary.csv`, `classification_report_test.csv`, matriz de confusión (`outputs/`).

## 8. Despliegue del modelo (`model_deploy.ipynb`)
El notebook realiza:
1. Carga de artefactos (`feature_pipeline` + `best_model`).
2. Generación de `app/main.py` con FastAPI:
   - Endpoint `GET /health` → estado del servicio.
   - Endpoint `POST /predict` → recibe `{"records": [ {...}, {...} ]}` y retorna `{"predictions": [...], "probabilities": [...]}`.
3. Creación de `Dockerfile` y `.dockerignore` para empaquetar la app.
4. Ejemplo para ejecutar la API localmente:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   # Health check
   curl http://127.0.0.1:8000/health
   # Predicción
   curl -X POST http://127.0.0.1:8000/predict \
        -H "Content-Type: application/json" \
        -d @payload.json
   ```
5. Construcción en Docker (opcional):
   ```bash
   docker build -t student-dropout-api .
   docker run -p 8000:8000 student-dropout-api
   ```

## 9. Monitoreo y drift (`model_monitoring.ipynb`)
- Simula un escenario **referencia vs. producción** (partición estratificada).
- Calcula métricas por variable:
  - Kolmogorov-Smirnov (numéricas), PSI (numéricas), Jensen-Shannon (numéricas), Chi-cuadrado (categóricas).
  - Genera alerta (`alerta == True`) si KS > 0.1, PSI > 0.25 o p-value < 0.05.
- Visualiza histogramas comparativos y tendencia del accuracy en “ventanas” temporales.
- Exporta `mlops_pipeline/outputs/drift_report.parquet` para integrarse en dashboards o notificaciones.

## 10. Principales insights del proyecto
- **Retención crítica:** 32 % de estudiantes abandonan; la tasa sube a 42.9 % en la jornada nocturna.
- **Soporte financiero:** 76 % de los becarios se gradúan vs. 41.3 % de quienes no tienen beca; la morosidad (`Debtor`) aumenta el riesgo de deserción.
- **Rendimiento temprano:** aprobar ≥4 materias por semestre y mantener notas promedio ≈ 10.6 reduce la probabilidad de abandono.
- **Modelo en producción:** la regresión logística ofrece un compromiso estable; se monitorea la clase `Enrolled` para mejorar sensibilidad.
- **Alertas y monitoreo:** se recomienda ejecutar el notebook de drift de forma periódica y levantar alertas (correo/Slack) cuando alguna variable muestre cambios significativos.

## 11. Pruebas y SonarCloud
- **Pruebas unitarias / cobertura:**
  ```bash
  pip install pytest pytest-cov
  pytest --cov=mlops_pipeline/src --cov-report=xml
  ```
- **CI/CD (opcional):** agregar un workflow de GitHub Actions o job de Jenkins que ejecute `pytest --cov` + `sonar-scanner` en cada push.

## 12. Reproducibilidad
- Artefactos principales: `feature_pipeline.joblib`, `feature_metadata.joblib`, `best_model_*.joblib`, `model_metadata.json`, `model_cv_summary.csv`, `classification_report_test.csv`, `drift_report.parquet`.
- La configuración (`config.json`) centraliza `data_path`, rutas de salida y parámetros de entrenamiento (semilla, tamaño de prueba).
- Cada notebook/script indica en su encabezado el propósito y las acciones principales; todos los hallazgos se basan en métricas calculadas dentro de este repositorio.

---
**Contacto / Notas finales:**
- Para dudas sobre la integración o recomendaciones de mejora, iniciar la discusión en los issues del repositorio.
- Mantener actualizados los artefactos después de cada retraining y documentar cualquier cambio relevante en este README.
