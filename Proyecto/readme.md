# Documentación del proyecto

## Proyecto MLOps Pipeline

Este proyecto implementa un pipeline completo de Machine Learning Operations (MLOps) que incluye:

### Estructura del Proyecto

```
Proyecto/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb          # Carga de dataset (CSV de ejemplo no productivo)
│       ├── comprension_eda.ipynb       # Análisis exploratorio de datos (EDA)
│       ├── ft_engineering.py           # Generación de features y creación de datasets
│       ├── heuristic_model.py          # Modelo heurístico base
│       ├── model_training.ipynb        # Entrenamiento y comparación de modelos
│       ├── model_deploy.ipynb          # Despliegue del modelo en un endpoint
│       ├── model_evaluation.ipynb      # Evaluación de métricas del modelo desplegado
│       └── model_monitoring.ipynb      # Monitoreo y detección de datadrift
├── config.json                         # Archivo de configuración del pipeline
├── Base_de_datos.csv                   # Dataset de ejemplo
├── requirements.txt                    # Librerías y dependencias
├── .gitignore                          # Exclusiones de Git
├── readme.md                           # Documentación del proyecto
└── set_up.bat                          # Script para preparar el entorno
```

### Instalación

1. Ejecutar el script de configuración:
```bash
set_up.bat
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

### Uso

El pipeline está diseñado para ejecutarse secuencialmente a través de los notebooks en la carpeta `src/`.

### Configuración

Modifica el archivo `config.json` para ajustar los parámetros del pipeline según tus necesidades.