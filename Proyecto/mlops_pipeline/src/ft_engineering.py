"""Pipeline de ingeniería de características y preparación de datasets.

Este módulo centraliza la lógica de carga, limpieza y transformación de los datos
para el proyecto de predicción de deserción estudiantil. Está alineado con las
pautas del proyecto final de Machine Learning y con la estructura MLOps definida
para la organización.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class DatasetArtifacts:
    """Conjunto de artefactos generados por la fase de feature engineering."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_pipeline: Pipeline
    feature_names: List[str]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"
OUTPUT_DIR = PROJECT_ROOT / "mlops_pipeline" / "outputs"
MODELS_DIR = PROJECT_ROOT / "mlops_pipeline" / "models"


def load_project_config(config_path: Path = CONFIG_PATH) -> Dict:
    """Carga la configuración general del proyecto."""
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de las columnas eliminando caracteres problemáticos."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\t", "", regex=False)
    )
    return df


def infer_variable_roles(
    df: pd.DataFrame,
    target_col: str = "Target",
    numeric_threshold: int = 20,
    exclude_cols: List[str] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Clasifica las variables en numéricas, categóricas nominales y categóricas ordinales."""

    numeric_cols: List[str] = []
    categorical_nominal: List[str] = []
    categorical_ordinal: List[str] = []

    exclude = set(exclude_cols or [])

    for col in df.columns:
        if col == target_col or col in exclude:
            continue

        series = df[col]
        unique_count = series.nunique(dropna=True)

        if is_numeric_dtype(series):
            if unique_count > numeric_threshold:
                numeric_cols.append(col)
            else:
                # Trata variables numéricas discretas como categóricas
                if unique_count <= 2:
                    categorical_ordinal.append(col)
                else:
                    categorical_nominal.append(col)
        else:
            if unique_count <= 2:
                categorical_ordinal.append(col)
            else:
                categorical_nominal.append(col)

    return numeric_cols, categorical_nominal, categorical_ordinal


def build_feature_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    ordinal_features: List[str],
) -> ColumnTransformer:
    """Crea un `ColumnTransformer` alineado con el pipeline solicitado (imputación + encoding)."""

    transformers: List[Tuple[str, Pipeline, List[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
                ),
            ]
        )
        transformers.append(("categoric", categorical_pipeline, categorical_features))

    if ordinal_features:
        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        transformers.append(("categoric_ord", ordinal_pipeline, ordinal_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def generate_datasets(test_size: float = 0.2, config: Dict | None = None) -> DatasetArtifacts:
    """
    Ejecuta la secuencia completa de feature engineering y devuelve los artefactos necesarios
    para entrenamiento y evaluación.
    """
    if config is None:
        config = load_project_config()

    data_path = PROJECT_ROOT / config["pipeline_config"]["data_path"]
    df_raw = pd.read_csv(data_path, sep=";", encoding="utf-8")
    df_raw = clean_column_names(df_raw)

    # Derivamos columna numérica del target para análisis posteriores.
    target_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    if "Target" not in df_raw.columns:
        raise KeyError("La columna 'Target' no existe en el dataset. Revisa la fuente de datos.")
    df_raw["Target_id"] = df_raw["Target"].map(target_mapping)

    numeric_cols, categorical_cols, ordinal_cols = infer_variable_roles(
        df_raw,
        target_col="Target",
        exclude_cols=["Target_id"],
    )

    # Separación de etiquetas
    y = df_raw["Target"].copy()
    X = df_raw.drop(columns=["Target", "Target_id"])

    stratify_col = y if y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config["model_config"]["random_state"],
        stratify=stratify_col,
    )

    feature_pipeline = build_feature_pipeline(numeric_cols, categorical_cols, ordinal_cols)
    feature_pipeline.fit(x_train)

    x_train_transformed = feature_pipeline.transform(x_train)
    x_test_transformed = feature_pipeline.transform(x_test)

    feature_names = feature_pipeline.get_feature_names_out().tolist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Persistimos pipeline para reutilizarlo en entrenamiento, despliegue y monitoreo.
    joblib.dump(feature_pipeline, OUTPUT_DIR / "feature_pipeline.joblib")
    joblib.dump({"feature_names": feature_names}, OUTPUT_DIR / "feature_metadata.joblib")

    return DatasetArtifacts(
        x_train=x_train_transformed,
        x_test=x_test_transformed,
        y_train=y_train,
        y_test=y_test,
        feature_pipeline=feature_pipeline,
        feature_names=feature_names,
    )


def save_numpy_datasets(artifacts: DatasetArtifacts) -> None:
    """Guarda los conjuntos de entrenamiento y prueba transformados en disco."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "x_train.npy", artifacts.x_train)
    np.save(OUTPUT_DIR / "x_test.npy", artifacts.x_test)
    joblib.dump(artifacts.y_train, OUTPUT_DIR / "y_train.joblib")
    joblib.dump(artifacts.y_test, OUTPUT_DIR / "y_test.joblib")


def run_feature_engineering() -> DatasetArtifacts:
    """Función orquestadora para ser utilizada desde notebooks o scripts externos."""
    config = load_project_config()
    train_size = 1.0 - config["model_config"]["test_size"]
    artifacts = generate_datasets(test_size=1 - train_size, config=config)
    save_numpy_datasets(artifacts)
    return artifacts


if __name__ == "__main__":
    artifacts = run_feature_engineering()
    print(
        "Feature engineering completado.",
        f"Dimensión X_train: {artifacts.x_train.shape}",
        f"Dimensión X_test: {artifacts.x_test.shape}",
        sep="\n",
    )