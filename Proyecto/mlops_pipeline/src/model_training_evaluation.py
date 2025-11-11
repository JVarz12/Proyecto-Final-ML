"""Entrenamiento y evaluación de modelos supervisados.

Este módulo entrena varios algoritmos para predecir la deserción estudiantil,
compara su desempeño mediante validación cruzada y selecciona el mejor modelo.
Genera reportes y gráficas alineadas con la rúbrica del proyecto.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import label_binarize

from ft_engineering import DatasetArtifacts, run_feature_engineering

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"
OUTPUTS_DIR = PROJECT_ROOT / "mlops_pipeline" / "outputs"
MODELS_DIR = PROJECT_ROOT / "mlops_pipeline" / "models"


@dataclass
class ModelResult:
    name: str
    estimator: ClassifierMixin
    metrics: Dict[str, float]
    fit_time: float
    score_time: float


def load_config() -> Dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def summarize_classification(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Devuelve un resumen tabular de métricas por clase y agregadas."""
    report = classification_report(y_true, y_pred, output_dict=True)
    return (
        pd.DataFrame(report)
        .T.rename_axis("clase")
        .reset_index()
    )


def build_model(model: ClassifierMixin) -> ClassifierMixin:
    """Permite centralizar futuros ajustes sobre los estimadores antes de entrenarlos."""
    return model


def configure_models(random_state: int) -> Dict[str, ClassifierMixin]:
    return {
        "logistic_regression": build_model(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
            )
        ),
        "random_forest": build_model(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": build_model(
            GradientBoostingClassifier(random_state=random_state)
        ),
    }


def evaluate_models(
    models: Dict[str, ClassifierMixin],
    artifacts: DatasetArtifacts,
    random_state: int,
) -> Tuple[pd.DataFrame, ModelResult]:
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results: List[ModelResult] = []

    for name, estimator in models.items():
        start = time.perf_counter()
        scores = cross_validate(
            estimator,
            artifacts.x_train,
            artifacts.y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        elapsed = time.perf_counter() - start

        metrics = {metric.replace("test_", ""): float(np.mean(values)) for metric, values in scores.items() if metric.startswith("test_")}
        results.append(
            ModelResult(
                name=name,
                estimator=estimator,
                metrics=metrics,
                fit_time=float(np.mean(scores["fit_time"])),
                score_time=float(np.mean(scores["score_time"])),
            )
        )
        print(f"Modelo {name} evaluado | f1_macro={metrics['f1_macro']:.3f} | tiempo total={elapsed:.2f}s")

    cv_summary = (
        pd.DataFrame(
            [
                {
                    "modelo": result.name,
                    **result.metrics,
                    "fit_time": result.fit_time,
                    "score_time": result.score_time,
                }
                for result in results
            ]
        )
        .sort_values(by="f1_macro", ascending=False)
        .reset_index(drop=True)
    )

    best_row = cv_summary.iloc[0]
    best_result = next(result for result in results if result.name == best_row["modelo"])
    return cv_summary, best_result


def plot_cv_results(cv_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_cols = [col for col in ["accuracy", "precision_macro", "recall_macro", "f1_macro"] if col in cv_summary.columns]
    cv_long = cv_summary.melt(id_vars="modelo", value_vars=metrics_cols, var_name="métrica", value_name="puntaje")
    sns.barplot(data=cv_long, x="modelo", y="puntaje", hue="métrica", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Comparación de métricas promedio (CV)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def evaluate_test_set(
    best_result: ModelResult,
    artifacts: DatasetArtifacts,
    random_state: int,
) -> Dict:
    estimator = best_result.estimator
    estimator.fit(artifacts.x_train, artifacts.y_train)

    y_pred = estimator.predict(artifacts.x_test)
    y_proba = estimator.predict_proba(artifacts.x_test)

    classes = estimator.classes_
    y_test_binarized = label_binarize(artifacts.y_test, classes=classes)
    roc_auc_macro = roc_auc_score(y_test_binarized, y_proba, average="macro")

    classification_summary = summarize_classification(artifacts.y_test, y_pred)

    cm = confusion_matrix(artifacts.y_test, y_pred, labels=list(classes))
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Matriz de confusión - {best_result.name}")
    plt.tight_layout()

    cm_path = OUTPUTS_DIR / f"confusion_matrix_{best_result.name}.png"
    fig_cm.savefig(cm_path, dpi=120)
    plt.close(fig_cm)

    classification_path = OUTPUTS_DIR / "classification_report_test.csv"
    classification_summary.to_csv(classification_path, index=False)

    return {
        "best_estimator": estimator,
        "classification_summary": classification_summary,
        "roc_auc_macro": float(roc_auc_macro),
        "confusion_matrix_path": cm_path,
    }


def persist_best_model(best_name: str, estimator: ClassifierMixin, cv_summary: pd.DataFrame, roc_auc_macro: float, artifacts: DatasetArtifacts) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"best_model_{best_name}.joblib"
    joblib.dump(estimator, model_path)

    metadata = {
        "model_name": best_name,
        "cv_metrics": cv_summary.set_index("modelo").loc[best_name].to_dict(),
        "metrics_test": {
            "accuracy": float((artifacts.y_test == estimator.predict(artifacts.x_test)).mean()),
            "roc_auc_macro": roc_auc_macro,
        },
        "feature_count": len(artifacts.feature_names),
        "feature_pipeline_path": str((OUTPUTS_DIR / "feature_pipeline.joblib").resolve()),
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    cv_summary.to_csv(OUTPUTS_DIR / "model_cv_summary.csv", index=False)
    return model_path


def main() -> None:
    config = load_config()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts: DatasetArtifacts = run_feature_engineering()

    models = configure_models(config["model_config"]["random_state"])
    cv_summary, best_result = evaluate_models(models, artifacts, config["model_config"]["random_state"])

    plot_cv_results(cv_summary, OUTPUTS_DIR / "model_cv_comparison.png")

    test_results = evaluate_test_set(best_result, artifacts, config["model_config"]["random_state"])

    model_path = persist_best_model(
        best_result.name,
        test_results["best_estimator"],
        cv_summary,
        test_results["roc_auc_macro"],
        artifacts,
    )

    print("\nResumen final:")
    print(cv_summary)
    print("\nReporte de clasificación (test):")
    print(test_results["classification_summary"].to_string(index=False))
    print(f"Modelo guardado en: {model_path}")
    print(f"Matriz de confusión: {test_results['confusion_matrix_path']}")


if __name__ == "__main__":
    main()
