
import argparse
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimentos de prediccion de diabetes"
    )

    parser.add_argument(
        "--data",
        default="dataset/diabetes_prediction_dataset.csv",
        help="Ruta del dataset CSV",
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)

    if "diabetes" not in df.columns:
        raise ValueError("No se encontro la columna diabetes.")

    X = df.drop(columns=["diabetes"])
    y = df["diabetes"]

    return X, y


def build_model(c, max_iter, class_weight, seed):
    variables_categoricas = [
        "gender",
        "smoking_history",
    ]

    variables_numericas = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    ]

    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                variables_categoricas,
            ),
            (
                "numericas",
                "passthrough",
                variables_numericas,
            ),
        ]
    )

    weight = None if class_weight == "none" else "balanced"

    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            (
                "modelo",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    class_weight=weight,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    return modelo


def evaluate_model(model, X_test, y_test, artifact_dir):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
    }

    with open(
        os.path.join(artifact_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ),
        display_labels=["No diabetes", "Diabetes"],
    ).plot()

    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(
        os.path.join(artifact_dir, "matriz_confusion.png")
    )
    plt.close()

    return metrics


def main():
    args = parse_args()

    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.seed,
        stratify=y,
    )

    model = build_model(
        c=args.c,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        seed=args.seed,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Prediccion de diabetes")

    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", args.c)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", 0.20)

        model.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as artifact_dir:
            metrics = evaluate_model(
                model, X_test, y_test, artifact_dir
            )

            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(
                artifact_dir,
                artifact_path="evaluation",
            )

        mlflow.sklearn.log_model(
            model,
            artifact_path="modelo",
        )

        print("Metricas:", metrics)
        print("Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
