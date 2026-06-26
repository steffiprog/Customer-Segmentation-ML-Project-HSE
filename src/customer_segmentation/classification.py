"""Обучение классификатора, предсказывающего кластер пользователя."""

import time

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


def build_preprocessor(numeric_features: list) -> ColumnTransformer:
    """Создаёт препроцессор, нормализующий числовые признаки."""
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)]
    )


def build_models(preprocessor: ColumnTransformer, random_state: int = 42) -> dict:
    """Собирает словарь моделей-пайплайнов для сравнения."""
    models = {}

    models["Logistic Regression (balanced)"] = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["Logistic Regression + SMOTE"] = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state, k_neighbors=2)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )

    models["Random Forest (balanced)"] = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["Random Forest + SMOTE-Tomek"] = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote_tomek", SMOTETomek(random_state=random_state)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["XGBoost (sample_weight)"] = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric="mlogloss",
                ),
            ),
        ]
    )

    models["SVM (balanced)"] = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                SVC(
                    class_weight="balanced",
                    kernel="linear",
                    random_state=random_state,
                    probability=True,
                ),
            ),
        ]
    )

    return models


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    use_sample_weight: bool = False,
) -> dict:
    """Обучает модель, вычисляет метрики и возвращает словарь с результатами."""
    start_time = time.time()

    if use_sample_weight and "XGBoost" in model_name:
        classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        sample_weights = np.array(
            [class_weights[list(classes).index(cls)] for cls in y_train]
        )
        model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_time
    y_pred = model.predict(X_test)

    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": float(np.mean(recall_per_class)),
        "train_time": train_time,
        "recall_per_class": recall_per_class,
        "y_pred": y_pred,
        "model_object": model,
    }


def compare_models(results: list) -> pd.DataFrame:
    """Строит сводную таблицу метрик и сортирует по Balanced Accuracy."""
    df_results = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Accuracy": r["accuracy"],
                "Balanced Acc": r["balanced_accuracy"],
                "F1-macro": r["f1_macro"],
                "Recall-macro": r["recall_macro"],
                "Train time (s)": r["train_time"],
            }
            for r in results
        ]
    )
    return df_results.sort_values("Balanced Acc", ascending=False)


def select_best_model(results: list, df_results: pd.DataFrame):
    """Возвращает (имя_модели, словарь_результата) лучшей модели по Balanced Accuracy."""
    best_model_name = df_results.iloc[0]["Model"]
    best_result = next(r for r in results if r["model"] == best_model_name)
    return best_model_name, best_result
