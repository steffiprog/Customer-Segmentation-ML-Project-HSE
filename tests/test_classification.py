import numpy as np
import pandas as pd
import pytest

from src.customer_segmentation.classification import (
    build_models,
    build_preprocessor,
    compare_models,
    evaluate_model,
    select_best_model,
)


@pytest.fixture
def classification_data():
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="cluster")
    X_train, X_test = X.iloc[:40], X.iloc[40:]
    y_train, y_test = y.iloc[:40], y.iloc[40:]
    return X_train, X_test, y_train, y_test


def test_build_preprocessor_has_expected_columns():
    preprocessor = build_preprocessor(["f1", "f2"])
    assert preprocessor.transformers[0][2] == ["f1", "f2"]


def test_build_models_returns_non_empty_dict():
    preprocessor = build_preprocessor(["f1", "f2"])
    models = build_models(preprocessor)
    assert len(models) > 0
    assert "Logistic Regression (balanced)" in models


def test_evaluate_model_returns_expected_keys(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    preprocessor = build_preprocessor(["f1", "f2"])
    models = build_models(preprocessor)
    model = models["Logistic Regression (balanced)"]

    result = evaluate_model(model, X_train, y_train, X_test, y_test, "Logistic Regression (balanced)")

    for key in ["model", "accuracy", "balanced_accuracy", "f1_macro", "recall_macro", "model_object"]:
        assert key in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_compare_models_sorts_by_balanced_accuracy():
    results = [
        {"model": "A", "accuracy": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5,
         "recall_macro": 0.5, "train_time": 0.1},
        {"model": "B", "accuracy": 0.9, "balanced_accuracy": 0.9, "f1_macro": 0.9,
         "recall_macro": 0.9, "train_time": 0.1},
    ]
    df_results = compare_models(results)
    assert df_results.iloc[0]["Model"] == "B"


def test_select_best_model_returns_best_result():
    results = [
        {"model": "A", "accuracy": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5,
         "recall_macro": 0.5, "train_time": 0.1},
        {"model": "B", "accuracy": 0.9, "balanced_accuracy": 0.9, "f1_macro": 0.9,
         "recall_macro": 0.9, "train_time": 0.1},
    ]
    df_results = compare_models(results)
    best_name, best_result = select_best_model(results, df_results)
    assert best_name == "B"
    assert best_result["model"] == "B"
