"""Подготовка признаков к кластеризации: обработка выбросов и нормализация."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clip_outliers(X: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """Заменяет выбросы на значения заданных перцентилей (winsorization)."""
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    for column in X.columns:
        lower = X[column].quantile(lower_q)
        upper = X[column].quantile(upper_q)
        X[column] = X[column].clip(lower=lower, upper=upper)
    return X


def scale_features(X: pd.DataFrame, scaler: StandardScaler = None):
    """Нормализует признаки StandardScaler'ом.

    Если scaler не передан, обучает новый и возвращает его вместе с данными.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df, scaler


def prepare_for_clustering(df: pd.DataFrame, features: list):
    """Полный пайплайн подготовки: выбор признаков -> обработка выбросов -> нормализация."""
    X = df[features].copy()
    X = clip_outliers(X)
    X_scaled_df, scaler = scale_features(X)
    return X_scaled_df, scaler
