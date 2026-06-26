"""Чистка исходных данных: удаление константных и бесполезных признаков."""

import pandas as pd

COLUMNS_TO_DROP = [
    "identifierHash",
    "type",
    "gender",
    "civilityGenderId",
    "civilityTitle",
    "websiteLongevity",
    "seniority",
    "seniorityAsMonths",
    "countryCode",
]


def find_constant_features(df: pd.DataFrame) -> list:
    """Возвращает список признаков, у которых только одно уникальное значение."""
    return [col for col in df.columns if df[col].nunique() <= 1]


def drop_useless_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Удаляет заранее известные бесполезные/дублирующие признаки."""
    columns = columns if columns is not None else COLUMNS_TO_DROP
    columns = [col for col in columns if col in df.columns]
    return df.drop(columns=columns)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Полная чистка данных: удаление константных и бесполезных признаков.

    Дубликаты строк не удаляются: разные пользователи могут иметь
    одинаковые характеристики поведения.
    """
    df = df.copy()
    constant_features = find_constant_features(df)
    df = drop_useless_columns(df, COLUMNS_TO_DROP + constant_features)
    return df
