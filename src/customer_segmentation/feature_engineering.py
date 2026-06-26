"""Feature engineering: категориальные, числовые и композитные признаки."""

import pandas as pd


def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет категориальные признаки: тип активности и уровни (tiers)."""
    df = df.copy()

    df["activity_type"] = "неактивные"
    df.loc[df["productsBought"] > 0, "activity_type"] = "только покупают"
    df.loc[df["productsSold"] > 0, "activity_type"] = "только продают"
    df.loc[
        (df["productsBought"] > 0) & (df["productsSold"] > 0), "activity_type"
    ] = "и то и другое"

    df["buyer_tier"] = pd.cut(
        df["productsBought"],
        bins=[-1, 0, 5, 20, float("inf")],
        labels=["Нет покупок", "Низкий", "Средний", "Высокий"],
    )

    df["seller_tier"] = pd.cut(
        df["productsSold"],
        bins=[-1, 0, 2, 10, float("inf")],
        labels=["Нет продаж", "Низкий", "Средний", "Высокий"],
    )

    df["social_tier"] = pd.cut(
        df["socialProductsLiked"],
        bins=[-1, 0, 10, 50, float("inf")],
        labels=["Нет лайков", "Низкий", "Средний", "Высокий"],
    )

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет коэффициенты и пропорции."""
    df = df.copy()
    df["buy_sell_ratio"] = df["productsBought"] / (df["productsSold"] + 1)
    df["wish_to_buy_ratio"] = df["productsWished"] / (df["productsBought"] + 1)
    df["like_to_buy_ratio"] = df["socialProductsLiked"] / (df["productsBought"] + 1)
    df["sales_efficiency"] = df["productsSold"] / (df["productsListed"] + 1)
    df["activity_density"] = (
        df["productsBought"] + df["productsSold"] + df["socialProductsLiked"]
    ) / (df["seniorityAsYears"] + 1)
    return df


def add_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет агрегированные признаки активности."""
    df = df.copy()
    df["total_transactions"] = df["productsBought"] + df["productsSold"]
    df["total_social_activity"] = (
        df["socialNbFollowers"] + df["socialNbFollows"] + df["socialProductsLiked"]
    )
    df["total_engagement"] = (
        df["productsBought"]
        + df["productsSold"]
        + df["productsWished"]
        + df["socialProductsLiked"]
    )
    df["product_portfolio_size"] = df["productsListed"] + df["productsWished"]
    return df


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет композитные индексы (предприниматель, мечтатель и т.д.)."""
    df = df.copy()
    df["entrepreneur_score"] = df["productsSold"] * (df["productsPassRate"] / 100)
    df["dreamer_score"] = df["productsWished"] / (df["productsBought"] + 1)
    df["social_buyer_score"] = df["socialProductsLiked"] * df["productsBought"]
    df["loyalty_score"] = df["seniorityAsYears"] * (
        df["productsBought"] + df["productsSold"] + 1
    )
    return df


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет бинарные флаги (про-продавец, шопоголик, инфлюенсер и т.д.)."""
    df = df.copy()
    df["is_pro_seller"] = (
        df["productsSold"] > df["productsSold"].quantile(0.9)
    ).astype(int)
    df["is_shopaholic"] = (
        df["productsBought"] > df["productsBought"].quantile(0.9)
    ).astype(int)
    df["is_influencer"] = (
        df["socialNbFollowers"] > df["socialNbFollowers"].quantile(0.9)
    ).astype(int)
    df["has_social_activity"] = (
        (df["socialNbFollowers"] > 0)
        | (df["socialNbFollows"] > 0)
        | (df["socialProductsLiked"] > 0)
    ).astype(int)
    df["is_newbie"] = (df["seniorityAsYears"] < 1).astype(int)
    df["is_veteran"] = (df["seniorityAsYears"] > 5).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Запускает полный пайплайн feature engineering."""
    df = add_categorical_features(df)
    df = add_ratio_features(df)
    df = add_aggregated_features(df)
    df = add_composite_scores(df)
    df = add_binary_flags(df)
    return df


CLUSTERING_FEATURES = [
    "productsBought",
    "productsSold",
    "productsWished",
    "socialProductsLiked",
    "seniorityAsYears",
    "socialNbFollowers",
    "socialNbFollows",
    "productsPassRate",
    "buy_sell_ratio",
    "wish_to_buy_ratio",
    "like_to_buy_ratio",
    "sales_efficiency",
    "activity_density",
    "total_transactions",
    "total_social_activity",
    "total_engagement",
    "product_portfolio_size",
    "entrepreneur_score",
    "dreamer_score",
    "social_buyer_score",
    "loyalty_score",
]


def select_clustering_features(df: pd.DataFrame, features: list = None) -> list:
    """Возвращает список признаков для кластеризации, реально присутствующих в df."""
    features = features if features is not None else CLUSTERING_FEATURES
    return [f for f in features if f in df.columns]
