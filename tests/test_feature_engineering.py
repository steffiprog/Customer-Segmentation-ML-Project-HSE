from src.customer_segmentation.feature_engineering import (
    CLUSTERING_FEATURES,
    add_aggregated_features,
    add_binary_flags,
    add_categorical_features,
    add_composite_scores,
    add_ratio_features,
    engineer_features,
    select_clustering_features,
)
from src.customer_segmentation.preprocessing import clean_data


def test_add_categorical_features_activity_type(raw_sample_df):
    df = clean_data(raw_sample_df)
    df = add_categorical_features(df)
    # row 0: bought=3, sold=0 -> только покупают
    assert df.loc[0, "activity_type"] == "только покупают"
    # row 1: bought=0, sold=5 -> только продают
    assert df.loc[1, "activity_type"] == "только продают"
    # row 2: bought=8, sold=1 -> и то и другое
    assert df.loc[2, "activity_type"] == "и то и другое"
    # row 3: bought=1, sold=0 -> только покупают
    assert df.loc[3, "activity_type"] == "только покупают"


def test_add_ratio_features_no_division_by_zero(raw_sample_df):
    df = clean_data(raw_sample_df)
    df = add_ratio_features(df)
    assert df["buy_sell_ratio"].notna().all()
    assert df["sales_efficiency"].notna().all()


def test_add_aggregated_features_values(raw_sample_df):
    df = clean_data(raw_sample_df)
    df = add_aggregated_features(df)
    assert df.loc[0, "total_transactions"] == 3  # 3 + 0
    assert df.loc[2, "total_transactions"] == 9  # 8 + 1


def test_add_composite_scores_entrepreneur(raw_sample_df):
    df = clean_data(raw_sample_df)
    df = add_composite_scores(df)
    # row1: sold=5, passRate=80 -> 5 * 0.8 = 4.0
    assert abs(df.loc[1, "entrepreneur_score"] - 4.0) < 1e-9


def test_add_binary_flags_are_0_or_1(raw_sample_df):
    df = clean_data(raw_sample_df)
    df = add_binary_flags(df)
    flag_cols = [
        "is_pro_seller",
        "is_shopaholic",
        "is_influencer",
        "has_social_activity",
        "is_newbie",
        "is_veteran",
    ]
    for col in flag_cols:
        assert set(df[col].unique()).issubset({0, 1})


def test_engineer_features_full_pipeline_runs(raw_sample_df):
    df = clean_data(raw_sample_df)
    result = engineer_features(df)
    assert len(result) == len(df)
    # все ожидаемые новые признаки должны существовать
    for col in ["activity_type", "buy_sell_ratio", "total_transactions",
                "entrepreneur_score", "is_newbie"]:
        assert col in result.columns


def test_select_clustering_features_filters_missing(raw_sample_df):
    df = clean_data(raw_sample_df)
    df = engineer_features(df)
    selected = select_clustering_features(df, CLUSTERING_FEATURES + ["does_not_exist"])
    assert "does_not_exist" not in selected
    assert set(selected).issubset(set(df.columns))
