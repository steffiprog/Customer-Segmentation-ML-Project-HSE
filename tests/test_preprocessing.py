from src.customer_segmentation.preprocessing import (
    COLUMNS_TO_DROP,
    clean_data,
    drop_useless_columns,
    find_constant_features,
)


def test_find_constant_features(raw_sample_df):
    constant = find_constant_features(raw_sample_df)
    assert "type" in constant
    assert "gender" in constant
    assert "country" not in constant


def test_drop_useless_columns_removes_known_columns(raw_sample_df):
    result = drop_useless_columns(raw_sample_df)
    for col in COLUMNS_TO_DROP:
        assert col not in result.columns
    assert "productsBought" in result.columns


def test_drop_useless_columns_ignores_missing_columns(raw_sample_df):
    df = raw_sample_df.drop(columns=["identifierHash"])
    result = drop_useless_columns(df)
    assert "identifierHash" not in result.columns


def test_clean_data_removes_constant_and_useless_columns(raw_sample_df):
    result = clean_data(raw_sample_df)
    for col in COLUMNS_TO_DROP:
        assert col not in result.columns
    assert "country" in result.columns
    assert "productsBought" in result.columns


def test_clean_data_does_not_drop_rows(raw_sample_df):
    result = clean_data(raw_sample_df)
    assert len(result) == len(raw_sample_df)
