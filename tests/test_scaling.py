import numpy as np
import pandas as pd

from src.customer_segmentation.scaling import (
    clip_outliers,
    prepare_for_clustering,
    scale_features,
)


def test_clip_outliers_replaces_inf_and_nan():
    df = pd.DataFrame({"a": [1, 2, np.inf, -np.inf, np.nan, 100]})
    result = clip_outliers(df)
    assert np.isfinite(result.values).all()
    assert not result.isna().any().any()


def test_clip_outliers_clips_extreme_values():
    df = pd.DataFrame({"a": list(range(100)) + [10000]})
    result = clip_outliers(df, lower_q=0.01, upper_q=0.99)
    assert result["a"].max() < 10000


def test_scale_features_zero_mean_unit_variance():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10, 20, 30, 40, 50]})
    scaled_df, scaler = scale_features(df)
    assert abs(scaled_df["a"].mean()) < 1e-9
    assert abs(scaled_df["b"].mean()) < 1e-9


def test_scale_features_reuses_existing_scaler():
    df_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    _, scaler = scale_features(df_train)

    df_new = pd.DataFrame({"a": [4.0, 5.0]})
    scaled_new, scaler_used = scale_features(df_new, scaler=scaler)

    assert scaler_used is scaler
    assert len(scaled_new) == 2


def test_prepare_for_clustering_returns_expected_shape():
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5],
            "f2": [10, 20, 30, 40, 50],
            "unused": [0, 0, 0, 0, 0],
        }
    )
    X_scaled_df, scaler = prepare_for_clustering(df, features=["f1", "f2"])
    assert list(X_scaled_df.columns) == ["f1", "f2"]
    assert X_scaled_df.shape == (5, 2)
