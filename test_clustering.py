import numpy as np

from src.customer_segmentation.clustering import (
    cluster_summary,
    find_optimal_k,
    run_dbscan,
    run_kmeans,
    run_optics,
)


def _make_blobs():
    rng = np.random.RandomState(0)
    cluster_a = rng.normal(loc=0, scale=0.5, size=(20, 2))
    cluster_b = rng.normal(loc=10, scale=0.5, size=(20, 2))
    return np.vstack([cluster_a, cluster_b])


def test_run_kmeans_returns_expected_number_of_labels():
    X = _make_blobs()
    model, labels = run_kmeans(X, n_clusters=2, random_state=0)
    assert len(labels) == len(X)
    assert len(set(labels)) == 2


def test_find_optimal_k_returns_metrics_for_each_k():
    X = _make_blobs()
    result = find_optimal_k(X, k_range=range(2, 4), random_state=0)
    assert list(result["k"]) == [2, 3]
    assert "wcss" in result.columns
    assert "silhouette" in result.columns


def test_run_dbscan_detects_two_well_separated_clusters():
    X = _make_blobs()
    model, labels = run_dbscan(X, eps=1.0, min_samples=3)
    summary = cluster_summary(labels)
    assert summary["n_clusters"] == 2


def test_run_optics_runs_without_errors():
    X = _make_blobs()
    model, labels = run_optics(X, max_eps=2.0, min_samples=3)
    assert len(labels) == len(X)


def test_cluster_summary_counts_noise_points():
    labels = np.array([0, 0, 1, 1, -1, -1, -1])
    summary = cluster_summary(labels)
    assert summary["n_clusters"] == 2
    assert summary["n_noise"] == 3
