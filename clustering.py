"""Кластеризация пользователей: K-Means, DBSCAN, OPTICS."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(X_scaled, k_range=range(2, 11), random_state: int = 42) -> pd.DataFrame:
    """Считает WCSS и силуэт для разных K (метод локтя + силуэт)."""
    rows = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "wcss": kmeans.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def run_kmeans(X_scaled, n_clusters: int = 4, random_state: int = 42):
    """Обучает K-Means и возвращает (модель, метки)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


def run_dbscan(X_scaled, eps: float, min_samples: int):
    """Обучает DBSCAN и возвращает (модель, метки)."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return dbscan, labels


def run_optics(X_scaled, max_eps: float, min_samples: int, xi: float = 0.07, min_cluster_size: int = 5):
    """Обучает OPTICS и возвращает (модель, метки)."""
    optics_model = OPTICS(
        max_eps=max_eps,
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        n_jobs=-1,
    )
    labels = optics_model.fit_predict(X_scaled)
    return optics_model, labels


def cluster_summary(labels) -> dict:
    """Возвращает количество кластеров и количество шумовых точек (-1) для DBSCAN/OPTICS."""
    labels = np.asarray(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    return {"n_clusters": n_clusters, "n_noise": n_noise}
