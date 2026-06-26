"""Главный скрипт: запускает весь пайплайн сегментации пользователей.

Запуск:
    python main.py

Ожидает, что исходный csv-файл с пользователями лежит по пути
data/<raw_data_file> (см. config/config.yaml).
"""

import joblib
from sklearn.model_selection import train_test_split

from src.customer_segmentation import classification, clustering, config, visualization
from src.customer_segmentation.data_loading import load_raw_data
from src.customer_segmentation.feature_engineering import (
    engineer_features,
    select_clustering_features,
)
from src.customer_segmentation.preprocessing import clean_data
from src.customer_segmentation.scaling import prepare_for_clustering


def main():
    print("1. Загрузка данных...")
    df = load_raw_data(config.RAW_DATA_FILE)
    print(f"   Загружено {df.shape[0]} строк, {df.shape[1]} столбцов")

    print("2. Чистка данных...")
    df = clean_data(df)

    print("3. Feature engineering...")
    df = engineer_features(df)
    df.to_csv(config.CLEANED_DATA_FILE, index=False)
    visualization.plot_categorical_features(df, config.FIGURES_DIR / "00a_categorical_features.png")
    visualization.plot_numeric_features(df, config.FIGURES_DIR / "00b_numeric_features.png")

    print("4. Подготовка признаков к кластеризации...")
    features = select_clustering_features(df)
    X_scaled_df, scaler = prepare_for_clustering(df, features)
    X_scaled_df.to_csv(config.NORMALIZED_DATA_FILE, index=False)
    joblib.dump(scaler, config.SCALER_FILE)
    with open(config.FEATURES_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("Признаки для кластеризации:\n")
        for feat in features:
            f.write(f"  - {feat}\n")

    print("5. Кластеризация (K-Means)...")
    k_scores = clustering.find_optimal_k(X_scaled_df.values, random_state=config.RANDOM_STATE)
    visualization.plot_elbow_and_silhouette(
        k_scores, config.FIGURES_DIR / "01_elbow_silhouette.png"
    )

    kmeans_model, labels = clustering.run_kmeans(
        X_scaled_df.values, n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE
    )
    visualization.plot_clusters_pca(
        X_scaled_df.values, labels, "K-Means кластеры на PCA-проекции",
        config.FIGURES_DIR / "02_kmeans_pca.png",
    )
    visualization.plot_cluster_sizes(
        labels, "Размеры кластеров (K-Means)", config.FIGURES_DIR / "03_kmeans_sizes.png"
    )

    X_scaled_df["cluster_label"] = labels
    X_scaled_df.to_csv(config.CLUSTERED_DATA_FILE, index=False)

    print("6. Обучение классификатора кластеров...")
    numeric_features = list(X_scaled_df.columns.drop("cluster_label"))
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df[numeric_features],
        X_scaled_df["cluster_label"],
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=X_scaled_df["cluster_label"],
    )

    preprocessor = classification.build_preprocessor(numeric_features)
    models = classification.build_models(preprocessor, random_state=config.RANDOM_STATE)

    results = []
    for name, pipeline in models.items():
        print(f"   Обучение: {name}...")
        use_weight = "XGBoost" in name
        result = classification.evaluate_model(
            pipeline, X_train, y_train, X_test, y_test, name, use_sample_weight=use_weight
        )
        results.append(result)
        print(
            f"     Balanced Accuracy: {result['balanced_accuracy']:.4f}, "
            f"F1-macro: {result['f1_macro']:.4f}"
        )

    df_results = classification.compare_models(results)
    visualization.plot_model_comparison(df_results, config.FIGURES_DIR / "04_model_comparison.png")

    best_name, best_result = classification.select_best_model(results, df_results)
    print(f"\nЛучшая модель: {best_name}")
    print(f"Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")

    visualization.plot_confusion_matrix(
        y_test, best_result["y_pred"], f"Матрица ошибок: {best_name}",
        config.FIGURES_DIR / "05_confusion_matrix.png",
    )

    joblib.dump(best_result["model_object"], config.BEST_MODEL_FILE)
    print(f"Модель сохранена: {config.BEST_MODEL_FILE}")
    print(f"Графики сохранены в: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
