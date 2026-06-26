# Customer Segmentation for Marketing Targeting

---

## Team

| Role | Name | GitHub |
|------|------|--------|
| Team Lead & Data Engineer 1 | Stefania | [https://github.com/steffiprog] |
| ML Engineer | Anna | [https://github.com/theMorana] |
| Data Engineer 2 & Data Analyst | Albina | [https://github.com/friendementiaa] |


# Customer Segmentation

Сегментация пользователей C2C fashion-маркетплейса на основе их поведенческой
активности: загрузка данных → очистка → feature engineering → кластеризация
(K-Means / DBSCAN / OPTICS) → обучение классификатора, который предсказывает
кластер нового пользователя по его признакам.

Датасет: [Ecommerce Users of a French C2C Fashion Store](https://www.kaggle.com/datasets/jmmvutu/ecommerce-users-of-a-french-c2c-fashion-store) (Kaggle).

## Структура проекта

```
customer_segmentation/
│
├── data/                              # Данные проекта (генерируются main.py, в git не попадают)
│   ├── users.6M0xxK.2024.public.csv   # исходный датасет (положить сюда вручную)
│   ├── cleaned_users_data.csv         # после чистки и feature engineering
│   ├── normalized_data_for_clustering.csv
│   ├── clustered_users_data.csv       # с метками кластеров
│   ├── features_list.txt
│   ├── scaler.pkl
│   └── best_model.pkl
│
├── notebooks/
│   └── eda_original.ipynb             # исходный исследовательский ноутбук (EDA, графики)
│
├── reports/figures/                   # графики, сохраняемые при запуске main.py
│   ├── 00a_categorical_features.png   # категориальные признаки feature engineering
│   ├── 00b_numeric_features.png       # ключевые числовые признаки feature engineering
│   ├── 01_elbow_silhouette.png        # метод локтя + силуэт для выбора K
│   ├── 02_kmeans_pca.png              # кластеры K-Means на PCA-проекции
│   ├── 03_kmeans_sizes.png            # размеры кластеров
│   ├── 04_model_comparison.png        # сравнение метрик классификаторов
│   └── 05_confusion_matrix.png        # матрица ошибок лучшей модели
│
├── src/customer_segmentation/         # production-код пайплайна
│   ├── config.py                      # пути и константы (из config/config.yaml)
│   ├── data_loading.py                # загрузка csv
│   ├── preprocessing.py               # чистка данных
│   ├── feature_engineering.py         # генерация признаков
│   ├── scaling.py                     # обработка выбросов + нормализация
│   ├── clustering.py                  # K-Means / DBSCAN / OPTICS
│   ├── classification.py              # обучение и сравнение моделей-классификаторов
│   └── visualization.py               # графики кластеризации и сравнения моделей
│
├── tests/                             # тесты pytest для каждого модуля
│
├── config/
│   └── config.yaml                    # параметры проекта (пути, random_state, n_clusters)
│
├── main.py                            # запуск всего пайплайна целиком
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

## Установка

```bash
git clone <ссылка на репозиторий>
cd customer_segmentation
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Подготовка данных

Скачайте датасет с Kaggle и положите файл `users.6M0xxK.2024.public.csv`
в папку `data/` (путь и имя файла настраиваются в `config/config.yaml`).

## Запуск пайплайна

```bash
python main.py
```

Скрипт последовательно:
1. загружает исходные данные;
2. чистит их (удаляет константные и дублирующие признаки);
3. генерирует новые признаки (категориальные, коэффициенты, агрегаты,
   композитные индексы, бинарные флаги) и сохраняет графики их распределения;
4. обрабатывает выбросы и нормализует данные (`StandardScaler`);
5. кластеризует пользователей методом K-Means (с подбором K по локтю/силуэту);
6. обучает и сравнивает несколько классификаторов (Logistic Regression,
   Random Forest, XGBoost, SVM — с балансировкой классов и SMOTE), сохраняет
   лучшую модель в `data/best_model.pkl`.

На шагах 5 и 6 сохраняются графики в `reports/figures/`: метод локтя и силуэт,
PCA-проекция кластеров, размеры кластеров, сравнение метрик моделей и матрица
ошибок лучшей модели.

Все промежуточные и итоговые артефакты (csv, модели, графики) сохраняются
в папки `data/` и `reports/figures/`.

## Тесты

```bash
pytest
```

Тесты покрывают каждый модуль пайплайна (`preprocessing`, `feature_engineering`,
`scaling`, `clustering`, `classification`) на небольших синтетических данных,
без необходимости скачивать полный датасет.

## EDA

Исходный разведочный анализ (графики распределений, корреляции и т.д.)
сохранён в `notebooks/eda_original.ipynb` для справки — в production-код
он не переносился, так как не является частью пайплайна.
