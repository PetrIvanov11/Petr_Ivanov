"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans


def mean_iou(y_true, y_pred, epsilon=1e-6):
    """Вычисление средней IoU метрики"""
    lower_true = y_true[:, 0]
    upper_true = y_true[:, 1]
    lower_pred = y_pred[:, 0]
    upper_pred = y_pred[:, 1]
    intersection = np.maximum(0, np.minimum(upper_true, upper_pred) - np.maximum(lower_true, lower_pred))
    union = (upper_true - lower_true) + (upper_pred - lower_pred) - intersection + epsilon
    iou = intersection / union
    return np.mean(iou)


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    import os
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    print(f"Submission файл сохранен: {submission_path}")
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # 1. Загрузка данных
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    train_df['dt'] = pd.to_datetime(train_df['dt'])
    test_df['dt'] = pd.to_datetime(test_df['dt'])
    
    # 2. Обработка аномалий
    iso = IsolationForest(contamination=0.01, random_state=322)
    anomalies = iso.fit_predict(train_df[['price_p05', 'price_p95']])
    train_df = train_df[anomalies == 1]
    
    # 3. PCA для погодных признаков
    weather_feats = ['precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    pca = PCA(n_components=2, random_state=322)
    train_weather_pca = pca.fit_transform(train_df[weather_feats])
    train_df['pca1'] = train_weather_pca[:, 0]
    train_df['pca2'] = train_weather_pca[:, 1]
    
    test_weather_pca = pca.transform(test_df[weather_feats])
    test_df['pca1'] = test_weather_pca[:, 0]
    test_df['pca2'] = test_weather_pca[:, 1]
    
    # 4. Кластеризация продуктов
    category_feats = ['first_category_id', 'second_category_id', 'third_category_id']
    product_category = train_df.groupby('product_id')[category_feats].first()
    kmeans = KMeans(n_clusters=10, random_state=322)
    product_clusters = kmeans.fit_predict(product_category)
    
    cluster_map = dict(zip(product_category.index, product_clusters))
    train_df['cluster'] = train_df['product_id'].map(cluster_map)
    
    test_product_category = test_df.groupby('product_id')[category_feats].first()
    test_clusters = kmeans.predict(test_product_category)
    test_cluster_map = dict(zip(test_product_category.index, test_clusters))
    test_df['cluster'] = test_df['product_id'].map(test_cluster_map)
    
    # 5. Подготовка фичей и категориальных признаков
    features = ['n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
                'holiday_flag', 'activity_flag', 'management_group_id', 'first_category_id',
                'second_category_id', 'third_category_id', 'dow', 'day_of_month', 'week_of_year', 'month',
                'pca1', 'pca2', 'cluster', 'product_id']
    
    categorical_features = ['holiday_flag', 'activity_flag', 'management_group_id', 'first_category_id',
                            'second_category_id', 'third_category_id', 'dow', 'month', 'cluster', 'product_id']
    
    # 6. Параметры модели
    params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'seed': 322
    }
    
    # 7. Разделение на train/validation
    val_mask = train_df['dt'] >= '2024-05-20'
    train_set = train_df[~val_mask]
    val_set = train_df[val_mask]
    
    # 8. Обучение модели для price_p05
    train_data_p05 = lgb.Dataset(train_set[features], label=train_set['price_p05'], 
                                 categorical_feature=categorical_features)
    valid_data_p05 = lgb.Dataset(val_set[features], label=val_set['price_p05'], 
                                 categorical_feature=categorical_features, reference=train_data_p05)
    model_p05 = lgb.train({**params, 'alpha': 0.05}, train_data_p05, num_boost_round=2000,
                          valid_sets=[valid_data_p05], 
                          callbacks=[lgb.early_stopping(stopping_rounds=100)])
    
    # 9. Обучение модели для price_p95
    train_data_p95 = lgb.Dataset(train_set[features], label=train_set['price_p95'], 
                                 categorical_feature=categorical_features)
    valid_data_p95 = lgb.Dataset(val_set[features], label=val_set['price_p95'], 
                                 categorical_feature=categorical_features, reference=train_data_p95)
    model_p95 = lgb.train({**params, 'alpha': 0.95}, train_data_p95, num_boost_round=2000,
                          valid_sets=[valid_data_p95], 
                          callbacks=[lgb.early_stopping(stopping_rounds=100)])
    
    # 10. Валидация
    val_pred_p05 = model_p05.predict(val_set[features])
    val_pred_p95 = model_p95.predict(val_set[features])
    y_pred = np.column_stack((val_pred_p05, val_pred_p95))
    
    # Исправление порядка предсказаний (p05 <= p95)
    mask = val_pred_p05 > val_pred_p95
    y_pred[mask, 0], y_pred[mask, 1] = y_pred[mask, 1], y_pred[mask, 0]
    
    y_true = val_set[['price_p05', 'price_p95']].values
    iou_score = mean_iou(y_true, y_pred)
    print('Validation IoU:', iou_score)
    
    # 11. Обучение на полных данных
    best_iter_p05 = model_p05.best_iteration
    best_iter_p95 = model_p95.best_iteration
    
    full_train_data_p05 = lgb.Dataset(train_df[features], label=train_df['price_p05'], 
                                      categorical_feature=categorical_features)
    full_model_p05 = lgb.train({**params, 'alpha': 0.05}, full_train_data_p05, 
                               num_boost_round=best_iter_p05)
    
    full_train_data_p95 = lgb.Dataset(train_df[features], label=train_df['price_p95'], 
                                      categorical_feature=categorical_features)
    full_model_p95 = lgb.train({**params, 'alpha': 0.95}, full_train_data_p95, 
                               num_boost_round=best_iter_p95)
    
    # 12. Предсказание на тестовых данных
    test_pred_p05 = full_model_p05.predict(test_df[features])
    test_pred_p95 = full_model_p95.predict(test_df[features])
    
    # Исправление порядка предсказаний
    mask = test_pred_p05 > test_pred_p95
    test_pred_p05[mask], test_pred_p95[mask] = test_pred_p95[mask], test_pred_p05[mask]
    
    # 13. Создание submission файла
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'price_p05': test_pred_p05,
        'price_p95': test_pred_p95
    })
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()