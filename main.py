"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
import os

test_ids = None

def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    global test_ids
    os.makedirs('results', exist_ok=True)
    submission = pd.DataFrame({
        'id': test_ids,
        'prediction': predictions
    })
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_df = pd.read_csv(os.path.join(os.path.join(os.path.dirname(__file__), "data"), "train.csv"))
    test_df = pd.read_csv(os.path.join(os.path.join(os.path.dirname(__file__), "data"), "test.csv"))

    global test_ids
    test_ids = test_df['id'].values
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = str(text).lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-z0-9\\s]', ' ', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    
    for field in ['query', 'product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']:
        train_df[f'{field}_clean'] = train_df[field].fillna('').apply(clean_text)
        test_df[f'{field}_clean'] = test_df[field].fillna('').apply(clean_text)
    
    train_df['product_text'] = (
        train_df['product_title_clean'] + ' ' +
        train_df['product_brand_clean'] + ' ' +
        train_df['product_description_clean'] + ' ' +
        train_df['product_bullet_point_clean']
    )
    test_df['product_text'] = (
        test_df['product_title_clean'] + ' ' +
        test_df['product_brand_clean'] + ' ' +
        test_df['product_description_clean'] + ' ' +
        test_df['product_bullet_point_clean']
    )
    
    for df in [train_df, test_df]:
        df['query_len'] = df['query_clean'].str.len()
        df['title_len'] = df['product_title_clean'].str.len()
        df['desc_len'] = df['product_description_clean'].str.len()
        df['query_words'] = df['query_clean'].str.split().str.len()
        df['title_words'] = df['product_title_clean'].str.split().str.len()
        
        def simple_overlap(row):
            q_words = set(row['query_clean'].split())
            t_words = set(row['product_title_clean'].split())
            if not q_words:
                return 0
            return len(q_words.intersection(t_words)) / len(q_words)
        
        df['word_overlap'] = df.apply(simple_overlap, axis=1)
        df['exact_match'] = df.apply(lambda x: 1 if x['query_clean'] in x['product_title_clean'] else 0, axis=1)
    
    all_queries = pd.concat([train_df['query_clean'], test_df['query_clean']]).unique()
    all_products = pd.concat([train_df['product_text'], test_df['product_text']]).unique()
    
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 3))
    all_tfidf_queries = tfidf.fit_transform(all_queries)
    all_tfidf_products = tfidf.transform(all_products)
    
    n_train_q = len(train_df['query_clean'].unique())
    n_train_p = len(train_df['product_text'].unique())
    
    svd = TruncatedSVD(n_components=50, random_state=993)
    svd_queries = svd.fit_transform(all_tfidf_queries)
    svd_products = svd.transform(all_tfidf_products)
    
    pca = PCA(n_components=30, random_state=993)
    pca_queries = pca.fit_transform(all_tfidf_queries.toarray())
    pca_products = pca.transform(all_tfidf_products.toarray())
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    query_emb_train = model.encode(train_df['query_clean'].values, batch_size=128, show_progress_bar=True, device=device)
    product_emb_train = model.encode(train_df['product_text'].values, batch_size=128, show_progress_bar=True, device=device)
    
    query_emb_test = model.encode(test_df['query_clean'].values, batch_size=128, show_progress_bar=True, device=device)
    product_emb_test = model.encode(test_df['product_text'].values, batch_size=128, show_progress_bar=True, device=device)
    
    train_df['bert_cos_sim'] = [util.cos_sim(q, p).item() for q, p in zip(query_emb_train, product_emb_train)]
    test_df['bert_cos_sim'] = [util.cos_sim(q, p).item() for q, p in zip(query_emb_test, product_emb_test)]
    
    for i in range(384):
        train_df[f'query_emb_{i}'] = query_emb_train[:, i]
        train_df[f'product_emb_{i}'] = product_emb_train[:, i]
        test_df[f'query_emb_{i}'] = query_emb_test[:, i]
        test_df[f'product_emb_{i}'] = product_emb_test[:, i]
    
    combined_emb_train = np.hstack((query_emb_train, product_emb_train))
    combined_emb_test = np.hstack((query_emb_test, product_emb_test))
    
    kmeans = KMeans(n_clusters=10, random_state=993, n_init=10)
    train_df['kmeans_cluster'] = kmeans.fit_predict(combined_emb_train)
    test_df['kmeans_cluster'] = kmeans.predict(combined_emb_test)
    
    iso = IsolationForest(contamination=0.05, random_state=993)
    train_df['iso_score'] = iso.fit_predict(combined_emb_train)
    test_df['iso_score'] = iso.predict(combined_emb_test)
    
    lof = LocalOutlierFactor(contamination=0.05, n_neighbors=30, novelty=True)
    lof.fit(combined_emb_train)
    train_df['lof_score'] = lof.decision_function(combined_emb_train)
    test_df['lof_score'] = lof.decision_function(combined_emb_test)
    
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(combined_emb_train)
    distances, _ = knn.kneighbors(combined_emb_train)
    train_df['knn_mean_dist'] = distances.mean(axis=1)
    train_df['knn_min_dist'] = distances.min(axis=1)
    distances_test, _ = knn.kneighbors(combined_emb_test)
    test_df['knn_mean_dist'] = distances_test.mean(axis=1)
    test_df['knn_min_dist'] = distances_test.min(axis=1)
    
    for col in ['product_brand', 'product_color', 'product_locale']:
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
        
        freq = train_df[col].value_counts(normalize=True)
        train_df[f'{col}_freq'] = train_df[col].map(freq)
        test_df[f'{col}_freq'] = test_df[col].map(freq).fillna(0)
        
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[f'{col}_encoded'] = le.transform(train_df[col])
        test_df[f'{col}_encoded'] = le.transform(test_df[col])
    
    le_query = LabelEncoder()
    all_queries = pd.concat([train_df['query_id'], test_df['query_id']])
    le_query.fit(all_queries)
    train_df['query_id_encoded'] = le_query.transform(train_df['query_id'])
    test_df['query_id_encoded'] = le_query.transform(test_df['query_id'])
    
    features = [
        'query_len', 'title_len', 'desc_len',
        'query_words', 'title_words',
        'word_overlap', 'exact_match',
        'bert_cos_sim',
        'kmeans_cluster', 'iso_score', 'lof_score',
        'knn_mean_dist', 'knn_min_dist',
        'product_brand_freq', 'product_color_freq', 'product_locale_freq',
        'product_brand_encoded', 'product_color_encoded', 'product_locale_encoded'
    ]
    
    svd_emb = TruncatedSVD(n_components=100, random_state=993)
    combined_emb_train_svd = svd_emb.fit_transform(combined_emb_train)
    combined_emb_test_svd = svd_emb.transform(combined_emb_test)
    
    for i in range(100):
        features.append(f'emb_svd_{i}')
        train_df[f'emb_svd_{i}'] = combined_emb_train_svd[:, i]
        test_df[f'emb_svd_{i}'] = combined_emb_test_svd[:, i]
    
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    
    y_train = train_df['relevance'].values
    groups_train = train_df['query_id_encoded'].values
    groups_test = test_df['query_id_encoded'].values
    
    def calculate_ndcg(predictions, relevances, groups, k=10):
        ndcg_scores = []
        unique_groups = np.unique(groups)
        
        for group_id in unique_groups:
            mask = groups == group_id
            group_preds = predictions[mask]
            group_rels = relevances[mask]
            
            if len(group_preds) < 2:
                continue
            
            sorted_idx = np.argsort(group_preds)[::-1]
            sorted_rels = group_rels[sorted_idx]
            
            dcg = 0
            for i, rel in enumerate(sorted_rels[:k]):
                dcg += (2 ** rel - 1) / np.log2(i + 2)
            
            ideal_sorted = np.sort(group_rels)[::-1]
            idcg = 0
            for i, rel in enumerate(ideal_sorted[:k]):
                idcg += (2 ** rel - 1) / np.log2(i + 2)
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0
    
    n_folds = 5
    group_kfold = GroupKFold(n_splits=n_folds)
    
    oof_preds_lgb = np.zeros(len(X_train))
    test_preds_lgb = np.zeros(len(X_test))
    
    oof_preds_cb = np.zeros(len(X_train))
    test_preds_cb = np.zeros(len(X_test))
    
    print(f"Кросс-валидация ({n_folds} фолдов):")
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, y_train, groups_train)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        groups_tr = groups_train[train_idx]
        groups_val = groups_train[val_idx]
        
        group_sizes_tr = np.bincount(groups_tr)
        group_sizes_tr = group_sizes_tr[group_sizes_tr > 0]
        group_sizes_val = np.bincount(groups_val)
        group_sizes_val = group_sizes_val[group_sizes_val > 0]
        
        lgb_model = lgb.LGBMRanker(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=63,
            objective="lambdarank",
            metric="ndcg",
            lambdarank_truncation_level=10,
            random_state=993 + fold,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            group=group_sizes_tr,
            eval_set=[(X_val, y_val)],
            eval_group=[group_sizes_val],
            eval_metric='ndcg'
        )
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
        test_preds_lgb += lgb_model.predict(X_test) / n_folds
        
        train_pool = cb.Pool(data=X_tr, label=y_tr, group_id=groups_tr)
        val_pool = cb.Pool(data=X_val, label=y_val, group_id=groups_val)
        
        cb_model = cb.CatBoostRanker(
            iterations=300,
            learning_rate=0.03,
            depth=8,
            loss_function='YetiRank',
            random_seed=42 + fold,
            verbose=0
        )
        cb_model.fit(train_pool, eval_set=val_pool)
        oof_preds_cb[val_idx] = cb_model.predict(X_val)
        test_preds_cb += cb_model.predict(X_test) / n_folds
    
    ndcg_lgb = calculate_ndcg(oof_preds_lgb, y_train, groups_train)
    ndcg_cb = calculate_ndcg(oof_preds_cb, y_train, groups_train)
    
    test_ensemble = 0.6 * test_preds_lgb + 0.4 * test_preds_cb
    
    meta_train = np.column_stack([
        X_train.values,
        oof_preds_lgb,
        oof_preds_cb
    ])
    meta_test = np.column_stack([
        X_test.values,
        test_preds_lgb,
        test_preds_cb
    ])
    
    meta_model = cb.CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
    meta_model.fit(meta_train, y_train)
    meta_predictions = meta_model.predict(meta_test)
    
    final_predictions = 0.7 * test_ensemble + 0.3 * meta_predictions
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(final_predictions)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()