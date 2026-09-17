"""Логика рекомендательной модели: retrieval + CatBoost ranking."""
import logging
import pandas as pd
from catboost import CatBoostClassifier, Pool

from app.config import (
    MODEL_PATH, POPULARITY_PATH, PRESELECT_N, TOP_K,
    FEATURE_COLUMNS, CAT_FEATURES
)

logger = logging.getLogger(__name__)


class RecommenderModel:
    """
    Гибридная модель: 
    1. Retrieval — отбор топ-N по популярности.
    2. Ranking — CatBoost ранжирует оставшихся.
    """

    def __init__(self):
        logger.info("Loading CatBoost model from %s", MODEL_PATH)
        self.model = CatBoostClassifier()
        self.model.load_model(MODEL_PATH)

        logger.info("Loading popularity snapshot from %s", POPULARITY_PATH)
        self.popularity = pd.read_parquet(POPULARITY_PATH)
        # Колонки: itemid, item_daily_addtocart_last_7d

        # Переименуем для ясности, чтобы не конфликтовало с признаками
        self.popularity = self.popularity.rename(
            columns={'item_daily_addtocart_last_7d': 'retrieval_score'}
        )

        logger.info("Model ready. Features: %d, cat: %d",
                    len(FEATURE_COLUMNS), len(CAT_FEATURES))

    def recommend(self, candidates_df: pd.DataFrame, k: int = TOP_K):
        """
        candidates_df: DataFrame, одна строка на кандидата,
        колонки: itemid + все FEATURE_COLUMNS.

        Возвращает: список из k itemid, отсортированных по убыванию вероятности.
        """
        if candidates_df.empty:
            return []

        # --- Шаг 1. Retrieval: добавляем скор популярности ---
        df = candidates_df.merge(self.popularity, on='itemid', how='left')
        df['retrieval_score'] = df['retrieval_score'].fillna(0)

        top_n = (
            df.sort_values('retrieval_score', ascending=False)
              .head(PRESELECT_N)
              .copy()
        )

        # --- Шаг 2. Ranking: прогон через CatBoost ---
        # Убеждаемся, что все ожидаемые колонки есть
        for col in FEATURE_COLUMNS:
            if col not in top_n.columns:
                top_n[col] = 0

        X = top_n[FEATURE_COLUMNS]
        pool = Pool(X, cat_features=CAT_FEATURES)
        top_n['prob'] = self.model.predict_proba(pool)[:, 1]

        # --- Шаг 3. Топ-k ---
        recommendations = (
            top_n.sort_values('prob', ascending=False)
                 ['itemid']
                 .head(k)
                 .tolist()
        )
        return recommendations