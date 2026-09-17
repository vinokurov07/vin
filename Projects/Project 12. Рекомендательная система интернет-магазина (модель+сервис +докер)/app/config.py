"""Конфигурация сервиса: пути к артефактам, гиперпараметры гибрида."""
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути к артефактам
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'catboost_best.cbm')
POPULARITY_PATH = os.path.join(BASE_DIR, 'data', 'item_popularity.parquet')
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'feature_columns.json')
CAT_FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'cat_features.json')

# Гиперпараметры гибрида
PRESELECT_N = 20   # сколько кандидатов оставляем после retrieval
TOP_K = 3          # сколько рекомендаций возвращаем

# Загружаем список колонок и категорий из JSON
with open(FEATURE_COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

with open(CAT_FEATURES_PATH) as f:
    CAT_FEATURES = json.load(f)