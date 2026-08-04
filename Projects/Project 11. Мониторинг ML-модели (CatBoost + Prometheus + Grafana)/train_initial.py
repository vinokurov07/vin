# train_initial.py
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import json
import os

# Загрузка данных (пример, как у вас)
df = pd.read_csv('winequality-red.csv', sep=';')   # или ваш файл
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=100)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)

# Сохраняем модель
model.save_model('model.cbm')

# Считаем метрики на валидации
preds = model.predict(X_val)
r2 = r2_score(y_val, preds)
rmse = root_mean_squared_error(y_val, preds)

# Сохраняем стартовые метрики в файл (для сервера)
metrics = {
    "r2_score": r2,
    "rmse": rmse,
    "training_samples": len(X_train),
    "last_training_time": pd.Timestamp.now().isoformat()
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"Initial model saved. R2={r2:.4f}, RMSE={rmse:.4f}")