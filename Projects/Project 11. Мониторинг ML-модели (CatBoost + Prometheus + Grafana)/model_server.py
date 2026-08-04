# model_server.py
import os
import json
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
from prometheus_client import Gauge, Counter, generate_latest, REGISTRY
from starlette.responses import Response
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import threading

app = FastAPI()

# --- Prometheus метрики ---
r2_gauge = Gauge('model_r2_score', 'R2 score on validation set')
rmse_gauge = Gauge('model_rmse', 'RMSE on validation set')
samples_total = Counter('model_training_samples_total', 'Total number of samples used for training')
last_train_time = Gauge('model_last_train_timestamp', 'Timestamp of last model training')
train_duration = Gauge('model_train_duration_seconds', 'Duration of last training in seconds')

# --- Загрузка модели и начальных метрик ---
if not os.path.exists('model.cbm'):
    raise FileNotFoundError("model.cbm not found. Run train_initial.py first.")
model = CatBoostRegressor()
model.load_model('model.cbm')

with open('metrics.json', 'r') as f:
    m = json.load(f)
    r2_gauge.set(m['r2_score'])
    rmse_gauge.set(m['rmse'])
    samples_total.inc(m['training_samples'])
    last_train_time.set(pd.Timestamp(m['last_training_time']).timestamp())

class TrainData(BaseModel):
    data: list[dict]

train_lock = threading.Lock()

@app.post("/train")
async def train_endpoint(train_data: TrainData):
    global model
    if not train_lock.acquire(blocking=False):
        raise HTTPException(status_code=423, detail="Training already in progress")
    try:
        df = pd.DataFrame(train_data.data)
        if 'quality' not in df.columns:
            raise HTTPException(status_code=422, detail="Column 'quality' missing")
        X = df.drop('quality', axis=1)
        y = df['quality']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        t_start = time.time()

        # --- Дообучение ---
        current_trees = model.tree_count_
        additional_trees = 100
        total_trees = current_trees + additional_trees

        params = model.get_params()
        params['iterations'] = total_trees
        params.pop('verbose', None)   # убираем verbose, если есть

        new_model = CatBoostRegressor(**params)
        new_model.fit(X_train, y_train,
                      eval_set=(X_val, y_val),
                      init_model=model,
                      verbose=False)
        model = new_model

        t_end = time.time()
        duration = t_end - t_start

        # Оценка качества
        preds = model.predict(X_val)
        new_r2 = r2_score(y_val, preds)
        new_rmse = root_mean_squared_error(y_val, preds)

        # Обновляем метрики
        r2_gauge.set(new_r2)
        rmse_gauge.set(new_rmse)
        samples_total.inc(len(X_train))
        last_train_time.set(time.time())
        train_duration.set(duration)

        # Сохраняем модель и метрики
        model.save_model('model.cbm')
        with open('metrics.json', 'w') as f:
            json.dump({
                "r2_score": new_r2,
                "rmse": new_rmse,
                "training_samples": int(samples_total._value.get()),
                "last_training_time": pd.Timestamp.now().isoformat()
            }, f)

        return {"message": "Model retrained", "new_r2": new_r2, "new_rmse": new_rmse}
    finally:
        train_lock.release()

@app.get("/predict")
async def predict_endpoint(features: str):
    try:
        vals = [float(x) for x in features.split(',')]
        if len(vals) != 11:
            raise ValueError
        pred = model.predict([vals])[0]
        return {"prediction": pred}
    except:
        raise HTTPException(status_code=400, detail="Features must be 11 comma-separated numbers")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)