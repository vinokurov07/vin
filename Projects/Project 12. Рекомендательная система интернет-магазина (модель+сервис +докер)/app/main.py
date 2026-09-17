"""Flask-приложение: HTTP API для рекомендательной модели."""
import logging
import time

import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
)

from app.model import RecommenderModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Prometheus метрики ---
REQUESTS_TOTAL = Counter(
    'requests_total', 'Total requests', ['endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'request_duration_seconds', 'Request duration', ['endpoint']
)
PREDICTIONS_TOTAL = Counter(
    'predictions_total', 'Total predictions'
)

# --- Загрузка модели при старте ---
logger.info("Initializing model...")
model = RecommenderModel()
logger.info("Model loaded successfully")


# ---------- Endpoints ----------

@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности сервиса."""
    return jsonify({"status": "ok"})


@app.route('/metrics', methods=['GET'])
def metrics():
    """Метрики Prometheus."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Принимает JSON:
    {
        "visitorid": 123,
        "candidates": [
            {"itemid": 101, "categoryid": 5, "hour": 20, ...},
            ...
        ]
    }
    Возвращает:
    {"recommendations": [101, 202, 303], "model_version": "1.0.0"}
    """
    start_time = time.time()
    REQUESTS_TOTAL.labels(endpoint='/recommend', status='started').inc()

    try:
        # --- Валидация ---
        data = request.get_json(silent=True)
        if not data:
            REQUESTS_TOTAL.labels(endpoint='/recommend', status='400').inc()
            return jsonify({"error": "Request body must be valid JSON"}), 400

        candidates = data.get('candidates')
        if not isinstance(candidates, list):
            REQUESTS_TOTAL.labels(endpoint='/recommend', status='400').inc()
            return jsonify({"error": "'candidates' must be a list"}), 400

        if len(candidates) == 0:
            REQUESTS_TOTAL.labels(endpoint='/recommend', status='400').inc()
            return jsonify({"error": "'candidates' must not be empty"}), 400

        if len(candidates) > 1000:
            REQUESTS_TOTAL.labels(endpoint='/recommend', status='400').inc()
            return jsonify({"error": "'candidates' must not exceed 1000 items"}), 400

        if not all(isinstance(c, dict) for c in candidates):
            REQUESTS_TOTAL.labels(endpoint='/recommend', status='400').inc()
            return jsonify({"error": "Each candidate must be an object"}), 400

        if not all('itemid' in c for c in candidates):
            REQUESTS_TOTAL.labels(endpoint='/recommend', status='400').inc()
            return jsonify({"error": "Each candidate must have 'itemid'"}), 400

        # --- Инференс ---
        candidates_df = pd.DataFrame(candidates)

        recommendations = model.recommend(candidates_df, k=3)
        PREDICTIONS_TOTAL.inc()

        elapsed = time.time() - start_time
        REQUEST_DURATION.labels(endpoint='/recommend').observe(elapsed)
        REQUESTS_TOTAL.labels(endpoint='/recommend', status='200').inc()

        return jsonify({
            "recommendations": recommendations,
            "model_version": "1.0.0"
        }), 200

    except Exception:
        logger.exception("Error in /recommend")
        REQUESTS_TOTAL.labels(endpoint='/recommend', status='500').inc()
        return jsonify({"error": "internal error"}), 500


if __name__ == '__main__':
    # Локальный запуск (для разработки). В проде — gunicorn.
    app.run(host='0.0.0.0', port=8000, debug=False)