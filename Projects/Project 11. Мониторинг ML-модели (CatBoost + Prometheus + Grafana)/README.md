#  Мониторинг ML-модели (CatBoost + Prometheus + Grafana)

Проект демонстрирует полный цикл мониторинга модели машинного обучения с возможностью **дообучения**.  
В качестве задачи выбрана регрессия качества красного вина (`quality`) по физико-химическим признакам.

## Используемые технологии

- **Python 3.10**, **FastAPI**, **CatBoost**, **scikit-learn**, **prometheus_client**
- **Docker** (для запуска Prometheus и Grafana)
- **Prometheus** – сбор и хранение метрик
- **Grafana** – визуализация метрик на дашборде


Все компоненты запускаются на локальной машине (Windows).  
Prometheus и Grafana работают в Docker-контейнерах и общаются с хостом через `host.docker.internal`.

## Состав проекта

| Файл | Назначение |
|------|------------|
| `winequality-red.csv` | Исходный датасет (1599 записей, 11 признаков + target `quality`) |
| `train_initial.py` | Первоначальное обучение модели. Создаёт `model.cbm` и `metrics.json` |
| `model_server.py` | FastAPI-сервер, отдающий метрики для Prometheus и принимающий запросы на дообучение |
| `additional_training.py` | Скрипт для отправки новых данных и запуска дообучения через API |
| `prometheus.yml` | Конфигурация Prometheus для сбора метрик с `model_server` |
| `model.cbm` | Сохранённая модель CatBoost (появляется после `train_initial.py`) |
| `metrics.json` | Файл с текущими метриками модели (R², RMSE, количество сэмплов, время обучения) |

## Быстрый старт

### 1. Установите зависимости
```bash
pip install fastapi uvicorn catboost scikit-learn prometheus_client pandas requests
```

### 2. Обучите начальную модель
```bash
python train_initial.py
```
Появятся файлы model.cbm и metrics.json.

### 3. Запустите сервер модели
```bash
python model_server.py
```
Сервер будет доступен на http://localhost:8000. Метрики отдаются по адресу /metrics.

### 4. Запустите Prometheus (Docker)
Во втором терминале, из папки с prometheus.yml:
```bash
docker run -p 9090:9090 -v ${PWD}\prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```
### 5. Запустите Grafana (Docker)
В третьем терминале:
```bash
docker run -d -p 3000:3000 --name=grafana grafana/grafana
```
Откройте http://localhost:3000 (admin / admin).
Добавьте Data Source Prometheus с URL: http://host.docker.internal:9090 → Save & Test.
Создайте дашборд с метриками: model_r2_score, model_rmse, model_training_samples_total, model_train_duration_seconds.

### 6. Проверьте дообучение
```bash
python additional_training.py
```
Он отправит 20 новых записей на POST /train.
Метрики обновятся, и Grafana покажет изменения.

## Полный сервис в работе
После успешного запуска всех компонентов система функционирует следующим образом:

- Модель (CatBoostRegressor) обучена на wine-датасете, сохранена в model.cbm, начальные метрики записаны в metrics.json.

- Сервер модели на FastAPI отдаёт Prometheus-метрики через GET /metrics и принимает новые данные для дообучения через POST /train.

- Prometheus в Docker-контейнере собирает метрики с хоста (host.docker.internal:8000) и сохраняет временные ряды.

- Grafana в Docker-контейнере подключена к Prometheus, на дашборде отображаются ключевые метрики: R², RMSE, количество обучающих примеров и длительность обучения.

- Модель можно дообучать — при отправке новых данных через POST-запрос к /train модель доучивается, метрики обновляются, и Grafana показывает изменения.

## Как работает дообучение
- Скрипт additional_training.py отправляет JSON с новыми объектами (признаки + quality) на http://localhost:8000/train.

- Сервер (model_server.py) принимает данные, разбивает их на train/val, дообучает текущую модель:
    - Определяет число уже имеющихся деревьев model.tree_count_.
    - Создаёт новый CatBoost-объект с увеличенным числом итераций (текущее + 100).
    - Вызывает fit(..., init_model=model), продолжая обучение с предыдущих деревьев.

- После обучения вычисляются новые метрики R² и RMSE на валидации, обновляются счётчики Prometheus.

- Модель и метрики сохраняются в model.cbm и metrics.json.

## Вывод
Проект демонстрирует:

 - Создание регрессионной модели.  
 - Экспорт метрик в формате Prometheus.  
 - Подключение Prometheus и Grafana с отображением ключевых показателей.  
 - Возможность дообучить модель через API без остановки сервиса.  

