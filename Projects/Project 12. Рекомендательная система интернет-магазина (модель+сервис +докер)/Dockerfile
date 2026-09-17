FROM python:3.10-slim

# Рабочая директория
WORKDIR /app

# Копируем и ставим зависимости (отдельным слоем — кэшируется)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и артефакты
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Открываем порт
EXPOSE 8000

# Запускаем через gunicorn (production-ready WSGI-сервер)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app.main:app"]