# Загрузка исходных данных

Проект использует публичный датасет **Retailrocket recommender system dataset**.

## Альтернатива: скачать только готовый Docker-образ

Если не хочется скачивать 1 ГБ данных и обучать модель заново — используйте готовый образ из [Releases](https://github.com/vinokurov07/vin/releases/download/v1.0.0-project12/recommender.tar). Он уже содержит все артефакты. Инструкция в основном README.

## Ссылка на датасет

**Kaggle:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

## Описание

Датасет собран с реального интернет-магазина за 4.5 месяца (**май — сентябрь 2015**).
Содержит неявную обратную связь пользователей: просмотры, добавления в корзину, покупки.

### Характеристики

| Показатель | Значение |
|-----------|----------|
| Всего событий | 2 756 101 |
| Уникальных пользователей | 1 407 580 |
| Уникальных товаров | 235 061 (в событиях), 417 053 (в свойствах) |
| Категорий | 1 669 |
| Период | 03.05.2015 — 18.09.2015 |

**Распределение событий:**
- `view` — 2 664 312 (96.7%)
- `addtocart` — 69 332 (2.5%)
- `transaction` — 22 457 (0.8%)

## Что нужно скачать

Все 4 файла:

| Файл | Размер | Описание |
|------|--------|----------|
| `events.csv` | ~90 МБ | Логи событий: timestamp, visitorid, event, itemid, transactionid |
| `item_properties_part1.csv` | ~475 МБ | Свойства товаров, часть 1: timestamp, itemid, property, value |
| `item_properties_part2.csv` | ~475 МБ | Свойства товаров, часть 2 (продолжение) |
| `category_tree.csv` | ~14 КБ | Иерархия категорий: categoryid, parentid |

**Итого:** ~1 ГБ.

## Куда положить

Создать папку `data/` в корне проекта и положить туда все 4 файла:

```
recommender-system/
└── data/
    ├── events.csv
    ├── item_properties_part1.csv
    ├── item_properties_part2.csv
    └── category_tree.csv
```

Папка `data/` добавлена в `.gitignore` — она не попадёт в git-репозиторий.

## Как скачать

### Вариант 1: через веб-интерфейс Kaggle

1. Зарегистрироваться на [kaggle.com](https://www.kaggle.com) (бесплатно).
2. Открыть страницу датасета: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
3. Нажать **Download** → скачается zip-архив `archive.zip` (~50 МБ сжатый).
4. Распаковать архив — внутри будет папка `events.csv` и другие файлы.
5. Переместить все `.csv` в папку `data/` проекта.


```

## Как запустить пайплайн после скачивания

1. Открыть `notebooks/recommender_pipeline.ipynb` в Jupyter:
   ```bash
   jupyter notebook notebooks/recommender_pipeline.ipynb
   ```
2. Выполнить ячейки последовательно (Kernel → Restart & Run All).
3. Ноутбук построит все признаки, обучит модель и сохранит артефакты:
   - `models/catboost_best.cbm` — обученная CatBoost-модель
   - `models/feature_columns.json` — порядок колонок для инференса
   - `models/cat_features.json` — список категориальных признаков
   - `data/item_popularity.parquet` — snapshot популярности для retrieval

После этого можно собирать Docker-образ (см. основной README).


