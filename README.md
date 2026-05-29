# ODS Project 2026 — Legal RAG for Russian Traffic Law

Проект для построения, запуска и экспериментального улучшения RAG-системы по юридическим документам, связанным с дорожным движением в РФ:

* Правила дорожного движения РФ;
* КоАП РФ, в первую очередь нормы об административной ответственности за нарушения ПДД;
* отдельные статьи УК РФ, связанные с ДТП, транспортными преступлениями и сопутствующими составами.

Основная цель проекта — сделать модульную инженерную реализацию Legal RAG, в которой можно отдельно менять и сравнивать компоненты retrieval/generation pipeline: модель эмбеддингов, стратегию чанкинга, FAISS/BM25-поиск, fusion, reranker, генератор ответа, промпты и формат цитирования.


---

## Что умеет проект

* FastAPI API для запуска RAG-пайплайна.
* Загрузка юридических документов в индекс.
* Chunking документов с сохранением metadata.
* Dense retrieval через FAISS.
* Sparse retrieval через BM25.
* Hybrid search с fusion-стратегиями:

  * Reciprocal Rank Fusion;
  * weighted z-score;
  * weighted sum.
* Reranking через Jina reranker v2.
* Генерация ответа через Mistral / OpenAI-compatible chat completions.
* Citation-aware ответы с маркерами источников вида `[C1]`, `[C2]`.
* Подготовка train/eval датасетов для fine-tuning эмбеддера.
* Ноутбуки для парсинга, генерации синтетики, обучения и экспериментов.
* Скрипты для расчёта retrieval-метрик инженерной реализации.

---

## Общая архитектура

```text
raw legal data
    ↓
parsing / normalization
    ↓
documents with metadata
    ↓
chunking
    ↓
embeddings
    ↓
FAISS index + BM25 index
    ↓
hybrid search
    ↓
score fusion
    ↓
reranking
    ↓
context with citations
    ↓
LLM generation
    ↓
answer + citations
```

Основная техническая реализация находится в `app/rag`.

---

## Структура репозитория

```text
.
├── app/
│   └── rag/                         # инженерная реализация RAG-сервиса
│       ├── api/                     # HTTP endpoints
│       ├── core/                    # конфигурация YAML + ENV
│       ├── models/                  # Pydantic-схемы API
│       ├── pipeline/                # сборка и выполнение RAG-пайплайна
│       ├── services/                # chunking, embeddings, FAISS, BM25, fusion, rerank, generation
│       ├── .env                     # локальные переменные окружения
│       ├── config.yaml              # основной конфиг RAG-сервиса
│       ├── Dockerfile               # Docker-образ API
│       ├── docker-compose.yml       # запуск через Docker Compose
│       ├── main.py                  # FastAPI application
│       ├── pyproject.toml           # настройки Python-проекта
│       ├── README.md                # техническое описание RAG API
│       └── requirements.txt         # зависимости API
│
├── data/
│   ├── datasets/                    # сырые и базовые юридические данные
│   ├── generation/                  # ноутбуки для генерации/обучения/экспериментов
│   ├── metricks/                    # скрипты и функции для расчёта метрик
│   ├── parsing/                     # загрузка данных в инженерное решение
│   └── train_eval_datasets/         # train/eval датасеты после разметки
│
├── test_requests.py                 # простой smoke-test запроса к API
└── README.md                        # описание проекта
```

---

## `app/rag` — техническая реализация

Папка `app/rag` содержит всю инженерную реализацию RAG API.

### Основные компоненты

| Путь                               | Назначение                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| `api/routes.py`                    | FastAPI endpoints: healthcheck, chunking, embeddings, ingest, search, generate, reset index. |
| `core/config.py`                   | Загрузка настроек из `config.yaml` и переменных окружения.                                   |
| `models/schemas.py`                | Pydantic DTO для запросов и ответов API.                                                     |
| `pipeline/factory.py`              | Сборка pipeline из конфигурации.                                                             |
| `pipeline/rag.py`                  | Основная логика `ingest → search → generate`.                                                |
| `services/chunking/`               | Разбиение документов на чанки.                                                               |
| `services/embeddings/`             | Обёртки над embedding-моделями, включая E5.                                                  |
| `services/vectorstores/`           | FAISS vector store.                                                                          |
| `services/fulltext/`               | BM25 full-text index.                                                                        |
| `services/fusion.py`               | Объединение dense/sparse результатов.                                                        |
| `services/rerankers/`              | Базовый интерфейс reranker, Jina v2 и noop-реализация.                                       |
| `services/generation/`             | Базовый интерфейс генератора, Mistral/OpenAI-compatible и noop-реализация.                   |
| `config.yaml`                      | Центральный конфиг моделей, top-k, fusion, reranker, LLM и prompt templates.                 |
| `main.py`                          | Точка входа FastAPI.                                                                         |
| `Dockerfile`, `docker-compose.yml` | Контейнеризация сервиса.                                                                     |
| `requirements.txt`                 | Python-зависимости.                                                                          |

### Ключевые настройки

В `app/rag/config.yaml` настраиваются:

* embedding model;
* device: `cuda` или `cpu`;
* параметры chunking;
* top-k для semantic/BM25/final retrieval;
* fusion method;
* reranker provider;
* LLM provider;
* prompt templates;
* директория для индексов.

Пример важных параметров:

```yaml
embedding:
  provider: "e5"
  model_name: "intfloat/multilingual-e5-large-instruct"
  device: "cuda"
  normalize: true

retrieval:
  semantic_top_k: 30
  bm25_top_k: 30
  final_top_k: 8
  fusion_method: "rrf"

reranker:
  provider: "jina_v2"
  model_name: "jinaai/jina-reranker-v2-base-multilingual"

llm:
  provider: "mistral"
  model: "mistral-large-latest"
```

Для локального запуска без GPU можно поставить:

```yaml
embedding:
  device: "cpu"

reranker:
  device: "cpu"
```

Для быстрого теста без LLM/reranker:

```yaml
reranker:
  provider: "noop"

llm:
  provider: "noop"
```

---

## `data/datasets` — сырые данные

Папка содержит исходные юридические данные, из которых собирается корпус для RAG.

| Файл            | Описание                                                                                                                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `koap.json`     | Структурированные статьи КоАП РФ, связанные с нарушениями ПДД и административной ответственностью. Используется как источник документов для индекса и как база для генерации train/eval примеров. |
| `pdd.json`      | Структурированное представление ПДД РФ: разделы, пункты, подпункты и тексты норм. Используется для загрузки ПДД в RAG и генерации вопросов по правилам дорожного движения.                        |
| `pdd_text.docx` | Исходный Word-документ с текстом ПДД. Используется как первичный источник для парсинга в `pdd.json`.                                                                                              |
| `uk_rf.json`    | Структурированные статьи УК РФ, связанные с ДТП, нарушениями ПДД и транспортными преступлениями. Используется для индексации и генерации train/eval данных.                                       |

---

## `data/generation` — ноутбуки генерации, обучения и экспериментов

В этой папке лежат offline-ноутбуки. Они не являются частью production API, но нужны для подготовки данных, обучения эмбеддера и проверки идей.

| Файл                                          | Описание                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RAG_for_law.ipynb`                           | Ноутбук для первичного формирования JSON из текстового документа с ПДД. Содержит парсинг `.docx`, преобразование ПДД в структурированный JSON, вывод статистики по разделам/пунктам, а также экспериментальный трёхуровневый FAISS-поиск по секциям, пунктам и подпунктам.                                  |
| `finetune_embedder_legal_rag.ipynb`           | Ноутбук для fine-tuning embedding-модели под legal RAG. Включает загрузку train JSONL, дедупликацию позитивов для борьбы с false negatives при `MultipleNegativesRankingLoss`, альтернативную стратегию `TripletLoss`, train/eval split по уникальным позитивам и evaluation через полный retrieval-корпус. |
| `synthetic_requests_generator_balanced.ipynb` | Ноутбук для генерации синтетических пользовательских запросов. Пайплайн включает построение таксономии реальных запросов, LLM-генерацию synthetic requests, LLM-валидацию, локальную фильтрацию дублей/PII/placeholders, repair невалидных примеров и балансировку сложности запросов.                      |

---

## `data/metrics` — расчёт метрик

Папка содержит код для оценки качества retrieval в инженерной реализации.


| Файл                | Описание                                                                                                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `metrics.py`       | Набор переиспользуемых функций для retrieval-метрик: `precision_at_k`, `recall_at_k`, `mrr_at_k`, `ndcg_at_k`. Также содержит извлечение `source_p_num` / `doc_id` из результатов поиска.                          |
| `count_metrics.py` | Скрипт для прогонки eval-набора через запущенный API `/v1/search`. Для каждого запроса сравнивает найденные источники с ожидаемым источником и печатает агрегированные `Recall@1`, `Recall@5`, `NDCG@1`, `NDCG@5`. |

Типичный сценарий:

```bash
# 1. Запустить API
cd app/rag
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. Загрузить данные в индекс
# см. data/parsing/parsing.py

# 3. Запустить подсчёт метрик из корня репозитория
python data/metricks/count_metricks.py
```

Перед запуском `count_metrics.py` проверь путь к eval-файлу внутри скрипта. Сейчас он может быть задан как локальный абсолютный путь разработчика, поэтому для переносимости лучше заменить его на относительный путь:

```python
data/train_evel_datasets/eval_with_source.json
```

---

## `data/parsing` — загрузка данных в инженерное решение

| Файл         | Описание                                                                                                                                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parsing.py` | Универсальный парсер и загрузчик документов в RAG API. Читает JSON-файлы с ПДД/КоАП/УК РФ, преобразует их в список документов с полями `id`, `text`, `metadata`, дедуплицирует документы и отправляет батчами в endpoint `/v1/ingest`. |

Что делает `parsing.py`:

1. Читает один JSON-файл или все JSON-файлы из папки.
2. Определяет тип структуры:

   * ПДД через ключ `section`;
   * КоАП/УК через ключ `articles`.
3. Собирает нормализованный текст документа.
4. Добавляет metadata:

   * `source_type`;
   * `source_p_num`;
   * `section_name`;
   * `article_num`;
   * `part_num`;
   * `source_file`;
   * и другие поля.
5. Отправляет документы в API батчами.

Пример настройки перед запуском:

```python
INPUT_PATH = Path("data/datasets")
INGEST_URL = "http://localhost:8000/v1/ingest"
RESET_INDEX = True
CHUNK = True
BATCH_SIZE = 100
```

Запуск из корня репозитория:

```bash
python data/parsing/parsing.py
```

---

## `data/train_evel_datasets` — датасеты после разметки

Папка содержит train/eval датасеты, полученные после парсинга, разметки, генерации запросов и очистки.

| Файл                        | Описание                                                                                                                                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_train_dataset.py`    | Скрипт сборки обучающего JSONL-датасета для fine-tuning эмбеддера. Генерирует пары/тройки вида `query`, `positive`, `negative` по ПДД, КоАП и УК.                                                                                |
| `eval_clean.jsonl`          | Очищенный eval-набор в формате JSONL. Каждая строка содержит `query`, `positive`, `negative`. Используется для оценки retrieval/fine-tuning без грязных или неконсистентных примеров.                                            |
| `eval_with_source.json`     | Eval-набор с ожидаемыми источниками. Используется инженерным скриптом метрик: запрос отправляется в `/v1/search`, а найденные `doc_id/source_p_num` сравниваются с эталонным `source`.                                           |
| `pdd_validation_set.jsonl`  | Валидационный набор по ПДД. Каждая строка содержит `query`, `expected_answer`, `source_p_num`; часть записей содержит `_old_p_num` для связи со старой нумерацией пунктов.                                                       |
| `train-2.jsonl`             | Расширенный обучающий JSONL-датасет для fine-tuning. Содержит триплеты `query`, `positive`, `negative`; используется в ноутбуке обучения как один из основных train-наборов.                                                     |
| `train_clean.jsonl`         | Очищенный train-набор в JSONL-формате. Используется для более стабильного обучения/сравнения после фильтрации и нормализации примеров.                                                                                           |
| `validation_set_koap.json`  | Валидационный набор по КоАП РФ: вопросы, ожидаемые ответы и/или привязки к релевантным статьям/частям. Нужен для проверки качества поиска и генерации по административной ответственности.                                       |
| `validation_set_uk_rf.json` | Валидационный набор по УК РФ: вопросы, ожидаемые ответы и/или привязки к релевантным статьям/частям. Нужен для проверки качества поиска и генерации по уголовно-правовым нормам, связанным с ДТП и транспортными преступлениями. |

---

## Форматы данных

### Документ для загрузки в RAG

```json
{
  "id": "pdd:12.14",
  "text": "ПДД РФ\n\nРаздел ...\n\nПункт 12.14\n\n...",
  "metadata": {
    "source_type": "pdd",
    "source_p_num": "12.14",
    "section_name": "Проезд пешеходных переходов...",
    "source_file": "data/datasets/pdd.json"
  }
}
```

### Train JSONL для эмбеддера

```json
{"query": "Какой штраф за езду без прав?", "positive": "...релевантный фрагмент...", "negative": "...похожий, но нерелевантный фрагмент..."}
```

### Validation JSONL для ПДД

```json
{"query": "Какие документы должен иметь при себе водитель?", "expected_answer": "...", "source_p_num": "1.1.1"}
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/inizioRUS/ods_project_2026.git
cd ods_project_2026
```

### 2. Создать окружение

```bash
cd app/rag

python -m venv .venv
source .venv/bin/activate
```

Для Windows:

```bash
.venv\Scripts\activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Настроить `.env`

Создай файл `app/rag/.env`.

Минимальный пример:

```env
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_BASE_URL=https://api.mistral.ai/v1/chat/completions
```

Если генерация через LLM не нужна, поставь в `config.yaml`:

```yaml
llm:
  provider: "noop"
```

### 5. Запустить API

Из папки `app/rag`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

Healthcheck:

```text
http://localhost:8000/v1/health
```

---

## Запуск через Docker

Из папки `app/rag`:

```bash
docker compose up --build
```

API будет доступен на порту `8000`.

---

## Основные endpoints

### Healthcheck

```bash
curl http://localhost:8000/v1/health
```

### Chunking

Разбивает тексты на чанки.

```bash
curl -X POST http://localhost:8000/v1/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Большой текст юридического документа..."],
    "ids": ["doc_1"],
    "metadata": [{"source": "manual"}],
    "config": {
      "chunk_size": 900,
      "chunk_overlap": 150
    }
  }'
```

### Embeddings

Строит embeddings для query или passage.

```bash
curl -X POST http://localhost:8000/v1/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Что будет за езду без прав?"],
    "input_type": "query"
  }'
```

### Ingest

Добавляет документы в FAISS и BM25 индексы.

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "reset": true,
    "chunk": true,
    "documents": [
      {
        "id": "pdd_1",
        "text": "1.1. Настоящие Правила дорожного движения устанавливают единый порядок дорожного движения...",
        "metadata": {
          "source": "pdd",
          "section": "1",
          "title": "Общие положения"
        }
      }
    ]
  }'
```

### Search

Выполняет hybrid search: FAISS + BM25 + fusion + rerank.

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Что будет за езду без прав?",
    "semantic_top_k": 30,
    "bm25_top_k": 30,
    "final_top_k": 8,
    "fusion_method": "rrf",
    "rerank": true
  }'
```

### Generate

Генерирует ответ на основе найденных чанков и возвращает citations.

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Что будет, если ехать без прав?",
    "final_top_k": 6,
    "rerank": true,
    "max_context_chars": 8000
  }'
```

### Reset index

Очищает FAISS и BM25 индексы.

```bash
curl -X POST http://localhost:8000/v1/index/reset
```

---

## Полный сценарий работы

### 1. Запустить API

```bash
cd app/rag
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Подготовить путь к данным

В `data/parsing/parsing.py` укажи:

```python
INPUT_PATH = Path("data/datasets")
RESET_INDEX = True
CHUNK = True
```

### 3. Загрузить документы в индекс

Из корня репозитория:

```bash
python data/parsing/parsing.py
```

### 4. Проверить поиск

```bash
python test_requests.py
```

Или напрямую:

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Что считается крупным ущербом?"}'
```

### 5. Посчитать метрики

```bash
python data/metricks/count_metricks.py
```

---

## Fine-tuning эмбеддера

Для обучения используется формат:

```json
{"query": "...", "positive": "...", "negative": "..."}
```

Основной ноутбук:

```text
data/generation/finetune_embedder_legal_rag.ipynb
```

Основные идеи обучения:

* использовать E5/BGE/Qwen-like embedding модели;
* добавлять префиксы `query:` и `passage:` для E5;
* бороться с false negatives при `MultipleNegativesRankingLoss`;
* дедуплицировать позитивы;
* разделять train/eval по уникальным позитивным документам;
* оценивать retrieval на полном корпусе, а не только на маленьком eval subset;
* сравнивать baseline и fine-tuned embedder.

---

## Метрики качества

Для оценки retrieval используются:

* `Precision@k`;
* `Recall@k`;
* `MRR@k`;
* `nDCG@k`.

Инженерный скрипт `data/metricks/count_metricks.py` сейчас печатает:

* `Recall@1`;
* `Recall@5`;
* `NDCG@1`;
* `NDCG@5`.

---

## License

MIT License
