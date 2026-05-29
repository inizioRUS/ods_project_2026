# ODS Project 2026 — Legal RAG for Russian Traffic Law

Проект для построения и экспериментального улучшения RAG-системы по юридическим документам, связанным с ПДД РФ: Правила дорожного движения, КоАП РФ и отдельные статьи УК РФ.

Основная идея проекта — сделать модульный RAG-пайплайн, в котором можно отдельно менять и сравнивать:

* модель эмбеддингов;
* стратегию чанкинга;
* векторный поиск;
* BM25-поиск;
* fusion-алгоритм;
* reranker;
* генератор ответа;
* формат промпта и цитирования.

Проект подходит для экспериментов с retrieval, reranking, fine-tuning эмбеддеров и построением citation-aware юридического ассистента.

---

## Возможности

* FastAPI API для RAG-пайплайна.
* Гибридный retrieval:

  * dense retrieval через FAISS;
  * sparse retrieval через BM25.
* Поддержка fusion-стратегий:

  * Reciprocal Rank Fusion;
  * weighted z-score;
  * weighted sum.
* Reranking через Jina reranker v2.
* Генерация ответа через Mistral / OpenAI-compatible chat completions.
* Ответы с цитированием источников в формате `[C1]`, `[C2]`.
* Подготовка train dataset для fine-tuning эмбеддера.
* Validation sets для проверки качества ответов.
* Возможность быстро отключать LLM/reranker через `noop`-провайдеры.

---

## Структура проекта

```text
.
├── app/
│   └── rag/
│       ├── api/
│       │   └── routes.py              # HTTP endpoints
│       ├── core/
│       │   └── config.py              # YAML + ENV конфигурация
│       ├── models/
│       │   └── schemas.py             # Pydantic-схемы запросов и ответов
│       ├── pipeline/
│       │   ├── factory.py             # сборка RAG-пайплайна
│       │   └── rag.py                 # основной пайплайн ingest/search/generate
│       ├── services/
│       │   ├── chunking/              # разбиение документов на чанки
│       │   ├── embeddings/            # E5 embedder
│       │   ├── fulltext/              # BM25 index
│       │   ├── generation/            # LLM generation
│       │   ├── rerankers/             # reranker interface + implementations
│       │   ├── vectorstores/          # FAISS vector store
│       │   └── fusion.py              # объединение dense/sparse результатов
│       ├── config.yaml                # основной конфиг
│       ├── Dockerfile
│       ├── docker-compose.yml
│       ├── main.py                    # FastAPI app
│       └── requirements.txt
│
└── data/
    ├── files/                         # исходные юридические документы
    ├── build_train_dataset.py         # генерация train.jsonl
    ├── train.jsonl                    # train dataset для эмбеддера
    ├── koap.json                      # КоАП РФ
    ├── uk_rf.json                     # статьи УК РФ
    ├── validation_set_koap.json       # validation set по КоАП
    ├── validation_set_uk_rf.json      # validation set по УК РФ
    ├── RAG_for_law.ipynb
    ├── finetune_embedder_legal_rag.ipynb
    └── work_with_data.ipynb
```

---

## Архитектура RAG-пайплайна

```text
documents
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
context building with citations
   ↓
LLM generation
   ↓
answer + citations
```

Основной pipeline реализован в:

```text
app/rag/pipeline/rag.py
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/inizioRUS/ods_project_2026.git
cd ods_project_2026/app/rag
```

### 2. Создать виртуальное окружение

```bash
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

### 4. Настроить переменные окружения

Создай файл `.env` в папке `app/rag`.

Минимальный пример:

```env
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_BASE_URL=https://api.mistral.ai/v1/chat/completions
```

Если генерация через LLM пока не нужна, можно отключить её в `config.yaml`:

```yaml
llm:
  provider: "noop"
```

Если нет GPU, поменяй `device` с `cuda` на `cpu`:

```yaml
embedding:
  device: "cpu"

reranker:
  device: "cpu"
```

### 5. Запустить API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

После запуска Swagger UI будет доступен по адресу:

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

Пример ответа:

```json
{
  "ok": true,
  "index_size": 0,
  "embedding_model": "intfloat/multilingual-e5-large-instruct",
  "reranker": "jina_v2",
  "llm": "mistral"
}
```

---

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

---

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

---

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

---

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

---

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

---

### Reset index

Очищает FAISS и BM25 индексы.

```bash
curl -X POST http://localhost:8000/v1/index/reset
```

---

## Конфигурация

Основной конфиг находится здесь:

```text
app/rag/config.yaml
```

В нём настраиваются:

* название и версия приложения;
* путь к индексам;
* embedding model;
* параметры чанкинга;
* top-k для retrieval;
* fusion method;
* reranker;
* LLM provider;
* prompt templates.

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

---

## Данные

В папке `data` лежат исходные юридические данные и файлы для экспериментов:

* `koap.json` — статьи КоАП РФ, связанные с нарушениями ПДД;
* `uk_rf.json` — статьи УК РФ, связанные с ДТП и транспортными преступлениями;
* `files/pdd.json` — Правила дорожного движения РФ;
* `train.jsonl` — triplet dataset для обучения/дообучения эмбеддера;
* `validation_set_koap.json` — валидационный набор по КоАП;
* `validation_set_uk_rf.json` — валидационный набор по УК РФ.

---

## Генерация train dataset

Скрипт:

```text
data/build_train_dataset.py
```

генерирует пары/тройки вида:

```json
{
  "query": "...",
  "positive": "...",
  "negative": "..."
}
```

Запуск из корня репозитория:

```bash
python data/build_train_dataset.py
```

Можно переопределить параметры через переменные окружения:

```bash
OUT_PATH=data/train.jsonl \
KEEP_SEED=1 \
MIN_LEN=60 \
SEED=42 \
python data/build_train_dataset.py
```

---

## Fine-tuning и эксперименты

В проекте есть ноутбуки для работы с данными и обучения:

```text
data/work_with_data.ipynb
data/RAG_for_law.ipynb
data/finetune_embedder_legal_rag.ipynb
```

---

## Идеи для дальнейшего развития

* Добавить offline evaluation pipeline.
* Добавить автоматический расчёт Recall@k, MRR@k, nDCG@k.
* Сравнить несколько embedding-моделей:

  * `intfloat/multilingual-e5-large-instruct`;
  * `BAAI/bge-m3`;
  * Jina embeddings;
  * fine-tuned domain-specific embedder.
* Добавить query rewriting.
* Добавить LLM-based query classification.
* Добавить Learning-to-Rank слой поверх retrieval features.
* Добавить metadata-aware scoring по разделам ПДД, статьям КоАП и УК РФ.
* Сделать frontend/demo-интерфейс для юридического ассистента.
* Добавить CI для lint/test.
* Добавить Docker GPU profile.

---

## Ограничения

* Это исследовательский проект, а не production-ready юридическая система.
* Ответы модели не являются юридической консультацией.
* Качество ответа зависит от качества индекса, чанкинга, retrieval и reranking.
* Для production-сценариев нужны:

  * актуализация правовых данных;
  * контроль версий документов;
  * evaluation на большом validation set;
  * логирование запросов;
  * мониторинг hallucinations;
  * human review для юридически значимых ответов.

---

## Disclaimer

Проект предназначен для образовательных и исследовательских целей.
Сгенерированные ответы не должны рассматриваться как официальная юридическая консультация.

---

## License

```text
MIT License
```
