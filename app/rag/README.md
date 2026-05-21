# Basic Modular RAG API

FastAPI-сервис для базового, но расширяемого RAG-пайплайна:

- embeddings: E5-large / multilingual-e5-large-instruct;
- vector DB: FAISS `IndexFlatIP` с нормализованными векторами;
- full-text retrieval: BM25;
- score fusion: RRF, weighted z-score, weighted sum;
- rerank: Jina reranker v2;
- generation: Mistral/OpenAI-compatible chat completions;
- citation-aware prompt: ответ генерируется только по выбранным чанкам и с маркерами `[C1]`, `[C2]`.

Архитектура специально сделана так, чтобы менять отдельные блоки: embedding, chunking, retriever, fusion, reranker, LLM, prompts.

## Структура

```text
app/
  api/routes.py                  # HTTP endpoints
  core/config.py                 # YAML + ENV config
  models/schemas.py              # Pydantic DTO
  pipeline/rag.py                # общий пайплайн
  services/
    chunking/recursive.py
    embeddings/e5.py
    vectorstores/faiss_store.py
    fulltext/bm25_store.py
    fusion.py
    rerankers/{base,jina_v2,noop,factory}.py
    generation/{base,mistral,noop,factory}.py
config.yaml
requirements.txt
```

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env .env
# заполни MISTRAL_API_KEY, если нужна генерация
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger будет доступен в `/docs`.

## Основные endpoints

### Health

```bash
curl http://localhost:8000/v1/health
```

### Chunking: принимает список текстов

```bash
curl -X POST http://localhost:8000/v1/chunk \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": ["Большой текст документа..."],
    "ids": ["doc_1"],
    "metadata": [{"source": "manual"}],
    "config": {"chunk_size": 900, "chunk_overlap": 150}
  }'
```

### Embeddings: принимает список текстов

```bash
curl -X POST http://localhost:8000/v1/embed \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": ["Что делать при ДТП?", "Пункт ПДД про остановку"],
    "input_type": "query"
  }'
```

### Ingest: чанкинг + embeddings + FAISS + BM25

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "reset": true,
    "chunk": true,
    "documents": [
      {
        "id": "pdd_1",
        "text": "1.1. Настоящие Правила дорожного движения устанавливают...",
        "metadata": {"section": "1", "title": "Общие положения"}
      }
    ]
  }'
```

### Hybrid search: FAISS + BM25 + fusion + Jina rerank

```bash
curl -X POST http://localhost:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "какие правила остановки перед пешеходным переходом?",
    "semantic_top_k": 30,
    "bm25_top_k": 30,
    "final_top_k": 8,
    "fusion_method": "rrf",
    "rerank": true
  }'
```

### Generate: RAG answer + citations

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Когда водитель должен уступить дорогу пешеходу?",
    "final_top_k": 6,
    "rerank": true,
    "max_context_chars": 8000
  }'
```

## Как модернизировать под эксперименты

### 1. Дообучение эмбеддера / реранкера

Добавь отдельный offline-модуль, который собирает пары/тройки:

```text
(query, positive_chunk, negative_chunk)
(query, relevant_chunk_ids)
```

Затем замеряй retrieval-метрики до и после:

- semantic stage: Recall@k, MRR@k, nDCG@k;
- hybrid stage: Recall@k, nDCG@k;
- rerank stage: MRR@k, nDCG@k, Precision@k.

В сервисе менять ничего не нужно: достаточно заменить `embedding.model_name` или `reranker.model_name` в `config.yaml` на путь к fine-tuned модели.

### 2. Сравнение эмбеддеров / реранкеров

Сделай несколько конфигов:

```text
configs/e5_large.yaml
configs/bge_m3.yaml
configs/jina_embeddings.yaml
configs/jina_reranker.yaml
configs/bge_reranker.yaml
```

Запускать можно так:

```bash
RAG_CONFIG_PATH=configs/e5_large.yaml uvicorn app.main:app --port 8001
RAG_CONFIG_PATH=configs/bge_m3.yaml uvicorn app.main:app --port 8002
```

### 3. Замена routing на LLM-классификацию / query rewriting

Добавь сервисы:

```text
services/query_transform/base.py
services/query_transform/llm_classifier.py
services/query_transform/query_rewriter.py
```

И вставь их перед retrieval:

```text
raw_query -> route/classify -> rewrite/subqueries -> retrieve -> fusion -> rerank -> generate
```

Метрики: сравнивай baseline query vs rewritten subqueries по Recall@k и nDCG@k.

### 4. CatBoost / Learning-to-Rank слой

Сейчас `fusion.py` объединяет semantic + BM25 + rerank. Следующий шаг — LTR-модель:

```text
features = [semantic_score, bm25_score, rerank_score, chunk_len, section_match, query_len, exact_terms]
score = catboost_ranker.predict(features)
```

Добавь новый модуль:

```text
services/ltr/catboost_ranker.py
```

И используй его после rerank или вместо fusion.

### 5. Зачем убирать/оставлять логику “схожесть по разделу ПДД + схожесть по пункту ПДД”

В базовой версии я убрал жесткую доменную логику. Вместо этого `metadata` остается в чанке:

```json
{"section": "13", "point": "13.1", "title": "Проезд перекрестков"}
```

Если понадобится, можно добавить metadata-aware score:

```text
final = alpha * chunk_score + beta * section_score + gamma * rerank_score
```

Но это нужно обосновывать экспериментом: section-score полезен, если запросы часто широкие и сначала важно найти раздел, а потом пункт. Если запросы конкретные, такой score может ухудшать ranking.

### 6. Что векторизировать

Для ПДД часто лучше векторизировать не только сырой пункт, а enriched chunk:

```text
Раздел: Проезд перекрестков
Пункт: 13.1
Текст: ...
Ключевые термины: уступить дорогу, перекресток, главная дорога
```

При этом в генерацию можно отдавать чистый текст + metadata, чтобы не засорять ответ.

## Что уже заложено для экспериментов

- `config.yaml` управляет моделями, top_k, fusion, prompts;
- retrieval и rerank разделены;
- BM25 и FAISS независимы;
- fusion можно менять без переписывания API;
- генератор заменяется через `BaseGenerator`;
- реранкер заменяется через `BaseReranker`;
- chunker заменяется через отдельный сервис.

## Важное про первый запуск

Первый вызов `/embed`, `/ingest`, `/search` загрузит E5-модель. Первый `/search` с `rerank=true` загрузит Jina reranker. На CPU это может быть тяжело; для локальных тестов можно временно поставить:

```yaml
reranker:
  provider: "noop"
llm:
  provider: "noop"
```

А потом включить Jina/Mistral обратно.
