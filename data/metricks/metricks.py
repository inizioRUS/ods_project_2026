import math
import requests
from typing import Any, Iterable


def _as_dict(obj: Any) -> dict:
    """Поддержка dict и pydantic-моделей."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return vars(obj)


def _normalize_sources(value: Any) -> set[str]:
    """
    source_p_num может быть строкой, списком или set.
    Приводим всё к set[str].
    """
    if value is None:
        return set()

    if isinstance(value, str):
        return {value.strip()}

    if isinstance(value, Iterable):
        result = set()
        for item in value:
            if item is not None:
                result.add(str(item).strip())
        return result

    return {str(value).strip()}


def extract_source_ids(result: Any) -> set[str]:
    """
    Достаём source_p_num из результата поиска.

    Ожидаемый вариант:
        result["metadata"]["source_p_num"]

    Дополнительно проверяются doc_id/id как fallback.
    """
    r = _as_dict(result)
    metadata = r.get("metadata") or {}

    source_ids = set()

    possible_metadata_keys = [
        "source_p_num",
        "p_num",
        "paragraph",
        "article",
        "source",
        "source_id",
    ]

    for key in possible_metadata_keys:
        source_ids |= _normalize_sources(metadata.get(key))

    # fallback, если source_p_num хранится не в metadata
    source_ids |= _normalize_sources(r.get("source_p_num"))
    source_ids |= _normalize_sources(r.get("doc_id"))
    source_ids |= _normalize_sources(r.get("id"))

    return source_ids


def relevance_vector(results: list[Any], expected_sources: Any, k: int) -> list[int]:
    """
    Возвращает список 0/1 длины k.
    1 — результат релевантный, 0 — нерелевантный.
    """
    expected = _normalize_sources(expected_sources)

    rels = []
    for result in results[:k]:
        found_sources = extract_source_ids(result)
        rels.append(1 if expected & found_sources else 0)

    # если API вернул меньше k результатов, считаем недостающие как нерелевантные
    while len(rels) < k:
        rels.append(0)

    return rels


def precision_at_k(results: list[Any], expected_sources: Any, k: int = 10) -> float:
    """
    Precision@k = сколько релевантных результатов в топ-k / k
    """
    rels = relevance_vector(results, expected_sources, k)
    return sum(rels) / k


def recall_at_k(results: list[Any], expected_sources: Any, k: int = 10) -> float:
    """
    Recall@k = сколько найдено релевантных источников / сколько всего ожидалось.

    В твоём датасете обычно один expected source_p_num,
    поэтому Recall@k будет 1.0, если источник найден в top-k, иначе 0.0.
    """
    expected = _normalize_sources(expected_sources)

    if not expected:
        return 0.0

    found = set()
    for result in results[:k]:
        found |= extract_source_ids(result)

    return len(expected & found) / len(expected)


def mrr_at_k(results: list[Any], expected_sources: Any, k: int = 10) -> float:
    """
    MRR@k = 1 / rank первого релевантного результата.
    Если релевантного результата нет, возвращает 0.
    """
    rels = relevance_vector(results, expected_sources, k)

    for index, rel in enumerate(rels, start=1):
        if rel == 1:
            return 1 / index

    return 0.0


def ndcg_at_k(results: list[Any], expected_sources: Any, k: int = 10) -> float:
    """
    nDCG@k для binary relevance.

    DCG = sum(rel_i / log2(i + 1))
    где i — позиция, начиная с 1.

    nDCG = DCG / IDCG
    """
    expected = _normalize_sources(expected_sources)

    if not expected:
        return 0.0

    rels = relevance_vector(results, expected, k)

    def dcg(values: list[int]) -> float:
        return sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(values, start=1)
        )

    dcg_score = dcg(rels)

    ideal_rels = [1] * min(len(expected), k)
    while len(ideal_rels) < k:
        ideal_rels.append(0)

    idcg_score = dcg(ideal_rels)

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score
