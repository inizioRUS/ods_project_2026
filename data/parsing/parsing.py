import json
import re
from pathlib import Path
from typing import Any

import requests


# ============================================================
# НАСТРОЙКИ
# ============================================================

# INPUT_PATH = Path("./data")
# Можно указать:
# INPUT_PATH = Path("./data/pdd.json")
# INPUT_PATH = Path("./data/koap.json")
INPUT_PATH = Path("data\\uk_rf.json")
# INPUT_PATH = Path("./data")

INGEST_URL = "http://localhost:8000/v1/ingest"

RESET_INDEX = True
CHUNK = False
BATCH_SIZE = 100

DRY_RUN = False

# Если в твоём DocumentInput поле называется doc_id,
# поменяй ID_FIELD = "doc_id"
ID_FIELD = "id"
TEXT_FIELD = "text"
METADATA_FIELD = "metadata"


# ============================================================
# UTILS
# ============================================================

def safe_id(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-zа-я0-9_.:-]+", "_", value)
    return value.strip("_")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    return sorted(path.glob("*.json"))


def make_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        ID_FIELD: doc_id,
        TEXT_FIELD: text.strip(),
        METADATA_FIELD: metadata,
    }


def format_sanctions(sanctions: list[dict[str, Any]] | None) -> str:
    if not sanctions:
        return ""

    lines = []

    for sanction in sanctions:
        parts = []

        if sanction.get("type"):
            parts.append(str(sanction["type"]))

        if sanction.get("amount"):
            parts.append(str(sanction["amount"]))

        if sanction.get("duration"):
            parts.append(str(sanction["duration"]))

        if sanction.get("unit"):
            parts.append(str(sanction["unit"]))

        line = " ".join(parts)

        if sanction.get("additional"):
            line += f"; дополнительно: {sanction['additional']}"

        if line:
            lines.append(f"- {line}")

    if not lines:
        return ""

    return "Санкции:\n" + "\n".join(lines)


def make_source_p_num(article_num: str, part_num: str | None) -> str:
    if not part_num:
        return article_num

    if part_num == article_num:
        return article_num

    return f"{article_num}.{part_num}"


# ============================================================
# PDD PARSER
# ============================================================

def parse_pdd_section(data: dict[str, Any]) -> list[dict[str, Any]]:
    section = data["section"]

    source_type = data.get("source_type", "pdd")
    sec_num = section.get("sec_num")
    section_name = section.get("name")
    section_description = section.get("description")

    documents: list[dict[str, Any]] = []

    def walk_nodes(
        nodes: list[dict[str, Any]],
        parent_p_num: str | None = None,
    ) -> None:
        for node in nodes:
            p_num = node.get("p_num")
            full_text = node.get("full_text", "")

            if p_num and full_text:
                source_p_num = p_num

                text_parts = [
                    "ПДД РФ",
                    f"Раздел {sec_num}. {section_name}",
                ]

                if section_description:
                    text_parts.append(f"Описание раздела: {section_description}")

                if parent_p_num:
                    text_parts.append(f"Родительский пункт: {parent_p_num}")

                text_parts.append(f"Пункт {p_num}")
                text_parts.append(full_text)

                text = "\n\n".join(text_parts)

                metadata = {
                    "source_type": source_type,
                    "source_p_num": source_p_num,
                    "sec_num": sec_num,
                    "section_name": section_name,
                    "p_num": p_num,
                    "parent_p_num": parent_p_num,
                }

                doc_id = safe_id(f"{source_type}:{source_p_num}")

                documents.append(
                    make_document(
                        doc_id=doc_id,
                        text=text,
                        metadata=metadata,
                    )
                )

            children = node.get("p_sup") or []
            if children:
                walk_nodes(children, parent_p_num=p_num)

    walk_nodes(section.get("content", []))

    return documents


# ============================================================
# KOAP / UK PARSER
# ============================================================

def parse_legal_code(data: dict[str, Any]) -> list[dict[str, Any]]:
    source_type = data.get("source_type", "unknown")
    code_title = data.get("title")
    chapter = data.get("chapter")
    last_updated = data.get("last_updated")
    description = data.get("description")

    documents: list[dict[str, Any]] = []

    for article in data.get("articles", []):
        article_num = article.get("article_num")
        article_title = article.get("title")
        article_notes = article.get("notes")
        article_note = article.get("note")
        article_special_notes = article.get("special_notes")

        for part in article.get("parts", []):
            part_num = part.get("part_num")
            part_title = part.get("title")
            content = part.get("content", "")
            sanctions = part.get("sanctions", [])

            if not article_num or not part_num or not content:
                continue

            source_p_num = make_source_p_num(article_num, part_num)

            text_parts = []

            if source_type == "koap":
                text_parts.append("КоАП РФ")
            elif source_type == "uk_rf":
                text_parts.append("УК РФ")
            else:
                text_parts.append(source_type)

            if code_title:
                text_parts.append(code_title)

            if description:
                text_parts.append(f"Описание: {description}")

            if chapter:
                text_parts.append(f"Глава {chapter}")

            text_parts.append(f"Статья {article_num}. {article_title}")

            if part_num != article_num:
                text_parts.append(f"Часть {part_num}")

            if part_title:
                text_parts.append(part_title)

            text_parts.append(content)

            sanctions_text = format_sanctions(sanctions)
            if sanctions_text:
                text_parts.append(sanctions_text)

            if article_notes:
                text_parts.append(f"Редакция / заметки: {article_notes}")

            if article_note:
                text_parts.append(f"Примечание к статье: {article_note}")

            if article_special_notes:
                text_parts.append(f"Специальные пояснения: {article_special_notes}")

            text = "\n\n".join(text_parts)

            metadata = {
                "source_type": source_type,
                "source_p_num": source_p_num,
                "article_num": article_num,
                "article_title": article_title,
                "part_num": part_num,
                "part_title": part_title,
                "chapter": chapter,
                "last_updated": last_updated,
                "sanctions": sanctions,
            }

            doc_id = safe_id(f"{source_type}:{source_p_num}")

            documents.append(
                make_document(
                    doc_id=doc_id,
                    text=text,
                    metadata=metadata,
                )
            )

    return documents


# ============================================================
# UNIVERSAL PARSER
# ============================================================

def parse_json_object(data: Any) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    if isinstance(data, list):
        for item in data:
            documents.extend(parse_json_object(item))
        return documents

    if not isinstance(data, dict):
        return documents

    if "section" in data:
        documents.extend(parse_pdd_section(data))
        return documents

    if "articles" in data:
        documents.extend(parse_legal_code(data))
        return documents

    return documents


def deduplicate_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    result = []

    for doc in documents:
        original_id = doc[ID_FIELD]

        if original_id not in seen:
            seen[original_id] = 1
            result.append(doc)
            continue

        seen[original_id] += 1

        new_doc = dict(doc)
        new_doc[ID_FIELD] = f"{original_id}:{seen[original_id]}"

        result.append(new_doc)

    return result


def parse_all_files(path: Path) -> list[dict[str, Any]]:
    all_documents: list[dict[str, Any]] = []

    json_files = get_json_files(path)

    for json_file in json_files:
        data = load_json(json_file)
        docs = parse_json_object(data)

        for doc in docs:
            doc[METADATA_FIELD]["source_file"] = str(json_file)

        all_documents.extend(docs)

    return deduplicate_documents(all_documents)


# ============================================================
# INGEST
# ============================================================

def split_batches(
    items: list[Any],
    batch_size: int,
) -> list[list[Any]]:
    return [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def ingest_documents(
    documents: list[dict[str, Any]],
) -> None:
    batches = split_batches(documents, BATCH_SIZE)

    for batch_index, batch in enumerate(batches):
        payload = {
            "documents": batch,
            "reset": RESET_INDEX if batch_index == 0 else False,
            "chunk": CHUNK,
            "chunk_config": None,
        }

        response = requests.post(
            INGEST_URL,
            json=payload,
            timeout=120,
        )

        if not response.ok:
            print(f"Ошибка batch {batch_index + 1}/{len(batches)}")
            print("Status:", response.status_code)
            print("Response:", response.text)
            response.raise_for_status()

        print(
            f"OK batch {batch_index + 1}/{len(batches)}: "
            f"{len(batch)} documents"
        )


# ============================================================
# RUN
# ============================================================

documents = parse_all_files(INPUT_PATH)

print(f"Parsed documents: {len(documents)}")

if not documents:
    print("Документы не найдены. Проверь путь и структуру JSON.")
else:
    print("Первый документ:")
    print(json.dumps(documents[0], ensure_ascii=False, indent=2))

    if DRY_RUN:
        print("DRY_RUN=True, данные не отправлены в API.")
    else:
        ingest_documents(documents)
        print("Готово. Данные добавлены в систему.")