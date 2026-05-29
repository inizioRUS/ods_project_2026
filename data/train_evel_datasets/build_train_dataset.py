#!/usr/bin/env python3
"""
Сборка обучающего датасета для дообучения эмбеддера в RAG по ПДД РФ.

Источники:
  - data/koap.json              — КоАП РФ, глава 12 (адм. ответственность за нарушения ПДД)
  - data/uk_rf.json             — УК РФ, статьи 104.1, 264, 264.1-264.3, 267 (уголовка по ПДД)
  - data/files/pdd.json         — Правила дорожного движения РФ (разделы 1..N)

Формат выхода (JSONL, по одной записи на строку):
  {"query": "...", "positive": "...", "negative": "..."}

  positive — корректный (релевантный) фрагмент с цитатой источника.
  negative — «hard negative»: близкий по теме, но НЕ отвечающий на вопрос фрагмент
             (по умолчанию — другая часть той же статьи / другой пункт того же раздела).

Использование:
    python data/build_train_dataset.py
    # → пересоберёт data/train.jsonl

Параметры через переменные окружения:
    OUT_PATH        путь к итоговому JSONL  (по умолчанию data/train.jsonl)
    KEEP_SEED       1/0 — мерджить ли существующие примеры из train.jsonl (по умолчанию 1)
    MIN_LEN         минимальная длина content для генерации (по умолчанию 60 символов)
    SEED            random seed (по умолчанию 42)
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ----------------------------- Конфиг -----------------------------

ROOT = Path(__file__).resolve().parent
OUT_PATH = Path(os.environ.get("OUT_PATH", str(ROOT / "train.jsonl")))
KEEP_SEED = os.environ.get("KEEP_SEED", "1") == "1"
MIN_LEN = int(os.environ.get("MIN_LEN", "60"))
SEED = int(os.environ.get("SEED", "3"))

random.seed(SEED)

# ------------------------- Загрузка JSON --------------------------


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


KOAP = load_json(ROOT / "koap.json")
UK = load_json(ROOT / "uk_rf.json")
PDD = load_json(ROOT / "files" / "pdd.json")


# ---------------------------- Утилиты -----------------------------


_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s).strip()
    s = s.replace(" - ", " — ")
    return s


def lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def cap_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def strip_trailing_punct(s: str) -> str:
    return s.rstrip(" .,;:—-")


# Разметка «нарушение / наказание»: всё, что до « - влечет » / «наказывается» — это «состав»
_SPLIT_MARKERS = (
    " — влечёт ",
    " - влечёт ",
    " — влечет ",
    " - влечет ",
    ", — влечёт ",
    ", - влечёт ",
    ", — влечет ",
    ", - влечет ",
    " наказывается ",
    " наказываются ",
)


def split_violation_penalty(content: str) -> tuple[str, str]:
    """Делит текст КоАП/УК на (состав, санкция). Если не разобрали — (content, '')."""
    for m in _SPLIT_MARKERS:
        idx = content.find(m)
        if idx > 20:
            return content[:idx].rstrip(" ,;:—-"), content[idx + len(m):]
    return content, ""


def truncate_subject(s: str, max_len: int = 160) -> tuple[str, bool]:
    """Урезает «состав» до читабельной длины. Возвращает (text, was_truncated)."""
    s = clean_text(s)
    if len(s) <= max_len:
        return s, False
    cut = s[:max_len]
    for sep in (", ", "; ", ": "):
        idx = cut.rfind(sep)
        if idx > max_len // 2:
            return cut[:idx], True
    return cut.rstrip(), True


# Маркеры «определительных» частей (примечания, дефиниции) — для них генерим другие вопросы
_DEFINITION_PREFIXES = (
    "под ",
    "примечание",
    "примечания",
    "крупным ",
    "лицо ",
    "лицом ",
    "в настоящей статье",
    "для целей настоящей",
)


def is_definition_content(content: str) -> bool:
    low = content.lower().lstrip()
    if any(low.startswith(p) for p in _DEFINITION_PREFIXES):
        return True
    # Содержит «следует понимать» / «признается» / «не привлекается» — определение/норма-исключение
    if "следует понимать" in low or "признается" in low or "признаётся" in low:
        return True
    return False


# Подсказки тем (для случаев «состав слишком длинный» — используем краткую тему из заголовка статьи)
def short_topic_from_title(title: str) -> str:
    t = clean_text(title).lower()
    # Часто заголовок уже компактен; режем сверх 120 символов
    if len(t) > 120:
        t = t[:120].rsplit(",", 1)[0]
    return t


# ----------------------- Сборка «пассажей» ------------------------


def build_koap_passages() -> list[dict]:
    """По одной записи на каждую часть статьи КоАП с содержательным текстом."""
    passages: list[dict] = []
    for article in KOAP.get("articles", []):
        anum = article["article_num"]
        atitle = clean_text(article["title"])
        for part in article.get("parts", []):
            pnum = part["part_num"]
            content = clean_text(part.get("content", ""))
            if len(content) < MIN_LEN:
                continue
            # Часть «заголовок» (part_num == article_num или совпадает с заголовком статьи) — пропускаем
            if pnum == anum or content == atitle:
                continue
            citation = f"Статья {anum} КоАП РФ, часть {pnum}."
            violation, _penalty = split_violation_penalty(content)
            subject_full, was_trunc = truncate_subject(
                lower_first(strip_trailing_punct(violation)), max_len=120
            )
            # При слишком длинной/обрезанной формулировке состава — используем короткую тему из заголовка
            if was_trunc or len(subject_full) > 120:
                subject = short_topic_from_title(atitle)
            else:
                subject = subject_full
            passages.append(
                {
                    "uid": f"koap:{anum}:{pnum}",
                    "source": "koap",
                    "anchor_key": f"koap-art-{anum}",
                    "article_num": anum,
                    "part_num": pnum,
                    "article_title": atitle,
                    "subject": subject,
                    "content": content,
                    "citation": citation,
                    "positive_text": f"{content} — {citation}",
                    "is_definition": is_definition_content(content),
                }
            )
    return passages


def build_uk_passages() -> list[dict]:
    passages: list[dict] = []
    for article in UK.get("articles", []):
        anum = article["article_num"]
        atitle = clean_text(article["title"])
        for part in article.get("parts", []):
            pnum = part["part_num"]
            content = clean_text(part.get("content", ""))
            if len(content) < MIN_LEN:
                continue
            citation = f"Статья {anum} УК РФ, часть {pnum}."
            violation, _penalty = split_violation_penalty(content)
            subject_full, was_trunc = truncate_subject(
                lower_first(strip_trailing_punct(violation)), max_len=120
            )
            low = content.lower()
            # Если состав ссылается на «частью первой настоящей статьи» (квалифицирующий признак),
            # используем заголовок статьи + краткий маркер из квалификации
            references_part_one = (
                "предусмотренное частью первой" in low
                or "предусмотренные частью первой" in low
                or "предусмотренное настоящей статьей" in low
            )
            if references_part_one:
                qualifier_bits = []
                if "тяжкого вреда" in low and "смерть" not in low:
                    qualifier_bits.append("причинение тяжкого вреда здоровью")
                if "смерть двух" in low or "смерть более" in low:
                    qualifier_bits.append("гибель двух или более лиц")
                elif "смерть человека" in low:
                    qualifier_bits.append("гибель человека")
                if "опьянения" in low or "опьянении" in low:
                    qualifier_bits.append("в состоянии опьянения")
                if "оставлен" in low:
                    qualifier_bits.append("со скрытием с места ДТП")
                if "не имеющим" in low or "лишенным права" in low:
                    qualifier_bits.append("лицом без прав")
                if qualifier_bits:
                    subject = lower_first(atitle) + " (" + ", ".join(qualifier_bits) + ")"
                else:
                    subject = lower_first(atitle)
            elif was_trunc or len(subject_full) > 120:
                subject = short_topic_from_title(atitle)
            else:
                subject = subject_full
            passages.append(
                {
                    "uid": f"uk:{anum}:{pnum}",
                    "source": "uk",
                    "anchor_key": f"uk-art-{anum}",
                    "article_num": anum,
                    "part_num": pnum,
                    "article_title": atitle,
                    "subject": subject,
                    "content": content,
                    "citation": citation,
                    "positive_text": f"{content} — {citation}",
                    "is_definition": is_definition_content(content),
                }
            )
    return passages


def build_pdd_passages() -> list[dict]:
    """По одной записи на каждый пункт ПДД (включая под-пункты p_sup)."""
    passages: list[dict] = []
    for entry in PDD:
        section = entry.get("section") or {}
        sec_num = section.get("sec_num")
        sec_name = clean_text(section.get("name", ""))
        for item in section.get("content", []) or []:
            queue = [(item.get("p_num"), item.get("full_text", ""))]
            for sup in item.get("p_sup", []) or []:
                queue.append((sup.get("p_num"), sup.get("full_text", "")))
            for pnum, full_text in queue:
                content = clean_text(full_text)
                if len(content) < MIN_LEN:
                    continue
                citation = f"пункт {pnum} ПДД РФ."
                passages.append(
                    {
                        "uid": f"pdd:{pnum}",
                        "source": "pdd",
                        "anchor_key": f"pdd-sec-{sec_num}",
                        "sec_num": sec_num,
                        "sec_name": sec_name,
                        "p_num": pnum,
                        "content": content,
                        "citation": citation,
                        "positive_text": f"{content} — {citation}",
                    }
                )
    return passages


# ----------------------- Генерация запросов -----------------------

KOAP_TEMPLATES = [
    "Какой штраф за {s}?",
    "Что грозит за {s}?",
    "Какое наказание за {s}?",
    "Какая ответственность за {s}?",
    "Что будет за {s}?",
    "Какой размер штрафа за {s}?",
    "Чем наказывается {s}?",
    "Какая статья КоАП РФ за {s}?",
]

UK_TEMPLATES = [
    "Какое уголовное наказание за {s}?",
    "Что грозит по УК РФ за {s}?",
    "Сколько лет дают за {s}?",
    "Какая уголовная ответственность за {s}?",
    "Что предусмотрено УК РФ за {s}?",
    "Какая статья УК РФ за {s}?",
]


DEFINITION_TEMPLATES_KOAP = [
    "Как определяется в КоАП РФ ({citation_short})?",
    "Что означает примечание к {citation_short}?",
    "Как трактуется в статье {anum} КоАП РФ?",
]

DEFINITION_TEMPLATES_UK = [
    "Как определяется в УК РФ ({citation_short})?",
    "Что означает примечание к {citation_short}?",
    "Что считается крупным ущербом по статье {anum} УК РФ?",
]


def make_law_queries(p: dict, templates: list[str], k: int = 2) -> list[str]:
    """Генерирует k вариантов вопросов по статье закона."""
    # Для определительных частей — отдельные шаблоны (не про штраф/наказание).
    if p.get("is_definition"):
        anum = p["article_num"]
        citation_short = (
            f"ст. {anum} КоАП РФ" if p["source"] == "koap" else f"ст. {anum} УК РФ"
        )
        tpls = (DEFINITION_TEMPLATES_KOAP if p["source"] == "koap" else DEFINITION_TEMPLATES_UK).copy()
        random.shuffle(tpls)
        out = []
        for t in tpls[:k]:
            out.append(t.format(anum=anum, citation_short=citation_short))
        return out

    subj = p.get("subject") or lower_first(p.get("article_title", ""))
    if not subj or len(subj) < 5:
        subj = lower_first(p.get("article_title", ""))
    subj = strip_trailing_punct(subj)
    # Защита от мусорных «обрывков»
    if subj.endswith("…") or len(subj) < 6:
        subj = lower_first(p.get("article_title", "")) or subj

    tpls = templates.copy()
    random.shuffle(tpls)
    seen, out = set(), []
    for t in tpls:
        q = t.format(s=subj)
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= k:
            break
    return out


# --------- ПДД: генерация вопросов ----------
#
# Стратегия: тематические шаблоны срабатывают только если ключевая тема (обгон, скорость,
# перекрёсток, ...) реально совпадает с темой раздела ПДД (sec_name). Это уменьшает
# семантические рассогласования вида «вопрос про пешеходов → ответ про водителей».


def _first_sentence(s: str, max_len: int = 160) -> str:
    cut = re.split(r"(?<=[\.;:])\s", s, maxsplit=1)[0]
    text, _ = truncate_subject(cut, max_len)
    return text


# Топик (по ключевому слову в названии раздела ПДД) → список вопросных шаблонов
PDD_SECTION_TEMPLATES: dict[str, list[str]] = {
    "скорост": [
        "Какая максимально разрешённая скорость по ПДД РФ (пункт {p})?",
        "Что говорит ПДД РФ об ограничениях скорости движения (пункт {p})?",
        "Какие ограничения скорости устанавливает пункт {p} ПДД РФ?",
    ],
    "обгон": [
        "Что говорит ПДД РФ про обгон (пункт {p})?",
        "В каких случаях запрещён обгон по ПДД РФ (пункт {p})?",
        "Какие правила выполнения обгона устанавливает пункт {p} ПДД?",
    ],
    "перекрест": [
        "Какие правила проезда перекрёстков по пункту {p} ПДД РФ?",
        "Как нужно проезжать перекрёсток по пункту {p} ПДД РФ?",
    ],
    "перекрёст": [
        "Какие правила проезда перекрёстков по пункту {p} ПДД РФ?",
    ],
    "сигнал": [
        "Что означают сигналы светофора (регулировщика) по пункту {p} ПДД РФ?",
        "Какой сигнал светофора (регулировщика) описывает пункт {p} ПДД?",
    ],
    "светофор": [
        "Что означают сигналы светофора по пункту {p} ПДД РФ?",
    ],
    "пешеход": [
        "Какие обязанности у пешеходов по пункту {p} ПДД РФ?",
        "Какие правила движения пешеходов по пункту {p} ПДД РФ?",
    ],
    "пассажир": [
        "Какие обязанности у пассажиров по пункту {p} ПДД РФ?",
    ],
    "разворот": [
        "Где разрешён или запрещён разворот по ПДД (пункт {p})?",
        "Какие правила выполнения разворота по пункту {p} ПДД РФ?",
    ],
    "манёвр": [
        "Какие правила выполнения манёвров устанавливает пункт {p} ПДД РФ?",
    ],
    "маневр": [
        "Какие правила выполнения манёвров устанавливает пункт {p} ПДД РФ?",
    ],
    "остановк": [
        "Где запрещена остановка или стоянка по пункту {p} ПДД РФ?",
        "Какие правила остановки и стоянки устанавливает пункт {p} ПДД РФ?",
    ],
    "стоянк": [
        "Где запрещена остановка или стоянка по пункту {p} ПДД РФ?",
    ],
    "буксиров": [
        "Какие правила буксировки устанавливает пункт {p} ПДД РФ?",
    ],
    "переезд": [
        "Какие правила движения через железнодорожный переезд устанавливает пункт {p} ПДД РФ?",
    ],
    "автомагистрал": [
        "Какие правила движения по автомагистрали устанавливает пункт {p} ПДД РФ?",
    ],
    "жил": [
        "Какие правила движения в жилых зонах устанавливает пункт {p} ПДД РФ?",
    ],
    "аварийн": [
        "Когда нужно включать аварийную сигнализацию или знак аварийной остановки (пункт {p} ПДД РФ)?",
    ],
    "сигнализаци": [
        "Когда нужно включать аварийную сигнализацию или знак аварийной остановки (пункт {p} ПДД РФ)?",
    ],
    "световые приборы": [
        "Когда нужно включать световые приборы по пункту {p} ПДД РФ?",
    ],
    "светов": [
        "Когда нужно включать световые приборы по пункту {p} ПДД РФ?",
    ],
    "людей": [
        "Какие правила перевозки людей устанавливает пункт {p} ПДД РФ?",
    ],
    "груз": [
        "Какие правила перевозки грузов устанавливает пункт {p} ПДД РФ?",
    ],
    "велосипед": [
        "Какие правила движения велосипедистов устанавливает пункт {p} ПДД РФ?",
    ],
    "мопед": [
        "Какие правила движения для мопедов устанавливает пункт {p} ПДД РФ?",
    ],
    "обязанност": [
        "Какие обязанности водителя по пункту {p} ПДД РФ?",
        "Что обязан делать водитель по пункту {p} ПДД РФ?",
    ],
    "дисциплин": [
        "Какие требования по дисциплине движения устанавливает пункт {p} ПДД РФ?",
    ],
    "расположен": [
        "Какие правила расположения транспортных средств на проезжей части устанавливает пункт {p} ПДД РФ?",
    ],
    "знак": [
        "Что устанавливает пункт {p} ПДД РФ о дорожных знаках?",
    ],
}


def make_pdd_queries(p: dict, k: int = 2) -> list[str]:
    sec = p.get("sec_name", "")
    p_num = p.get("p_num")
    sec_low = sec.lower()
    text_low = p["content"].lower()

    candidates: list[str] = []

    # 1) Шаблоны, привязанные к теме раздела (если ключ присутствует и в названии раздела,
    #    и в тексте пункта — это укрепляет соответствие вопрос↔ответ).
    for key, tpls in PDD_SECTION_TEMPLATES.items():
        if key in sec_low and key in text_low:
            candidates += [t.format(p=p_num) for t in tpls]

    # 2) Универсальные смысловые шаблоны на основе модальных маркеров в тексте.
    if "запрещ" in text_low:
        candidates.append(f"Что запрещает пункт {p_num} ПДД РФ?")
    if "обязан" in text_low or "должен" in text_low:
        candidates.append(f"Что обязан делать участник движения по пункту {p_num} ПДД РФ?")
    if "разреш" in text_low:
        candidates.append(f"Что разрешено пунктом {p_num} ПДД РФ?")

    # 3) Универсальный fallback с именем раздела
    candidates.append(f"Что регулирует пункт {p_num} ПДД РФ (раздел «{sec.lower()}»)?")
    candidates.append(f"О чём пункт {p_num} ПДД РФ?")

    seen, out = set(), []
    random.shuffle(candidates)
    for q in candidates:
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= k:
            break
    return out


# -------------------- Поиск hard negatives ------------------------


def pick_hard_negative(
    pos: dict,
    grouped: dict[str, list[dict]],
    all_passages: list[dict],
    rng: random.Random,
) -> dict | None:
    """Берём другую часть/пункт того же якоря (та же статья / тот же раздел ПДД)."""
    same_group = [x for x in grouped.get(pos["anchor_key"], []) if x["uid"] != pos["uid"]]
    if same_group:
        return rng.choice(same_group)
    # Запасной вариант: соседняя группа того же источника
    same_source = [x for x in all_passages if x["source"] == pos["source"] and x["uid"] != pos["uid"]]
    if same_source:
        return rng.choice(same_source)
    return None


# ----------------------- Главный пайплайн -------------------------


def build_triplets() -> list[dict]:
    koap_pass = build_koap_passages()
    uk_pass = build_uk_passages()
    pdd_pass = build_pdd_passages()
    all_pass = koap_pass + uk_pass + pdd_pass

    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in all_pass:
        grouped[p["anchor_key"]].append(p)

    rng = random.Random(SEED)
    triplets: list[dict] = []
    skipped_no_negative = 0

    def emit(p: dict, queries: Iterable[str]) -> None:
        nonlocal skipped_no_negative
        for q in queries:
            neg = pick_hard_negative(p, grouped, all_pass, rng)
            if neg is None:
                skipped_no_negative += 1
                continue
            triplets.append(
                {
                    "query": q,
                    "positive": p["positive_text"],
                    "negative": neg["positive_text"],
                }
            )

    for p in koap_pass:
        emit(p, make_law_queries(p, KOAP_TEMPLATES, k=2))
    for p in uk_pass:
        emit(p, make_law_queries(p, UK_TEMPLATES, k=2))
    for p in pdd_pass:
        emit(p, make_pdd_queries(p, k=2))

    if skipped_no_negative:
        print(f"[warn] Пропущено {skipped_no_negative} примеров: не нашёлся hard negative.")
    print(f"[info] Сгенерировано триплетов: {len(triplets)}")
    print(f"       КоАП passages: {len(koap_pass)}")
    print(f"       УК РФ passages: {len(uk_pass)}")
    print(f"       ПДД passages:  {len(pdd_pass)}")
    return triplets


# --------------------- Мердж с уже имеющимся ----------------------


def read_existing_seeds(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in obj for k in ("query", "positive")):
                continue
            rows.append(obj)
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r in rows:
        key = (r["query"].strip(), r["positive"][:200].strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    generated = build_triplets()

    seeds: list[dict] = []
    if KEEP_SEED:
        seeds = read_existing_seeds(OUT_PATH)
        print(f"[info] Существующих seed-примеров: {len(seeds)}")

    combined = seeds + generated
    before = len(combined)
    combined = dedupe(combined)
    print(f"[info] После дедупликации: {len(combined)} (убрано {before - len(combined)})")

    rng = random.Random(SEED)
    rng.shuffle(combined)

    write_jsonl(combined, OUT_PATH)
    print(f"[ok]  Записано {len(combined)} строк → {OUT_PATH}")


if __name__ == "__main__":
    main()
