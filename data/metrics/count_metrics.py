import json
import math
import requests
from tqdm import tqdm


def normalize_doc_id(doc_id: str) -> str:
    parts = doc_id.split(":")
    if len(parts) == 3:
        return ":".join(parts[:-1])
    return doc_id


def dcg(relevances):
    return sum(
        rel / math.log2(idx + 2)
        for idx, rel in enumerate(relevances)
    )


def ndcg_at_k(predicted_sources, true_source, k):
    relevances = [
        1 if source == true_source else 0
        for source in predicted_sources[:k]
    ]

    dcg_score = dcg(relevances)
    ideal_dcg = dcg([1] + [0] * (k - 1))

    return dcg_score / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(predicted_sources, true_source, k):
    return int(true_source in predicted_sources[:k])


with open(
    r"C:\Users\garan\PycharmProjects\ods_project_2026\data\train_evel_datasets\eval_with_source.json",
    encoding="utf8"
) as f:
    data = json.load(f)

recall_1 = 0
recall_5 = 0
ndcg_1 = 0.0
ndcg_5 = 0.0

for item in tqdm(data):
    response = requests.post(
        "http://127.0.0.1:8000/v1/search",
        json={"query": item["query"]}
    ).json()

    predicted_sources = [
        normalize_doc_id(result["doc_id"])
        for result in response["results"]
    ]

    true_source = item["source"]

    recall_1 += recall_at_k(predicted_sources, true_source, 1)
    recall_5 += recall_at_k(predicted_sources, true_source, 5)

    ndcg_1 += ndcg_at_k(predicted_sources, true_source, 1)
    ndcg_5 += ndcg_at_k(predicted_sources, true_source, 5)

n = len(data)

print("Recall@1:", recall_1 / n)
print("Recall@5:", recall_5 / n)
print("NDCG@1:", ndcg_1 / n)
print("NDCG@5:", ndcg_5 / n)