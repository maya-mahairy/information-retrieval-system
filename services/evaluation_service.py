import csv
import json
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

from config import (
    EVALUATION_PER_QUERY_PATH,
    EVALUATION_RUN_DETAILS_PATH,
    EVALUATION_SUMMARY_PATH,
    PROCESSED_QUERIES_PATH,
    QRELS_PATH,
)
from services.bm25_search_service import BM25SearchService
from services.tfidf_search_service import TFIDFSearchService
from services.hybrid_serial_search_service import HybridSerialSearchService
from services.hybrid_parallel_search_service import HybridParallelSearchService
from services.refined_bm25_search_service import RefinedBM25SearchService


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_queries_by_id() -> Dict[str, Dict]:
    queries = load_jsonl(PROCESSED_QUERIES_PATH)

    return {
        str(query["query_id"]): query
        for query in queries
    }


def load_qrels_by_query() -> Dict[str, Dict[str, int]]:
    qrels = load_jsonl(QRELS_PATH)

    qrels_by_query = defaultdict(dict)

    for row in qrels:
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        relevance = int(row.get("relevance", 0))

        previous_relevance = qrels_by_query[query_id].get(doc_id, 0)
        qrels_by_query[query_id][doc_id] = max(previous_relevance, relevance)

    return dict(qrels_by_query)


def get_relevant_docs(qrels_for_query: Dict[str, int]) -> Dict[str, int]:
    return {
        doc_id: relevance
        for doc_id, relevance in qrels_for_query.items()
        if relevance > 0
    }


def precision_at_k(retrieved_doc_ids: List[str], relevant_docs: Dict[str, int], k: int = 10) -> float:
    top_k_docs = retrieved_doc_ids[:k]

    if not top_k_docs:
        return 0.0

    relevant_retrieved = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)

    return relevant_retrieved / k


def recall_at_k(retrieved_doc_ids: List[str], relevant_docs: Dict[str, int], k: int = 100) -> float:
    if not relevant_docs:
        return 0.0

    top_k_docs = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)

    return relevant_retrieved / len(relevant_docs)


def average_precision_at_k(retrieved_doc_ids: List[str], relevant_docs: Dict[str, int], k: int = 100) -> float:
    if not relevant_docs:
        return 0.0

    score_sum = 0.0
    relevant_found = 0

    top_k_docs = retrieved_doc_ids[:k]

    for rank, doc_id in enumerate(top_k_docs, start=1):
        if doc_id in relevant_docs:
            relevant_found += 1
            score_sum += relevant_found / rank

    denominator = min(len(relevant_docs), k)

    if denominator == 0:
        return 0.0

    return score_sum / denominator


def dcg_at_k(retrieved_doc_ids: List[str], relevant_docs: Dict[str, int], k: int = 10) -> float:
    dcg = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        relevance = relevant_docs.get(doc_id, 0)

        if relevance <= 0:
            continue

        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)

        dcg += gain / discount

    return dcg


def ideal_dcg_at_k(relevant_docs: Dict[str, int], k: int = 10) -> float:
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]

    idcg = 0.0

    for rank, relevance in enumerate(ideal_relevances, start=1):
        gain = (2 ** relevance) - 1
        discount = math.log2(rank + 1)
        idcg += gain / discount

    return idcg


def ndcg_at_k(retrieved_doc_ids: List[str], relevant_docs: Dict[str, int], k: int = 10) -> float:
    idcg = ideal_dcg_at_k(relevant_docs, k=k)

    if idcg == 0:
        return 0.0

    return dcg_at_k(retrieved_doc_ids, relevant_docs, k=k) / idcg


def get_search_service(model_name: str):
    model_name = model_name.lower()

    if model_name == "tfidf":
        return TFIDFSearchService()

    if model_name == "bm25":
        return BM25SearchService()

    if model_name == "hybrid_serial":
        return HybridSerialSearchService()

    if model_name == "hybrid_parallel":
        return HybridParallelSearchService()

    if model_name == "bm25_refined_corrected":
        return RefinedBM25SearchService(mode="corrected")

    if model_name == "bm25_refined_expanded":
        return RefinedBM25SearchService(mode="expanded")

    raise ValueError(f"Unsupported model name: {model_name}")


def run_search(model_name: str, search_service, query_text: str, top_k: int):
    if model_name == "bm25":
        return search_service.search(
            query_text=query_text,
            top_k=top_k,
            k1=1.5,
            b=0.75,
        )

    if model_name == "hybrid_serial":
        return search_service.search(
            query_text=query_text,
            top_k=top_k,
            bm25_candidates=100,
            k1=1.5,
            b=0.75,
            alpha=0.9,
        )

    if model_name == "hybrid_parallel":
        return search_service.search(
            query_text=query_text,
            top_k=top_k,
            bm25_top_k=top_k,
            tfidf_top_k=top_k,
            rrf_k=60,
            bm25_weight=0.95,
            tfidf_weight=0.05,
            k1=1.5,
            b=0.75,
        )

    if model_name in {"bm25_refined_corrected", "bm25_refined_expanded"}:
        return search_service.search(
            query_text=query_text,
            top_k=top_k,
            k1=1.5,
            b=0.75,
        )

    return search_service.search(
        query_text=query_text,
        top_k=top_k,
    )


def evaluate_model(model_name: str, top_k: int = 100) -> Tuple[Dict, List[Dict], List[Dict]]:
    model_name = model_name.lower()

    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print("=" * 70)

    queries_by_id = load_queries_by_id()
    qrels_by_query = load_qrels_by_query()

    qrel_query_ids = sorted(
        qrels_by_query.keys(),
        key=lambda value: int(value) if value.isdigit() else value,
    )

    print("Total qrel queries:", len(qrel_query_ids))
    print("Loaded processed queries:", len(queries_by_id))
    print("Top K for evaluation:", top_k)

    search_service = get_search_service(model_name)

    per_query_rows = []
    run_details_rows = []

    missing_queries = []
    failed_queries = []

    for query_id in tqdm(qrel_query_ids, desc=f"Evaluating {model_name}"):
        query = queries_by_id.get(query_id)

        if query is None:
            missing_queries.append(query_id)
            continue

        query_text = query.get("original_text", "")
        relevant_docs = get_relevant_docs(qrels_by_query[query_id])

        try:
            start_time = time.time()
            results = run_search(
                model_name=model_name,
                search_service=search_service,
                query_text=query_text,
                top_k=top_k,
            )
            elapsed = time.time() - start_time

            retrieved_doc_ids = [str(result.get("doc_id", "")) for result in results]

            p10 = precision_at_k(retrieved_doc_ids, relevant_docs, k=10)
            r100 = recall_at_k(retrieved_doc_ids, relevant_docs, k=top_k)
            ap100 = average_precision_at_k(retrieved_doc_ids, relevant_docs, k=top_k)
            ndcg10 = ndcg_at_k(retrieved_doc_ids, relevant_docs, k=10)

            per_query_rows.append(
                {
                    "model": model_name,
                    "query_id": query_id,
                    "query_text": query_text,
                    "relevant_docs_count": len(relevant_docs),
                    "retrieved_docs_count": len(retrieved_doc_ids),
                    "precision_at_10": p10,
                    f"recall_at_{top_k}": r100,
                    f"average_precision_at_{top_k}": ap100,
                    "ndcg_at_10": ndcg10,
                    "time_seconds": elapsed,
                }
            )

            run_details_rows.append(
                {
                    "model": model_name,
                    "query_id": query_id,
                    "query_text": query_text,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "top_results": [
                        {
                            "rank": result.get("rank"),
                            "doc_id": result.get("doc_id"),
                            "score": result.get("score"),
                            "title": result.get("title"),
                        }
                        for result in results[:10]
                    ],
                }
            )

        except Exception as error:
            failed_queries.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "error": str(error),
                }
            )

    evaluated_queries = len(per_query_rows)

    def mean_metric(metric_name: str) -> float:
        if not per_query_rows:
            return 0.0

        return sum(float(row[metric_name]) for row in per_query_rows) / len(per_query_rows)

    summary = {
        "model": model_name,
        "total_qrel_queries": len(qrel_query_ids),
        "loaded_processed_queries": len(queries_by_id),
        "evaluated_queries": evaluated_queries,
        "missing_queries_count": len(missing_queries),
        "failed_queries_count": len(failed_queries),
        "missing_query_ids": missing_queries,
        "failed_queries": failed_queries,
        "top_k": top_k,
        "precision_at_10": mean_metric("precision_at_10"),
        f"recall_at_{top_k}": mean_metric(f"recall_at_{top_k}"),
        f"map_at_{top_k}": mean_metric(f"average_precision_at_{top_k}"),
        "ndcg_at_10": mean_metric("ndcg_at_10"),
        "mean_time_seconds": mean_metric("time_seconds"),
    }

    print("\nEvaluation check:")
    print("Total qrel queries:", summary["total_qrel_queries"])
    print("Evaluated queries:", summary["evaluated_queries"])
    print("Missing queries:", summary["missing_queries_count"])
    print("Failed queries:", summary["failed_queries_count"])

    print("\nMetrics:")
    print("Precision@10:", round(summary["precision_at_10"], 4))
    print(f"Recall@{top_k}:", round(summary[f"recall_at_{top_k}"], 4))
    print(f"MAP@{top_k}:", round(summary[f"map_at_{top_k}"], 4))
    print("nDCG@10:", round(summary["ndcg_at_10"], 4))
    print("Mean query time:", round(summary["mean_time_seconds"], 4), "seconds")

    return summary, per_query_rows, run_details_rows


def save_evaluation_outputs(summaries: List[Dict], per_query_rows: List[Dict], run_details_rows: List[Dict]):
    with open(EVALUATION_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)

    if per_query_rows:
        fieldnames = list(per_query_rows[0].keys())

        with open(EVALUATION_PER_QUERY_PATH, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)

    with open(EVALUATION_RUN_DETAILS_PATH, "w", encoding="utf-8") as file:
        for row in run_details_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nSaved evaluation summary to:", EVALUATION_SUMMARY_PATH)
    print("Saved per-query metrics to:", EVALUATION_PER_QUERY_PATH)
    print("Saved run details to:", EVALUATION_RUN_DETAILS_PATH)


def evaluate_models(model_names: List[str], top_k: int = 100):
    all_summaries = []
    all_per_query_rows = []
    all_run_details_rows = []

    for model_name in model_names:
        summary, per_query_rows, run_details_rows = evaluate_model(
            model_name=model_name,
            top_k=top_k,
        )

        all_summaries.append(summary)
        all_per_query_rows.extend(per_query_rows)
        all_run_details_rows.extend(run_details_rows)

    save_evaluation_outputs(
        summaries=all_summaries,
        per_query_rows=all_per_query_rows,
        run_details_rows=all_run_details_rows,
    )

    print("\nAll evaluations completed successfully.")

    return all_summaries