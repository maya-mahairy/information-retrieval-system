import csv
import json
import time
from typing import Dict, List

from tqdm import tqdm

from config import (
    HYBRID_PARALLEL_TUNING_CSV_PATH,
    HYBRID_PARALLEL_TUNING_PER_QUERY_PATH,
    HYBRID_PARALLEL_TUNING_SUMMARY_PATH,
)
from services.evaluation_service import (
    average_precision_at_k,
    get_relevant_docs,
    load_qrels_by_query,
    load_queries_by_id,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from services.hybrid_parallel_search_service import HybridParallelSearchService


def evaluate_hybrid_parallel_weights(
    search_service: HybridParallelSearchService,
    bm25_weight: float,
    tfidf_weight: float,
    top_k: int = 100,
    rrf_k: int = 60,
    k1: float = 1.5,
    b: float = 0.75,
):
    """
    Evaluates Hybrid Parallel for one BM25/TF-IDF weight configuration.

    The fusion method is Weighted Reciprocal Rank Fusion.
    """
    print("\n" + "=" * 70)
    print(
        f"EVALUATING HYBRID PARALLEL | "
        f"BM25 weight = {bm25_weight} | TF-IDF weight = {tfidf_weight}"
    )
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
    print("RRF k:", rrf_k)
    print("BM25 weight:", bm25_weight)
    print("TF-IDF weight:", tfidf_weight)
    print("BM25 k1:", k1)
    print("BM25 b:", b)

    per_query_rows = []
    missing_queries = []
    failed_queries = []

    for query_id in tqdm(
        qrel_query_ids,
        desc=f"Parallel BM25 {bm25_weight} TFIDF {tfidf_weight}",
    ):
        query = queries_by_id.get(query_id)

        if query is None:
            missing_queries.append(query_id)
            continue

        query_text = query.get("original_text", "")
        relevant_docs = get_relevant_docs(qrels_by_query[query_id])

        try:
            start_time = time.time()

            results = search_service.search(
                query_text=query_text,
                top_k=top_k,
                bm25_top_k=top_k,
                tfidf_top_k=top_k,
                rrf_k=rrf_k,
                bm25_weight=bm25_weight,
                tfidf_weight=tfidf_weight,
                k1=k1,
                b=b,
            )

            elapsed = time.time() - start_time

            retrieved_doc_ids = [
                str(result.get("doc_id", ""))
                for result in results
            ]

            p10 = precision_at_k(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_docs=relevant_docs,
                k=10,
            )

            r100 = recall_at_k(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_docs=relevant_docs,
                k=top_k,
            )

            ap100 = average_precision_at_k(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_docs=relevant_docs,
                k=top_k,
            )

            ndcg10 = ndcg_at_k(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_docs=relevant_docs,
                k=10,
            )

            per_query_rows.append(
                {
                    "model": "hybrid_parallel",
                    "fusion_method": "weighted_rrf",
                    "bm25_weight": bm25_weight,
                    "tfidf_weight": tfidf_weight,
                    "rrf_k": rrf_k,
                    "query_id": query_id,
                    "query_text": query_text,
                    "relevant_docs_count": len(relevant_docs),
                    "retrieved_docs_count": len(retrieved_doc_ids),
                    "precision_at_10": p10,
                    f"recall_at_{top_k}": r100,
                    f"average_precision_at_{top_k}": ap100,
                    "ndcg_at_10": ndcg10,
                    "time_seconds": elapsed,
                    "k1": k1,
                    "b": b,
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
        "model": "hybrid_parallel",
        "fusion_method": "weighted_rrf",
        "bm25_weight": bm25_weight,
        "tfidf_weight": tfidf_weight,
        "rrf_k": rrf_k,
        "total_qrel_queries": len(qrel_query_ids),
        "loaded_processed_queries": len(queries_by_id),
        "evaluated_queries": evaluated_queries,
        "missing_queries_count": len(missing_queries),
        "failed_queries_count": len(failed_queries),
        "missing_query_ids": missing_queries,
        "failed_queries": failed_queries,
        "top_k": top_k,
        "k1": k1,
        "b": b,
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

    return summary, per_query_rows


def save_parallel_tuning_outputs(summary_rows: List[Dict], per_query_rows: List[Dict]):
    """
    Saves parallel tuning results as JSON and CSV.
    """
    with open(HYBRID_PARALLEL_TUNING_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary_rows, file, ensure_ascii=False, indent=2)

    if summary_rows:
        with open(HYBRID_PARALLEL_TUNING_CSV_PATH, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if per_query_rows:
        with open(HYBRID_PARALLEL_TUNING_PER_QUERY_PATH, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_query_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_query_rows)

    print("\nSaved parallel tuning summary JSON to:", HYBRID_PARALLEL_TUNING_SUMMARY_PATH)
    print("Saved parallel tuning summary CSV to:", HYBRID_PARALLEL_TUNING_CSV_PATH)
    print("Saved parallel tuning per-query CSV to:", HYBRID_PARALLEL_TUNING_PER_QUERY_PATH)


def run_hybrid_parallel_tuning(
    bm25_weights: List[float],
    top_k: int = 100,
    rrf_k: int = 60,
    k1: float = 1.5,
    b: float = 0.75,
):
    """
    Runs Hybrid Parallel evaluation for multiple BM25/TF-IDF weight settings.

    TF-IDF weight is calculated as:
    tfidf_weight = 1 - bm25_weight
    """
    print("=" * 70)
    print("HYBRID PARALLEL TUNING")
    print("=" * 70)
    print("BM25 weights:", bm25_weights)

    search_service = HybridParallelSearchService()

    all_summary_rows = []
    all_per_query_rows = []

    for bm25_weight in bm25_weights:
        tfidf_weight = round(1.0 - bm25_weight, 4)

        summary, per_query_rows = evaluate_hybrid_parallel_weights(
            search_service=search_service,
            bm25_weight=bm25_weight,
            tfidf_weight=tfidf_weight,
            top_k=top_k,
            rrf_k=rrf_k,
            k1=k1,
            b=b,
        )

        all_summary_rows.append(summary)
        all_per_query_rows.extend(per_query_rows)

    save_parallel_tuning_outputs(
        summary_rows=all_summary_rows,
        per_query_rows=all_per_query_rows,
    )

    print("\n" + "=" * 70)
    print("PARALLEL TUNING SUMMARY")
    print("=" * 70)

    for row in all_summary_rows:
        print(
            "BM25 weight =", row["bm25_weight"],
            "| TF-IDF weight =", row["tfidf_weight"],
            "| P@10 =", round(row["precision_at_10"], 4),
            f"| MAP@{top_k} =", round(row[f"map_at_{top_k}"], 4),
            "| nDCG@10 =", round(row["ndcg_at_10"], 4),
            "| time =", round(row["mean_time_seconds"], 4),
        )

    best_by_map = max(
        all_summary_rows,
        key=lambda row: row[f"map_at_{top_k}"],
    )

    print("\nBest BM25 weight by MAP:", best_by_map["bm25_weight"])
    print("Best TF-IDF weight by MAP:", best_by_map["tfidf_weight"])
    print(f"Best MAP@{top_k}:", round(best_by_map[f"map_at_{top_k}"], 4))

    return all_summary_rows