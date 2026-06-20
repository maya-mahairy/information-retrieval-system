import json
from typing import Optional

from tqdm import tqdm

from config import (
    DATASET_INFO_PATH,
    EXPECTED_DOCS_COUNT,
    EXPECTED_QUERIES_COUNT,
    IRDS_DATASET_ID,
    PROCESSED_DOCS_PATH,
    PROCESSED_QUERIES_PATH,
    QRELS_PATH,
)
from services.dataset_service import load_docs_stream, load_qrels, load_queries
from services.preprocessing_service import preprocess_document, preprocess_query


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_queries():
    queries = load_queries()
    processed_queries = []

    for query in queries:
        query_dict = dict(query)
        processed_query = preprocess_query(query_dict)
        processed_queries.append(processed_query)

    write_jsonl(PROCESSED_QUERIES_PATH, processed_queries)

    return {
        "queries_count": len(processed_queries),
        "processed_queries_path": str(PROCESSED_QUERIES_PATH),
    }


def prepare_qrels():
    qrels = load_qrels()
    qrels_rows = []
    unique_query_ids = set()

    for qrel in qrels:
        qrel_dict = dict(qrel)

        row = {
            "query_id": str(qrel_dict.get("query_id", "")),
            "doc_id": str(qrel_dict.get("doc_id", "")),
            "relevance": int(qrel_dict.get("relevance", 0)),
            "iteration": str(qrel_dict.get("iteration", "0")),
        }

        qrels_rows.append(row)
        unique_query_ids.add(row["query_id"])

    write_jsonl(QRELS_PATH, qrels_rows)

    return {
        "qrels_count": len(qrels_rows),
        "unique_qrel_queries_count": len(unique_query_ids),
        "qrels_path": str(QRELS_PATH),
    }


def prepare_docs(limit: Optional[int] = None):
    docs_stream = load_docs_stream()

    processed_count = 0

    with open(PROCESSED_DOCS_PATH, "w", encoding="utf-8") as file:
        for doc in tqdm(docs_stream, desc="Preprocessing documents"):
            doc_dict = dict(doc)
            processed_doc = preprocess_document(doc_dict)

            row = {
                "doc_id": processed_doc["doc_id"],
                "title": processed_doc["title"],
                "stance": processed_doc["stance"],
                "url": processed_doc["url"],
                "original_text": processed_doc["original_text"],
                "searchable_text": processed_doc["searchable_text"],
                "cleaned_text": processed_doc["cleaned_text"],
                "token_count": len(processed_doc["tokens"]),
            }

            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed_count += 1

            if limit is not None and processed_count >= limit:
                break

    return {
        "processed_docs_count": processed_count,
        "processed_docs_path": str(PROCESSED_DOCS_PATH),
        "is_full_dataset": limit is None,
    }


def prepare_all_data(limit: Optional[int] = None):
    print("=" * 70)
    print("DATASET PREPARATION PIPELINE")
    print("=" * 70)

    print("\nPreparing queries...")
    queries_info = prepare_queries()
    print(queries_info)

    print("\nPreparing qrels...")
    qrels_info = prepare_qrels()
    print(qrels_info)

    print("\nPreparing documents...")
    docs_info = prepare_docs(limit=limit)
    print(docs_info)

    dataset_info = {
        "dataset_id": IRDS_DATASET_ID,
        "expected_docs_count": EXPECTED_DOCS_COUNT,
        "expected_queries_count": EXPECTED_QUERIES_COUNT,
        **queries_info,
        **qrels_info,
        **docs_info,
    }

    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)

    print("\nDataset info saved to:", DATASET_INFO_PATH)
    print("\nDataset preparation completed successfully.")

    return dataset_info