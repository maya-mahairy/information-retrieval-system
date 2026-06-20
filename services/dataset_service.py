from datasets import load_dataset

from config import (
    IRDS_DATASET_ID,
    HF_DATASET_ID,
    HF_CACHE_DIR,
    EXPECTED_DOCS_COUNT,
    EXPECTED_QUERIES_COUNT,
)


def _load_hf_config(config_name: str, streaming: bool = False):
    data = load_dataset(
        HF_DATASET_ID,
        config_name,
        cache_dir=str(HF_CACHE_DIR),
        streaming=streaming,
        trust_remote_code=True,
    )

    if hasattr(data, "keys"):
        if config_name in data:
            return data[config_name]
        first_key = list(data.keys())[0]
        return data[first_key]

    return data


def load_queries():
    return _load_hf_config("queries", streaming=False)


def load_qrels():
    return _load_hf_config("qrels", streaming=False)


def load_docs_stream():
    return _load_hf_config("docs", streaming=True)


def get_unique_qrel_query_ids(qrels_dataset):
    query_ids = set()

    for row in qrels_dataset:
        query_ids.add(str(row["query_id"]))

    return query_ids


def print_dataset_summary():
    print("=" * 70)
    print("IR DATASET SUMMARY")
    print("=" * 70)

    print("Official ir_datasets ID:", IRDS_DATASET_ID)
    print("HuggingFace mirror:", HF_DATASET_ID)
    print("HF cache folder:", HF_CACHE_DIR)

    print("\nExpected documents count:", EXPECTED_DOCS_COUNT)
    print("Expected queries count:", EXPECTED_QUERIES_COUNT)

    print("\n" + "-" * 70)
    print("Loading queries")
    print("-" * 70)
    queries = load_queries()
    print("Queries count:", len(queries))
    print("First query sample:")
    print(dict(queries[0]))

    print("\n" + "-" * 70)
    print("Loading qrels")
    print("-" * 70)
    qrels = load_qrels()
    print("Qrels count:", len(qrels))
    print("First qrel sample:")
    print(dict(qrels[0]))

    print("\n" + "-" * 70)
    print("Evaluation query count check")
    print("-" * 70)
    unique_qrel_queries = get_unique_qrel_query_ids(qrels)
    print("Unique query IDs in qrels:", len(unique_qrel_queries))
    print("Important: all these queries must be used in evaluation.")
    print("First 10 qrel query IDs:", sorted(list(unique_qrel_queries))[:10])

    print("\n" + "-" * 70)
    print("Loading one document sample using streaming")
    print("-" * 70)
    docs_stream = load_docs_stream()
    first_doc = next(iter(docs_stream))
    print("First document sample:")
    print(dict(first_doc))

    print("\nDataset check completed successfully.")