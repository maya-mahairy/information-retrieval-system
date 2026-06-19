from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Official IR dataset ID for the report
IRDS_DATASET_ID = "beir/webis-touche2020"

# HuggingFace mirror of the same ir-datasets dataset
HF_DATASET_ID = "irds/beir_webis-touche2020"

# Counts documented for this dataset
EXPECTED_DOCS_COUNT = 382545
EXPECTED_QUERIES_COUNT = 49

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"
HF_CACHE_DIR = DATA_DIR / "hf_cache"

for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEXES_DIR, HF_CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# Offline Data Preparation outputs (processed docs/queries/qrels)
PROCESSED_DOCS_PATH = PROCESSED_DIR / "processed_docs.jsonl"
PROCESSED_QUERIES_PATH = PROCESSED_DIR / "processed_queries.jsonl"
QRELS_PATH = PROCESSED_DIR / "qrels.jsonl"
DATASET_INFO_PATH = PROCESSED_DIR / "dataset_info.json"


# TF-IDF index artifacts and the shared document store
TFIDF_VECTORIZER_PATH = INDEXES_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = INDEXES_DIR / "tfidf_matrix.npz"
DOCUMENT_STORE_DB_PATH = INDEXES_DIR / "document_store.sqlite"

# BM25 index artifacts
BM25_COUNT_VECTORIZER_PATH = INDEXES_DIR / "bm25_count_vectorizer.joblib"
BM25_COUNT_MATRIX_PATH = INDEXES_DIR / "bm25_count_matrix_csc.npz"
BM25_DOC_LENGTHS_PATH = INDEXES_DIR / "bm25_doc_lengths.npy"
BM25_IDF_PATH = INDEXES_DIR / "bm25_idf.npy"
BM25_METADATA_PATH = INDEXES_DIR / "bm25_metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MATRIX_PATH = INDEXES_DIR / "embedding_matrix.npy"
EMBEDDING_METADATA_PATH = INDEXES_DIR / "embedding_metadata.json"

# We limit text length before embedding to keep CPU processing practical.
# The title is kept, and the beginning of the argument text usually contains useful context.
EMBEDDING_MAX_CHARS = 1500

OUTPUTS_DIR = BASE_DIR / "outputs"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

EVALUATION_SUMMARY_PATH = EVALUATION_DIR / "evaluation_summary.json"
EVALUATION_PER_QUERY_PATH = EVALUATION_DIR / "evaluation_per_query.csv"
EVALUATION_RUN_DETAILS_PATH = EVALUATION_DIR / "evaluation_run_details.jsonl"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

CHARTS_DIR = OUTPUTS_DIR / "charts"

PRECISION_CHART_PATH = CHARTS_DIR / "precision_at_10_comparison.png"
RECALL_CHART_PATH = CHARTS_DIR / "recall_at_100_comparison.png"
MAP_CHART_PATH = CHARTS_DIR / "map_at_100_comparison.png"
NDCG_CHART_PATH = CHARTS_DIR / "ndcg_at_10_comparison.png"
TIME_CHART_PATH = CHARTS_DIR / "mean_query_time_comparison.png"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)


HYBRID_SERIAL_TUNING_SUMMARY_PATH = EVALUATION_DIR / "hybrid_serial_tuning_summary.json"
HYBRID_SERIAL_TUNING_CSV_PATH = EVALUATION_DIR / "hybrid_serial_tuning_summary.csv"
HYBRID_SERIAL_TUNING_PER_QUERY_PATH = EVALUATION_DIR / "hybrid_serial_tuning_per_query.csv"


HYBRID_PARALLEL_TUNING_SUMMARY_PATH = EVALUATION_DIR / "hybrid_parallel_tuning_summary.json"
HYBRID_PARALLEL_TUNING_CSV_PATH = EVALUATION_DIR / "hybrid_parallel_tuning_summary.csv"
HYBRID_PARALLEL_TUNING_PER_QUERY_PATH = EVALUATION_DIR / "hybrid_parallel_tuning_per_query.csv"