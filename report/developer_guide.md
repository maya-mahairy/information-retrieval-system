# Developer Guide — File Responsibilities

A practical, file-by-file reference for anyone extending this project. Each
entry covers what the file does, what it depends on, and what calls it.

## `app.py`

The Streamlit UI and the only entry point for interactive search. Builds a
sidebar (dataset, execution mode, model, top-k, BM25/hybrid parameters),
exposes a query box, and five tabs: Search, Query Refinement, Dataset,
Evaluation, Charts.

- Loads each retrieval/refinement service once via `st.cache_resource`
  (`load_tfidf_service`, `load_bm25_service`, `load_hybrid_serial_service`,
  `load_hybrid_parallel_service`, `load_refined_corrected_service`,
  `load_refined_expanded_service`, `load_query_refinement_service`).
- `get_search_service(model_key)` maps a model key to its cached service.
- `search_with_model(...)` calls `service.search(...)` with the parameters
  appropriate for that model.
- `render_result_card(...)` renders one result, adapting to whichever score
  fields are present (`score`, `hybrid_score`, `fusion_score`, refinement
  metadata).
- `render_dataset_info()`, `render_evaluation_summary()`, `render_charts()`
  read static JSON/CSV/PNG outputs — they never trigger preparation,
  index building, or evaluation themselves.

Depends on: every retrieval/refinement service, `config.py` paths.
Called by: the Streamlit runtime (`streamlit run app.py`).

## `config.py`

Single source of truth for every file path and dataset constant used across
the project: dataset IDs, processed-data paths, index artifact paths, the
document store path, evaluation output paths, and chart output paths. Also
ensures the relevant directories exist (`mkdir(parents=True, exist_ok=True)`)
on import.

Depends on: nothing project-specific (just `pathlib`).
Called by: every other module in this project.

## `services/dataset_service.py`

Loads the raw BEIR Webis-Touché2020 dataset from its HuggingFace mirror:
`load_queries()`, `load_qrels()`, `load_docs_stream()` (streaming, to avoid
downloading all 382K docs just for inspection), and
`get_unique_qrel_query_ids(...)`. `print_dataset_summary()` is a manual
sanity-check entry point (used by `check_dataset.py`).

Depends on: `datasets` (HuggingFace), `config.py`.
Called by: `services/dataset_preparation_service.py`.

## `services/preprocessing_service.py`

The single shared text pipeline used for both documents and queries, so
their representations stay compatible:
- `normalize_text` — HTML unescape, lowercase, strip URLs/punctuation.
- `tokenize_text` — regex word tokenization.
- `preprocess_tokens` — stopword removal + lemmatization (or stemming).
- `preprocess_text` — runs the full pipeline, returns original/normalized/
  tokens/cleaned text.
- `build_document_search_text` / `preprocess_document` — document-specific
  wrapper (title + text).
- `build_query_search_text` / `preprocess_query` — query-specific wrapper.

Depends on: `nltk` (stopwords, `WordNetLemmatizer`, `PorterStemmer`).
Called by: `dataset_preparation_service.py`, every search service (via
`preprocess_text`), `query_refinement_service.py` (via `normalize_text`/
`tokenize_text`).

## `services/dataset_preparation_service.py`

Orchestrates the offline preparation pipeline: `prepare_queries()`,
`prepare_qrels()`, `prepare_docs(limit=...)`, and `prepare_all_data(limit=...)`
which runs all three and writes `dataset_info.json`. Writes the processed
JSONL files that every retrieval/evaluation service ultimately reads
(directly, or through prebuilt indexes).

Depends on: `services/dataset_service.py`, `services/preprocessing_service.py`,
`config.py`.
Called by: `prepare_dataset.py`.

## `services/tfidf_search_service.py`

`TFIDFSearchService` — loads the prebuilt TF-IDF vectorizer and sparse
matrix, transforms a query into the same vector space, scores documents by
cosine similarity (dot product against the L2-normalized matrix), and joins
the top-k row ids back to metadata stored in `document_store.sqlite`.

Depends on: `data/indexes/tfidf_vectorizer.joblib`,
`data/indexes/tfidf_matrix.npz`, `document_store.sqlite`,
`preprocess_text`.
Called by: `app.py`, `HybridParallelSearchService`,
`services/evaluation_service.py`.

## `services/bm25_search_service.py`

`BM25SearchService` — loads the prebuilt term-count matrix (CSC format),
document lengths, IDF vector, and metadata (avg doc length, default
`k1`/`b`). `search(...)` accepts per-query `k1`/`b` overrides and computes
the standard BM25 formula directly over the sparse matrix columns for each
query term, then joins results to `document_store.sqlite`.

Depends on: `data/indexes/bm25_count_matrix_csc.npz`,
`bm25_doc_lengths.npy`, `bm25_idf.npy`, `bm25_metadata.json`,
`document_store.sqlite`, `preprocess_text`.
Called by: `app.py`, `HybridSerialSearchService`,
`HybridParallelSearchService`, `RefinedBM25SearchService`,
`services/evaluation_service.py`.

## `services/hybrid_serial_search_service.py`

`HybridSerialSearchService` — a two-stage (serial) pipeline: BM25 retrieves
`bm25_candidates` documents, then a `SentenceTransformer` embedding model
reranks only those candidates by cosine similarity against the query
embedding. The final score is `alpha * bm25_normalized + (1 - alpha) *
embedding_normalized`. Reranking only the BM25 shortlist keeps embedding
computation cheap on CPU.

Depends on: `BM25SearchService`, `sentence-transformers`
(`EMBEDDING_MODEL_NAME`), `document_store.sqlite`.
Called by: `app.py`, `services/evaluation_service.py`.

## `services/hybrid_parallel_search_service.py`

`HybridParallelSearchService` — runs BM25 and TF-IDF independently (in
parallel, conceptually) and fuses their rankings with Weighted Reciprocal
Rank Fusion: `score += weight / (rrf_k + rank)` for each list. Produces a
single ranked list annotated with both models' individual ranks/scores.

Depends on: `BM25SearchService`, `TFIDFSearchService`.
Called by: `app.py`, `services/evaluation_service.py`.

## `services/query_refinement_service.py`

`QueryRefinementService` — conservative query rewriting built on the BM25
vocabulary:
- `correct_token`/`correct_query` — manual correction map first, then a
  protected-terms allowlist, then a cautious `difflib` fuzzy match
  (`cutoff=0.92`) restricted to same-first-letter, similar-length
  candidates, to avoid corrupting valid queries.
- `expand_query` — adds a bounded set of curated synonyms per token.
- `refine_query` — runs both and returns original/corrected/expanded
  queries plus the corrections/added-terms detail used for UI display.

Depends on: `data/indexes/bm25_count_vectorizer.joblib` (for vocabulary),
`preprocessing_service.normalize_text`/`tokenize_text`.
Called by: `app.py` (Query Refinement tab), `RefinedBM25SearchService`.

## `services/refined_bm25_search_service.py`

`RefinedBM25SearchService(mode="corrected"|"expanded")` — composes
`QueryRefinementService` and `BM25SearchService`: refines the query first
(corrected or expanded form depending on `mode`), then searches with BM25
using the refined text. Annotates results with the original/refined query
and refinement details for display.

Depends on: `BM25SearchService`, `QueryRefinementService`.
Called by: `app.py` (as the two "BM25 Refined" model options),
`services/evaluation_service.py`.

## `services/evaluation_service.py`

The evaluation engine. Loads processed queries and qrels, then for each
model in the qrels query set:
- `get_search_service(model_name)` builds the corresponding retrieval
  service (same classes the UI uses).
- `run_search(...)` calls it with that model's standard parameters.
- Computes `precision_at_k`, `recall_at_k`, `average_precision_at_k`
  (MAP), and `ndcg_at_k` (with `dcg_at_k`/`ideal_dcg_at_k` helpers) per
  query, plus timing.
- `evaluate_model(...)` aggregates per-query metrics into a summary dict
  for one model; `evaluate_models(...)` runs this for a list of models and
  calls `save_evaluation_outputs(...)` to write the summary JSON, per-query
  CSV, and run-details JSONL defined in `config.py`.

Depends on: every retrieval service, `PROCESSED_QUERIES_PATH`,
`QRELS_PATH`, evaluation output paths from `config.py`.
Called by: `evaluate_models.py`.

## `services/evaluation_chart_service.py`

Reads `evaluation_summary.json` and renders one bar chart per metric
(Precision@10, Recall@100, MAP@100, nDCG@10, mean query time) using
matplotlib, saving each as a PNG under `outputs/charts/`.
`MODEL_DISPLAY_NAMES` controls the human-readable labels on the x-axis.

Depends on: `EVALUATION_SUMMARY_PATH` and the five chart output paths from
`config.py`, `matplotlib`.
Called by: `generate_evaluation_charts.py`.

## `evaluate_models.py`

CLI entry point for running evaluation. Parses `--models` (defaults to
`tfidf`, `bm25`, `hybrid_serial`, `hybrid_parallel`) and `--top-k` (default
100), then calls `services.evaluation_service.evaluate_models(...)`.

Usage: `python evaluate_models.py --models bm25 hybrid_serial --top-k 100`

## `generate_evaluation_charts.py`

CLI entry point that calls
`services.evaluation_chart_service.generate_evaluation_charts()` to
(re)generate the five comparison PNGs from the current
`evaluation_summary.json`. Run this after `evaluate_models.py` whenever the
summary changes.

Usage: `python generate_evaluation_charts.py`

---

See [architecture_diagram.md](architecture_diagram.md) for the layered
diagram and [soa_explanation.md](soa_explanation.md) for the design
rationale.
