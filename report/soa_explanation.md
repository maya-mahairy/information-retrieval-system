# Service Oriented Architecture (SOA) — Explanation

This project is organized as a set of independent, single-purpose services
under `services/`, coordinated by thin entry points (`app.py`,
`evaluate_models.py`, `generate_evaluation_charts.py`, `build_*.py`,
`prepare_dataset.py`). This document explains what each service does, why
that separation is useful, and how the pieces connect at runtime.

## What each service does

**Offline Data Preparation**
- `services/dataset_service.py` — loads the raw dataset (docs, queries, qrels)
  from the HuggingFace mirror of `beir/webis-touche2020`.
- `services/preprocessing_service.py` — normalizes, tokenizes, lemmatizes,
  and builds the searchable text shared by documents and queries.
- `services/dataset_preparation_service.py` — orchestrates loading +
  preprocessing and writes the processed JSONL files used by every
  downstream service.
- `build_tfidf_index.py`, `build_bm25_index.py`, `build_embedding_index.py` —
  one-time scripts that turn the processed data into the index artifacts
  (TF-IDF matrix, BM25 statistics, embedding vectors) and the SQLite
  document store.

**Query Refinement**
- `services/query_refinement_service.py` — takes a raw user query and
  produces a spelling-corrected version and a synonym-expanded version,
  using the BM25 vocabulary as its source of truth. It does not retrieve
  documents itself; it only rewrites the query text.

**Retrieval Services**
- `services/tfidf_search_service.py` — TF-IDF + cosine similarity search.
- `services/bm25_search_service.py` — BM25 scoring with tunable `k1`/`b`.
- `services/hybrid_serial_search_service.py` — BM25 retrieves candidates,
  then a sentence-embedding model reranks them (BM25 → rerank pipeline).
- `services/hybrid_parallel_search_service.py` — BM25 and TF-IDF run
  independently and are merged with Weighted Reciprocal Rank Fusion.
- `services/refined_bm25_search_service.py` — composes Query Refinement
  with BM25 search: it refines the query first, then delegates to
  `BM25SearchService`.

**Indexes and Document Store**
- Not a Python service but a shared resource layer: the TF-IDF/BM25/embedding
  artifacts in `data/indexes/` and the `document_store.sqlite` database. Every
  retrieval service reads from these artifacts and never recomputes them —
  that work happens once, offline, in the `build_*.py` scripts.

**Evaluation and Charts**
- `services/evaluation_service.py` — runs a retrieval model over every query
  in the qrels, computes Precision@10, Recall@100, MAP@100, and nDCG@10, and
  saves the summary/per-query/run-detail outputs.
- `services/evaluation_chart_service.py` — reads the evaluation summary and
  renders the comparison bar charts (precision, recall, MAP, nDCG, mean
  query time).
- `evaluate_models.py` and `generate_evaluation_charts.py` — CLI entry points
  that call the two services above.

## Why this improves maintainability

- **Single responsibility per service.** Each retrieval service only knows
  how to rank documents for its own algorithm. Changing how BM25 scores
  documents never touches TF-IDF, hybrid fusion, or the UI.
- **Composition over duplication.** `HybridSerialSearchService`,
  `HybridParallelSearchService`, and `RefinedBM25SearchService` all reuse
  `BM25SearchService`/`TFIDFSearchService` instead of re-implementing
  scoring. A bug fix or tuning change to BM25 automatically benefits every
  service that composes it.
- **Stable contracts.** Every search service exposes the same `search(...)`
  method shape and returns a list of result dicts with consistent keys
  (`doc_id`, `score`, `title`, ...). `app.py` and `evaluate_models.py` can
  call any of them through the same calling convention.
- **Offline/online separation.** Data preparation and index building are
  expensive, one-time steps kept entirely separate from the request-time
  search path. The UI and evaluation only ever *read* prebuilt artifacts,
  which keeps queries fast and keeps the heavy dataset/model logic out of
  the request path.
- **Testability.** Because each service is isolated with a narrow interface,
  each one has a focused test file (`test_bm25_search.py`,
  `test_tfidf_search.py`, `test_hybrid_serial_search.py`,
  `test_query_refinement.py`, etc.) that can validate it without spinning up
  the whole app.

## How the UI calls the services

`app.py` never implements retrieval logic itself. It:
1. Lazily constructs each search service once via `st.cache_resource`
   (`load_tfidf_service`, `load_bm25_service`, `load_hybrid_serial_service`,
   `load_hybrid_parallel_service`, `load_refined_corrected_service`,
   `load_refined_expanded_service`, `load_query_refinement_service`).
2. Maps the model selected in the sidebar to the matching service through
   `get_search_service(model_key)`.
3. Calls `service.search(...)` with the parameters the user set (top-k,
   `k1`/`b`, hybrid alpha/weights, RRF `k`) inside `search_with_model(...)`.
4. Renders whatever dict list comes back with `render_result_card(...)`,
   which adapts its display based on which score/metadata keys are present
   (BM25 score, hybrid score, fusion score, refinement details, etc.).

The Query Refinement tab calls `QueryRefinementService.refine_query(...)`
directly so the user can inspect corrections/expansions without running a
full search.

## How evaluation and charts are connected

1. `evaluate_models.py` is the CLI entry point. It calls
   `services.evaluation_service.evaluate_models(model_names, top_k)`.
2. For each model, `evaluation_service.get_search_service(...)` builds the
   same retrieval service classes used by the UI, and `run_search(...)`
   calls them with the model's standard parameters.
3. Results are scored against `qrels.jsonl` using Precision@10, Recall@100,
   MAP@100, and nDCG@10, then written to
   `outputs/evaluation/evaluation_summary.json` (plus per-query CSV and
   run-detail JSONL).
4. `generate_evaluation_charts.py` calls
   `services.evaluation_chart_service.generate_evaluation_charts()`, which
   reads that same `evaluation_summary.json` and renders the five PNG
   comparison charts into `outputs/charts/`.
5. `app.py`'s "Evaluation" and "Charts" tabs read those same summary and PNG
   files directly — they do not re-run evaluation. This keeps the
   expensive evaluation step decoupled from the interactive UI.

See [architecture_diagram.md](architecture_diagram.md) for the layered
diagram and [developer_guide.md](developer_guide.md) for what each file
contains in detail.
