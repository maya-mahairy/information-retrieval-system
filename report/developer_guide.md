### Replace the `app.py` section with this updated version

## `app.py`

The Streamlit UI and the main entry point for interactive search. It builds a sidebar for dataset selection, execution mode, retrieval model, top-k, BM25 parameters, hybrid parameters, and post-retrieval analysis options.

The interface includes five tabs:

* Search
* Query Refinement
* Dataset
* Evaluation
* Charts

The Search tab now supports both retrieval and post-retrieval analysis. After a retrieval model returns results, the UI can optionally run Result Clustering and Topic Detection to make the retrieved results easier to understand.

* Loads each retrieval/refinement/analysis service once via `st.cache_resource`
  (`load_tfidf_service`, `load_bm25_service`, `load_hybrid_serial_service`,
  `load_hybrid_parallel_service`, `load_refined_corrected_service`,
  `load_refined_expanded_service`, `load_query_refinement_service`,
  `load_topic_detection_service`).
* `get_search_service(model_key)` maps a model key to its cached retrieval service.
* `search_with_model(...)` calls `service.search(...)` with the parameters appropriate for that model.
* `render_result_card(...)` renders one retrieved document, adapting to whichever score fields are present.
* `render_topic_detection(...)` extracts and displays the main topic terms from the query and retrieved results.
* `render_result_clusters(...)` groups retrieved documents into topic-based clusters using the clustering service.
* `render_dataset_info()`, `render_evaluation_summary()`, and `render_charts()` read static JSON/CSV/PNG outputs. They do not trigger preparation, index building, or evaluation themselves.

Depends on: retrieval services, query refinement service, post-retrieval analysis services, and `config.py` paths.

Called by: the Streamlit runtime:

```bash
streamlit run app.py
```

---

### Add this section after `services/refined_bm25_search_service.py`

## `services/result_clustering_service.py`

`ResultClusteringService` is a post-retrieval analysis service. It does not retrieve documents itself. Instead, it takes the results returned by a retrieval model such as BM25, TF-IDF, Hybrid Serial, Hybrid Parallel, or Refined BM25.

The service builds TF-IDF vectors over the retrieved documents, then uses KMeans to group similar results into topic-based clusters.

Main purpose:

* Organize retrieved documents into understandable groups.
* Help the user see subtopics inside the result set.
* Improve usability and explainability without changing the original retrieval score.

Example output for the query:

```text
Should teachers get tenure?
```

Possible clusters:

```text
Cluster 1: tenures / teachers / fired
Cluster 2: tenure / teacher / teacher tenure
Cluster 3: teachers / tenure / school
```

Depends on: `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.cluster.KMeans`.

Called by: `app.py` through `render_result_clusters(...)`.

Tested by:

```bash
python tests/test_result_clustering.py
```

---

### Add this section after `services/result_clustering_service.py`

## `services/topic_detection_service.py`

`TopicDetectionService` is a post-retrieval analysis service. It extracts the main topic terms from the user query and the retrieved documents.

The selected retrieval model runs first. For example, if the user selects BM25, then BM25 retrieves the documents. After that, Topic Detection uses TF-IDF term weighting over the query and retrieved documents to extract the most representative topic terms.

This means TF-IDF is used here as a topic extraction method, not necessarily as the retrieval model.

Example:

```text
Retrieval Model: BM25
Topic Extraction Method: TF-IDF term weighting over retrieved results
Detected Topic: teachers / teacher / tenure / tenures
```

Main purpose:

* Explain the main topic of the result set.
* Help the user understand why the retrieved documents are related.
* Add an explainability layer after retrieval.

Depends on: `sklearn.feature_extraction.text.TfidfVectorizer`.

Called by: `app.py` through `render_topic_detection(...)`.

Tested by:

```bash
python tests/test_topic_detection.py
```

---

### Replace the script usage lines near the end with this updated version

## `scripts/evaluate_models.py`

CLI entry point for running evaluation. Parses `--models` and `--top-k`, then calls `services.evaluation_service.evaluate_models(...)`.

Usage:

```bash
python scripts/evaluate_models.py --models bm25 hybrid_serial --top-k 100
```

## `scripts/generate_evaluation_charts.py`

CLI entry point that calls `services.evaluation_chart_service.generate_evaluation_charts()` to regenerate the five comparison PNGs from the current `evaluation_summary.json`.

Usage:

```bash
python scripts/generate_evaluation_charts.py
```

---

### Add this section after `scripts/generate_evaluation_charts.py`

## Academic Notebook

## `notebooks/01_ir_project_demo_and_evaluation.ipynb`

This notebook does not implement new logic. It reuses the existing services and already-generated outputs to present the project academically, step by step, for course evaluation purposes.

It:

* Reads `data/processed/dataset_info.json` and shows `documents_count`, `queries_count`, `qrels_count`, and `unique_qrel_queries_count`.
* Calls `services.preprocessing_service.preprocess_text(...)` on a sample query to show normalization, tokenization, and cleaned text.
* Loads `BM25SearchService` and runs a sample search to show the top retrieved titles, scores, and stances.
* Loads `QueryRefinementService` and runs `refine_query(...)` on a misspelled query to show spelling correction and synonym expansion.
* Reads `outputs/evaluation/evaluation_summary.json` and displays it as a pandas table, showing all 6 evaluated models with `evaluated_queries = 49` and `failed_queries_count = 0`.
* Displays the five evaluation charts from `outputs/charts/`.
* Demonstrates `ResultClusteringService` and `TopicDetectionService` on the BM25 results obtained earlier in the notebook.

Depends on: `config.py` paths, the preprocessing service, the BM25 search service, the query refinement service, the post-retrieval analysis services, and the evaluation/chart outputs already produced by `scripts/evaluate_models.py` and `scripts/generate_evaluation_charts.py`.

Called by: nobody. It is opened manually by a developer or evaluator through Jupyter, not part of the Streamlit runtime or any service-to-service call chain.

Usage:

```bash
conda activate ir_env
jupyter notebook
```
