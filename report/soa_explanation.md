### Add this section after the `Retrieval Services` section

**Post-Retrieval Analysis Services**

* `services/result_clustering_service.py` — takes the retrieved documents returned by the selected retrieval model and groups them into topic-based clusters. It uses TF-IDF vectors over the retrieved results and KMeans clustering. This service does not replace BM25, TF-IDF, or Hybrid Search. It analyzes the results after retrieval.
* `services/topic_detection_service.py` — extracts the main topic terms from the query and retrieved documents using TF-IDF term weighting. If the selected retrieval model is BM25, then BM25 retrieves the documents first, and TF-IDF is used only as a topic extraction method over the retrieved results.

These services improve result explainability and usability. They do not directly change Precision@10, Recall@100, MAP@100, or nDCG@10 because they are analysis features, not ranking models.

---

### Replace the `How the UI calls the services` section with this version

## How the UI calls the services

`app.py` does not implement retrieval logic itself. It works as a Streamlit gateway that coordinates services.

It:

1. Lazily constructs each retrieval, refinement, and analysis service once via `st.cache_resource`:

   * `load_tfidf_service`
   * `load_bm25_service`
   * `load_hybrid_serial_service`
   * `load_hybrid_parallel_service`
   * `load_refined_corrected_service`
   * `load_refined_expanded_service`
   * `load_query_refinement_service`
   * `load_topic_detection_service`

2. Maps the model selected in the sidebar to the matching retrieval service through `get_search_service(model_key)`.

3. Calls `service.search(...)` with the parameters the user set:

   * top-k
   * `k1`
   * `b`
   * hybrid alpha
   * hybrid weights
   * RRF `k`

4. Renders the retrieved documents with `render_result_card(...)`.

5. If Topic Detection is enabled, it calls `TopicDetectionService` after retrieval and displays the detected topic terms.

6. If Result Clustering is enabled, it calls `ResultClusteringService` after retrieval and displays the clusters of retrieved documents.

The important architectural point is that Topic Detection and Result Clustering are post-retrieval analysis services. The selected retrieval model retrieves the documents first, then these services analyze the returned result set.

---

### Add this paragraph to the maintainability section

The addition of Result Clustering and Topic Detection demonstrates the benefit of the SOA design. These features were added as independent services without rewriting BM25, TF-IDF, Hybrid Search, or the evaluation engine. The UI only needed to call the new services after retrieval, which keeps the retrieval logic separate from the analysis logic.

---

### Update the final references if scripts are mentioned

Use the new organized paths:

```text
scripts/evaluate_models.py
scripts/generate_evaluation_charts.py
scripts/build_tfidf_index.py
scripts/build_bm25_index.py
scripts/build_embedding_index.py
scripts/prepare_dataset.py
```
