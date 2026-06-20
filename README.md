# Information Retrieval System — IR Project 2026

This project is a Python-based Information Retrieval system built for the IR course project.

It uses the **BEIR / Webis Touché 2020** dataset and compares multiple retrieval models using standard IR evaluation metrics.
The system includes classical retrieval models, hybrid retrieval models, query refinement, post-retrieval analysis features, evaluation, charts, and a Streamlit user interface.

---

## Project Overview

The system allows the user to search a large argument retrieval dataset through a Streamlit interface.

It supports:

* Dataset preparation and preprocessing
* TF-IDF and BM25 indexing
* Multiple retrieval models
* Hybrid retrieval
* Query refinement
* Topic detection
* Search result clustering
* Evaluation using official qrels
* Visualization through charts
* Service Oriented Architecture style organization

Main goals:

* Load and preprocess a real IR dataset.
* Build searchable indexes.
* Implement multiple retrieval models.
* Compare models using official qrels.
* Provide a user interface for searching and analysis.
* Add extra IR features beyond basic search.
* Organize the project using a Service Oriented Architecture style.

---

## Dataset

Dataset used:

```text
BEIR / Webis Touché 2020
```

Dataset statistics:

```text
Documents: 382,545
Queries: 49
Qrels: 2,962
```

This dataset is suitable for argument retrieval and debate-style search.

The qrels contain graded relevance judgments, so metrics such as **nDCG** are important.

---

## Implemented Retrieval Models

The project implements the following retrieval models:

1. TF-IDF Search
2. BM25 Search
3. Hybrid Serial Search
4. Hybrid Parallel Search
5. BM25 Refined Corrected
6. BM25 Refined Expanded

---

## TF-IDF

TF-IDF is used as a classical vector space baseline.

It transforms documents and queries into weighted term vectors and ranks documents using cosine similarity.

---

## BM25

BM25 is the main lexical retrieval model.

It uses:

* Term frequency
* Inverse document frequency
* Document length normalization

The UI allows controlling BM25 parameters:

```text
k1
b
```

BM25 achieved the best practical balance between ranking quality and query speed in the final evaluation.

---

## Hybrid Serial Search

Hybrid Serial uses BM25 first to retrieve candidate documents, then reranks those candidates using Sentence-BERT embeddings.

This combines:

```text
Lexical retrieval + Semantic reranking
```

Hybrid Serial achieved the best MAP@100 and nDCG@10, but it was slower because it uses embedding-based reranking.

---

## Hybrid Parallel Search

Hybrid Parallel runs BM25 and TF-IDF independently, then combines their rankings using **Weighted Reciprocal Rank Fusion**.

This model demonstrates rank fusion between multiple retrieval systems.

---

## Query Refinement

Query Refinement supports:

* Spelling correction
* Synonym expansion
* Corrected BM25 search
* Expanded BM25 search

Example:

```text
Original Query: Should techers get tenur?
Corrected Query: should teachers get tenure
Expanded Query: should teachers get tenure educator instructor job security protection
```

Query refinement is useful when the user query contains spelling mistakes or when synonym expansion can improve recall.

---

## Extra Features

The project implements several extra IR features beyond basic search.

### 1. Semantic Representation / Vector Store / Embedding-Based Reranking

The project uses Sentence-BERT embeddings to represent text semantically and applies embedding reranking in Hybrid Serial Search.

Because of local hardware limitations, the full embedding vector store was heavy to build for all documents, so the embedding representation was demonstrated and then integrated practically into Hybrid Serial reranking over BM25 candidates.

### 2. Search Result Clustering

The system includes a post-retrieval clustering feature.

After a retrieval model returns the top documents, the system groups the retrieved results into topic-based clusters using TF-IDF vectors and KMeans clustering.

This helps the user understand the main subtopics inside the result set instead of reading one flat list of documents.

Example:

```text
Query: Should teachers get tenure?

Cluster 1: tenures / teachers / fired
Cluster 2: tenure / teacher / teacher tenure
Cluster 3: teachers / tenure / school
```

### 3. Topic Detection

The system includes a topic detection feature.

After retrieving documents using the selected retrieval model, the system extracts the main topic terms from the query and retrieved documents using TF-IDF term weighting.

Example:

```text
Retrieval Model: BM25
Topic Extraction Method: TF-IDF term weighting over retrieved results
Detected Topic: teachers / teacher / tenure / tenures
```

This feature is a post-retrieval analysis step.
It does not replace the retrieval model. For example, BM25 can retrieve the documents, then TF-IDF can be used only to extract topic terms from the retrieved results.

---

## Evaluation

The project evaluates all retrieval models using the official qrels.

Metrics used:

```text
Precision@10
Recall@100
MAP@100
nDCG@10
Mean Query Time
```

Final evaluation was performed on:

```text
Evaluated queries: 49
Failed queries: 0
```

---

## Final Results Summary

| Model                  | Precision@10 | Recall@100 | MAP@100 | nDCG@10 | Mean Time |
| ---------------------- | -----------: | ---------: | ------: | ------: | --------: |
| TF-IDF                 |       0.1408 |     0.1804 |  0.0349 |  0.0535 |    0.1767 |
| BM25                   |       0.6306 |     0.4333 |  0.2432 |  0.3408 |    0.0159 |
| Hybrid Serial          |       0.6306 |     0.4333 |  0.2438 |  0.3417 |    2.6539 |
| Hybrid Parallel        |       0.6184 |     0.4333 |  0.2387 |  0.3299 |    0.1018 |
| BM25 Refined Corrected |       0.6041 |     0.4147 |  0.2307 |  0.3258 |    0.0221 |
| BM25 Refined Expanded  |       0.5653 |     0.4014 |  0.2145 |  0.3142 |    0.0229 |

---

## Results Analysis

BM25 achieved the best practical balance between quality and speed.

Hybrid Serial achieved the best MAP@100 and nDCG@10, but it was much slower because it uses embedding reranking.

Hybrid Parallel successfully implemented rank fusion using BM25 and TF-IDF, but it did not outperform BM25 because TF-IDF was weaker on this dataset.

Query Refinement is useful when the user query contains spelling mistakes, but on the official clean qrels queries it does not outperform standard BM25.

Result Clustering and Topic Detection are post-retrieval analysis features.
They improve explainability and usability by helping the user understand the retrieved results more clearly.

---

## Service Oriented Architecture

The project is organized using a Service Oriented Architecture style.

The Streamlit UI works as a gateway that calls independent services for dataset loading, preprocessing, retrieval, refinement, evaluation, chart generation, and result analysis.

Main services:

* Dataset Service
* Preprocessing Service
* Dataset Preparation Service
* TF-IDF Index Service
* TF-IDF Search Service
* BM25 Index Service
* BM25 Search Service
* Embedding Index Service
* Embedding Search Service
* Hybrid Serial Search Service
* Hybrid Parallel Search Service
* Query Refinement Service
* Refined BM25 Search Service
* Result Clustering Service
* Topic Detection Service
* Evaluation Service
* Chart Service
* Streamlit UI Gateway

Architecture documentation is available in:

```text
report/architecture_diagram.md
report/soa_explanation.md
report/developer_guide.md
```

---

## Project Structure

```text
IR_Project_2026/
│
├── app.py
├── config.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── services/
│   ├── dataset_service.py
│   ├── preprocessing_service.py
│   ├── dataset_preparation_service.py
│   ├── tfidf_index_service.py
│   ├── tfidf_search_service.py
│   ├── bm25_index_service.py
│   ├── bm25_search_service.py
│   ├── embedding_index_service.py
│   ├── embedding_search_service.py
│   ├── hybrid_serial_search_service.py
│   ├── hybrid_parallel_search_service.py
│   ├── query_refinement_service.py
│   ├── refined_bm25_search_service.py
│   ├── result_clustering_service.py
│   ├── topic_detection_service.py
│   ├── evaluation_service.py
│   └── evaluation_chart_service.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── build_tfidf_index.py
│   ├── build_bm25_index.py
│   ├── build_embedding_index.py
│   ├── evaluate_models.py
│   ├── generate_evaluation_charts.py
│   ├── tune_hybrid_serial.py
│   ├── tune_hybrid_parallel.py
│   └── check_dataset.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_tfidf_search.py
│   ├── test_bm25_search.py
│   ├── test_bm25_parameters.py
│   ├── test_embedding_search.py
│   ├── test_hybrid_serial_search.py
│   ├── test_hybrid_parallel_search.py
│   ├── test_query_refinement.py
│   ├── test_refined_bm25_search.py
│   ├── test_result_clustering.py
│   └── test_topic_detection.py
│
├── data/
│   ├── processed/
│   └── indexes/
│
├── outputs/
│   ├── evaluation/
│   └── charts/
│
├── notebooks/
│   └── 01_ir_project_demo_and_evaluation.ipynb
│
└── report/
    ├── architecture_diagram.md
    ├── soa_explanation.md
    ├── developer_guide.md
    └── screenshots/
```

---

## Screenshots

Important screenshots are stored in:

```text
report/screenshots/
```

Examples:

```text
01_evaluation_summary.png
02_precision_at_10_chart.png
03_recall_at_100_chart.png
04_map_at_100_chart.png
05_ndcg_at_10_chart.png
06_mean_query_time_chart.png
07_search_bm25.png
08_query_refinement.png
09_bm25_refined_corrected.png
10_hybrid_parallel.png
11_hybrid_serial.png
12_dataset_tab.png
13_architecture_diagram.png
14_result_clustering.png
15_topic_detection.png
```

---

## Jupyter Notebook

The project includes one academic notebook:

```text
notebooks/01_ir_project_demo_and_evaluation.ipynb
```

This notebook is **not a replacement for the Streamlit UI**. It is an academic file used to demonstrate, verify, and explain the project pipeline step by step for course evaluation purposes: dataset summary, preprocessing demo, BM25 search demo, query refinement demo, evaluation summary, charts, result clustering, and topic detection.

It only reads existing services and already-generated outputs (processed dataset, prebuilt indexes, evaluation summary, charts). It does not modify any service code and does not run a full evaluation.

To run it:

```bash
conda activate ir_env
jupyter notebook
```

Then open:

```text
notebooks/01_ir_project_demo_and_evaluation.ipynb
```

---

## How to Run

Create and activate the environment:

```bash
conda activate ir_env
```

Install requirements:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python scripts/prepare_dataset.py
```

Build indexes:

```bash
python scripts/build_tfidf_index.py
python scripts/build_bm25_index.py
```

Optional embedding index:

```bash
python scripts/build_embedding_index.py --limit 1000 --batch-size 32
```

Run full evaluation:

```bash
python scripts/evaluate_models.py --models tfidf bm25 hybrid_serial hybrid_parallel bm25_refined_corrected bm25_refined_expanded --top-k 100
```

Generate charts:

```bash
python scripts/generate_evaluation_charts.py
```

Run Streamlit UI:

```bash
streamlit run app.py
```

---

## How to Test

Run selected feature tests:

```bash
python tests/test_query_refinement.py
python tests/test_refined_bm25_search.py
python tests/test_result_clustering.py
python tests/test_topic_detection.py
```

Run syntax check for the Streamlit app:

```bash
python -m py_compile app.py
```

---

## Important Notes

The generated dataset files, indexes, SQLite document store, evaluation files, and charts can be large.

Before uploading to GitHub, large generated files should be excluded using `.gitignore`, especially:

```text
data/
outputs/
__pycache__/
*.pyc
```

The project can regenerate these files using the preparation, indexing, evaluation, and chart scripts.

---

## Team Work Division

| Member | Responsibility                                                    |
| ------ | ----------------------------------------------------------------- |
| Shdn   | Dataset preparation, dataset loading, and text preprocessing                             |
| Khaled | TF-IDF indexing/search and BM25 indexing/search                                   |
| Maya   | Evaluation metrics, model comparison, chart generation  ,Result Clustering service, and Topic Detection service  |
| Mahmod | Hybrid retrieval, query refinement,                                  |
| Sami   | Streamlit UI, documentation, report, and result analysis features |

---

## Conclusion

This project implements a complete Information Retrieval pipeline starting from dataset preparation and preprocessing, then indexing and retrieval, then evaluation and visualization through a Streamlit dashboard.

The final results show that BM25 is the best practical model due to its strong ranking quality and fast query time, while Hybrid Serial gives slightly better ranking quality at the cost of much higher execution time.

The additional features, including query refinement, embedding-based reranking, result clustering, and topic detection, make the system more useful, explainable, and professional for both users and evaluators.
/*

cd C:\Users\ASUSD\IR_WORK\IR_Project_2026
conda activate ir_env
python -m py_compile app.py
python tests\test_preprocessing.py
python tests\test_tfidf_search.py
python tests\test_bm25_search.py
python tests\test_bm25_parameters.py
python tests\test_embedding_search.py
python tests\test_hybrid_serial_search.py
python tests\test_hybrid_parallel_search.py
python tests\test_query_refinement.py
python tests\test_refined_bm25_search.py
python tests\test_result_clustering.py
python tests\test_topic_detection.py
streamlit run app.py

*/