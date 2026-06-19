# Information Retrieval System — IR Project 2026

This project is a Python-based Information Retrieval system built for the IR course project.
It uses the BEIR / Webis Touché 2020 dataset and compares multiple retrieval models using standard IR evaluation metrics.

## Project Overview

The system allows the user to search a large argument retrieval dataset through a Streamlit interface.
It supports basic retrieval models, hybrid retrieval models, query refinement, evaluation, and charts.

Main goals:

* Load and preprocess a real IR dataset.
* Build searchable indexes.
* Implement multiple retrieval models.
* Compare models using official qrels.
* Provide a user interface for searching and analysis.
* Organize the project using a Service Oriented Architecture style.

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
The qrels contain graded relevance judgments, so metrics such as nDCG are important.

## Implemented Retrieval Models

The project implements the following retrieval models:

1. TF-IDF Search
2. BM25 Search
3. Hybrid Serial Search
4. Hybrid Parallel Search
5. BM25 Refined Corrected
6. BM25 Refined Expanded

### TF-IDF

TF-IDF is used as a classical vector space baseline.
It transforms documents and queries into weighted term vectors and ranks documents using cosine similarity.

### BM25

BM25 is the main lexical retrieval model.
It uses term frequency, inverse document frequency, and document length normalization.

The UI allows controlling BM25 parameters:

```text
k1
b
```

### Hybrid Serial

Hybrid Serial uses BM25 first to retrieve candidate documents, then reranks those candidates using Sentence-BERT embeddings.

This combines lexical retrieval with semantic reranking.

### Hybrid Parallel

Hybrid Parallel runs BM25 and TF-IDF independently, then combines their rankings using Weighted Reciprocal Rank Fusion.

### Query Refinement

Query Refinement supports:

* Spelling correction
* Synonym expansion
* Corrected BM25 search
* Expanded BM25 search

Example:

```text
Original Query: Should techers get tenur?
Corrected Query: should teachers get tenure
```

## Extra Feature

The implemented extra feature is:

```text
Semantic Representation / Vector Store / Embedding-Based Reranking
```

The project uses Sentence-BERT embeddings to represent text semantically and applies embedding reranking in Hybrid Serial Search.

Because of local hardware limitations, the full embedding vector store was heavy to build for all documents, so the embedding representation was demonstrated and then integrated practically into Hybrid Serial reranking over BM25 candidates.

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

## Final Results Summary

| Model                  | Precision@10 | Recall@100 | MAP@100 | nDCG@10 | Mean Time |
| ---------------------- | -----------: | ---------: | ------: | ------: | --------: |
| TF-IDF                 |       0.1408 |     0.1804 |  0.0349 |  0.0535 |    0.1767 |
| BM25                   |       0.6306 |     0.4333 |  0.2432 |  0.3408 |    0.0159 |
| Hybrid Serial          |       0.6306 |     0.4333 |  0.2438 |  0.3417 |    2.6539 |
| Hybrid Parallel        |       0.6184 |     0.4333 |  0.2387 |  0.3299 |    0.1018 |
| BM25 Refined Corrected |       0.6041 |     0.4147 |  0.2307 |  0.3258 |    0.0221 |
| BM25 Refined Expanded  |       0.5653 |     0.4014 |  0.2145 |  0.3142 |    0.0229 |

## Results Analysis

BM25 achieved the best practical balance between quality and speed.

Hybrid Serial achieved the best MAP@100 and nDCG@10, but it was much slower because it uses embedding reranking.

Hybrid Parallel successfully implemented rank fusion using BM25 and TF-IDF, but it did not outperform BM25 because TF-IDF was weaker on this dataset.

Query Refinement is useful when the user query contains spelling mistakes, but on the official clean qrels queries it does not outperform standard BM25.

## Service Oriented Architecture

The project is organized using a Service Oriented Architecture style.

Main services:

* Dataset Service
* Preprocessing Service
* Dataset Preparation Service
* TF-IDF Search Service
* BM25 Search Service
* Hybrid Serial Search Service
* Hybrid Parallel Search Service
* Query Refinement Service
* Refined BM25 Search Service
* Evaluation Service
* Chart Service
* Streamlit UI Gateway

Architecture documentation is available in:

```text
report/architecture_diagram.md
report/soa_explanation.md
report/developer_guide.md
```

## Project Structure

```text
IR_Project_2026/
│
├── app.py
├── config.py
├── prepare_dataset.py
├── build_tfidf_index.py
├── build_bm25_index.py
├── build_embedding_index.py
├── evaluate_models.py
├── generate_evaluation_charts.py
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
│   ├── evaluation_service.py
│   └── evaluation_chart_service.py
│
├── data/
│   ├── processed/
│   └── indexes/
│
├── outputs/
│   ├── evaluation/
│   └── charts/
│
└── report/
    ├── architecture_diagram.md
    ├── soa_explanation.md
    ├── developer_guide.md
    └── screenshots/
```

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

## Team Work Division



| Member   | Responsibility                                |
| -------- | --------------------------------------------- |
| Shdn | Dataset preparation and preprocessing         |
| Khaled | TF-IDF and BM25 indexing/search               |
| Maya | Hybrid retrieval and query refinement         |
| Mahmod | Evaluation and charts                         |
| Sami | Streamlit UI, documentation, and final report |

## Conclusion

This project implements a complete IR pipeline starting from dataset preparation and preprocessing, then indexing and retrieval, then evaluation and visualization through a Streamlit dashboard.

The final results show that BM25 is the best practical model due to its strong ranking quality and fast query time, while Hybrid Serial gives slightly better ranking quality at the cost of much higher execution time.
