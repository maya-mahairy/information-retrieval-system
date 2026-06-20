```mermaid
flowchart TD

    USER["User / Evaluator"]
    UI["Streamlit UI<br/>app.py"]

    USER --> UI

    subgraph ONLINE["Online Search Flow"]
        QR["Query Refinement<br/>query_refinement_service.py"]
        RET["Retrieval Services<br/>tfidf_search_service.py<br/>bm25_search_service.py<br/>hybrid_serial_search_service.py<br/>hybrid_parallel_search_service.py<br/>refined_bm25_search_service.py"]
        ANALYSIS["Post-Retrieval Analysis Services<br/>result_clustering_service.py<br/>topic_detection_service.py"]
        STORE["Indexes + Document Store<br/>data/indexes<br/>document_store.sqlite"]
    end

    subgraph OFFLINE["Offline Data Preparation Flow"]
        DS["Dataset Service<br/>dataset_service.py"]
        PREP["Preprocessing Service<br/>preprocessing_service.py"]
        DPREP["Dataset Preparation<br/>dataset_preparation_service.py"]
        BUILD["Index Building Scripts<br/>scripts/build_tfidf_index.py<br/>scripts/build_bm25_index.py<br/>scripts/build_embedding_index.py"]
    end

    subgraph EVALUATION["Evaluation and Reporting Flow"]
        EV["Evaluation Service<br/>evaluation_service.py<br/>scripts/evaluate_models.py"]
        CH["Chart Service<br/>evaluation_chart_service.py<br/>scripts/generate_evaluation_charts.py"]
        OUT["Outputs<br/>outputs/evaluation<br/>outputs/charts"]
    end

    subgraph ACADEMIC["Academic Demonstration Layer"]
        NB["Jupyter Notebook<br/>Academic Demo &amp; Evaluation<br/>notebooks/01_ir_project_demo_and_evaluation.ipynb"]
    end

    UI --> QR
    UI --> RET
    QR --> RET

    RET --> STORE
    STORE --> RET

    RET --> ANALYSIS
    ANALYSIS --> UI

    DS --> PREP
    PREP --> DPREP
    DPREP --> BUILD
    BUILD --> STORE

    RET --> EV
    EV --> CH
    EV --> OUT
    CH --> OUT

    UI --> OUT

    NB -.-> DS
    NB -.-> PREP
    NB -.-> RET
    NB -.-> OUT
    NB -.-> CH
    NB -.-> ANALYSIS
```

The dashed lines above show that the **Jupyter Notebook is not part of the Online Search Flow**. It does not sit between the user and the retrieval services, and it does not change ranking or evaluation results.

**Streamlit (`app.py`)** is the actual user-facing interface for searching, refining queries, and browsing results interactively.

The **Notebook** (`notebooks/01_ir_project_demo_and_evaluation.ipynb`) is a separate academic demonstration layer. It calls the same Dataset Service, Preprocessing Service, and Retrieval Services, and reads the same Evaluation Outputs and Charts, plus the Post-Retrieval Analysis Services (Result Clustering and Topic Detection), purely to document, verify, and re-present the project's pipeline and results for course evaluation — it does not serve end users and does not replace the Streamlit UI.

> **Note:** `report/screenshots/13_architecture_diagram.png` still reflects the diagram *before* this Academic Demonstration Layer was added. No tool to render the updated Mermaid diagram into a PNG was available in this environment, so the screenshot should be regenerated manually (e.g. by re-exporting this diagram from a Mermaid-compatible viewer) and replaced later.
