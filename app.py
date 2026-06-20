import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    DATASET_INFO_PATH,
    EVALUATION_SUMMARY_PATH,
    PRECISION_CHART_PATH,
    RECALL_CHART_PATH,
    MAP_CHART_PATH,
    NDCG_CHART_PATH,
    TIME_CHART_PATH,
)
from services.bm25_search_service import BM25SearchService
from services.tfidf_search_service import TFIDFSearchService
from services.hybrid_serial_search_service import HybridSerialSearchService
from services.hybrid_parallel_search_service import HybridParallelSearchService
from services.query_refinement_service import QueryRefinementService
from services.refined_bm25_search_service import RefinedBM25SearchService
from services.result_clustering_service import ResultClusteringService
from services.topic_detection_service import TopicDetectionService



st.set_page_config(
    page_title="IR Project 2026",
    page_icon="🔎",
    layout="wide",
)


MODEL_OPTIONS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "hybrid_serial": "Hybrid Serial",
    "hybrid_parallel": "Hybrid Parallel",
    "bm25_refined_corrected": "BM25 Refined Corrected",
    "bm25_refined_expanded": "BM25 Refined Expanded",
}


@st.cache_resource(show_spinner=False)
def load_tfidf_service():
    return TFIDFSearchService()


@st.cache_resource(show_spinner=False)
def load_bm25_service():
    return BM25SearchService()


@st.cache_resource(show_spinner=False)
def load_hybrid_serial_service():
    return HybridSerialSearchService()


@st.cache_resource(show_spinner=False)
def load_hybrid_parallel_service():
    return HybridParallelSearchService()


@st.cache_resource(show_spinner=False)
def load_refined_corrected_service():
    return RefinedBM25SearchService(mode="corrected")


@st.cache_resource(show_spinner=False)
def load_refined_expanded_service():
    return RefinedBM25SearchService(mode="expanded")


@st.cache_resource(show_spinner=False)
def load_query_refinement_service():
    return QueryRefinementService()

@st.cache_resource(show_spinner=False)
def load_topic_detection_service():
    return TopicDetectionService()

def load_json_file(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_search_service(model_key: str):
    if model_key == "tfidf":
        return load_tfidf_service()

    if model_key == "bm25":
        return load_bm25_service()

    if model_key == "hybrid_serial":
        return load_hybrid_serial_service()

    if model_key == "hybrid_parallel":
        return load_hybrid_parallel_service()

    if model_key == "bm25_refined_corrected":
        return load_refined_corrected_service()

    if model_key == "bm25_refined_expanded":
        return load_refined_expanded_service()

    raise ValueError(f"Unsupported model: {model_key}")


def search_with_model(
    model_key: str,
    query_text: str,
    top_k: int,
    k1: float,
    b: float,
    hybrid_serial_alpha: float,
    hybrid_serial_candidates: int,
    hybrid_parallel_bm25_weight: float,
    hybrid_parallel_tfidf_weight: float,
    rrf_k: int,
):
    service = get_search_service(model_key)

    if model_key == "tfidf":
        return service.search(
            query_text=query_text,
            top_k=top_k,
        )

    if model_key == "bm25":
        return service.search(
            query_text=query_text,
            top_k=top_k,
            k1=k1,
            b=b,
        )

    if model_key == "hybrid_serial":
        return service.search(
            query_text=query_text,
            top_k=top_k,
            bm25_candidates=hybrid_serial_candidates,
            k1=k1,
            b=b,
            alpha=hybrid_serial_alpha,
        )

    if model_key == "hybrid_parallel":
        return service.search(
            query_text=query_text,
            top_k=top_k,
            bm25_top_k=max(top_k, 100),
            tfidf_top_k=max(top_k, 100),
            rrf_k=rrf_k,
            bm25_weight=hybrid_parallel_bm25_weight,
            tfidf_weight=hybrid_parallel_tfidf_weight,
            k1=k1,
            b=b,
        )

    if model_key in {"bm25_refined_corrected", "bm25_refined_expanded"}:
        return service.search(
            query_text=query_text,
            top_k=top_k,
            k1=k1,
            b=b,
        )

    return []


def get_result_score(result: dict):
    if "hybrid_score" in result:
        return result.get("hybrid_score")

    if "fusion_score" in result:
        return result.get("fusion_score")

    return result.get("score", 0.0)


def render_result_card(result: dict, fallback_rank: int):
    rank = result.get("rank", fallback_rank)
    score = get_result_score(result)

    title = result.get("title", "Untitled Document")
    doc_id = result.get("doc_id", "")
    stance = result.get("stance", "")
    url = result.get("url", "")
    text = result.get("original_text", "")

    short_preview_length = 350
    short_preview = text[:short_preview_length]

    if len(text) > short_preview_length:
        short_preview += "..."

    with st.container(border=True):
        st.markdown(f"### #{rank} — {title}")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            try:
                st.metric("Score", f"{float(score):.4f}")
            except Exception:
                st.metric("Score", str(score))

        with col2:
            st.metric("Stance", stance if stance else "N/A")

        with col3:
            st.caption(f"Doc ID: {doc_id}")

        if url:
            st.markdown(f"[Open document source]({url})")

        if result.get("refinement_mode"):
            st.info(
                f"Refinement Mode: {result.get('refinement_mode')} | "
                f"Refined Query: {result.get('refined_query')}"
            )

        st.markdown("#### Short Preview")
        st.write(short_preview)

        score_details = {}

        if "bm25_score" in result:
            score_details["BM25 Score"] = result.get("bm25_score")

        if "tfidf_score" in result:
            score_details["TF-IDF Score"] = result.get("tfidf_score")

        if "embedding_score" in result:
            score_details["Embedding Score"] = result.get("embedding_score")

        if "hybrid_score" in result:
            score_details["Hybrid Score"] = result.get("hybrid_score")

        if "fusion_score" in result:
            score_details["Fusion Score"] = result.get("fusion_score")

        if "bm25_rank" in result:
            score_details["BM25 Rank"] = result.get("bm25_rank")

        if "tfidf_rank" in result:
            score_details["TF-IDF Rank"] = result.get("tfidf_rank")

        if score_details:
            with st.expander("Show model score details"):
                st.json(score_details)

        with st.expander("Show full document text"):
            st.write(text)

        if result.get("corrections") or result.get("added_terms"):
            with st.expander("Show query refinement details"):
                if result.get("corrections"):
                    st.markdown("**Corrections**")
                    st.json(result.get("corrections"))

                if result.get("added_terms"):
                    st.markdown("**Added Terms**")
                    st.json(result.get("added_terms"))

def render_topic_detection(query_text, results, topic_terms_count, model_label):
    topic_service = load_topic_detection_service()

    topic_output = topic_service.detect_topic(
        query_text=query_text,
        results=results,
        top_n=topic_terms_count,
    )

    st.subheader("Topic Detection")
    st.caption(
        "The system extracts the main topic terms from the query and the retrieved documents."
    )

    if not topic_output["top_terms"]:
        st.info(topic_output["message"])
        return

    st.markdown(f"**Detected Topic:** `{topic_output['topic_label']}`")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Retrieval Model", model_label)

    with col2:
        st.metric("Documents Analyzed", topic_output["document_count"])

    st.caption(
        "Topic Extraction Method: TF-IDF term weighting over the retrieved results. "
        "This is a post-retrieval analysis step, not the retrieval model itself."
    )

    topic_rows = []

    for item in topic_output["top_terms"]:
        topic_rows.append(
            {
                "Term": item["term"],
                "Score": item["score"],
            }
        )

    st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)
def render_result_clusters(results, cluster_count, cluster_result_count):
    clustering_service = ResultClusteringService(max_clusters=cluster_count)

    results_to_cluster = results[:cluster_result_count]

    cluster_output = clustering_service.cluster_results(
        results=results_to_cluster,
        max_clusters=cluster_count,
    )

    st.subheader("Search Result Clustering")
    st.caption(
        "The top retrieved documents are grouped according to textual similarity."
    )

    if cluster_output["cluster_count"] == 0:
        st.info(cluster_output["message"])
        return

    for cluster in cluster_output["clusters"]:
        with st.expander(
            f"Cluster {cluster['cluster_id']}: {cluster['label']} "
            f"({cluster['size']} documents)",
            expanded=False,
        ):
            if cluster["top_terms"]:
                st.markdown(
                    "**Top Terms:** "
                    + ", ".join(f"`{term}`" for term in cluster["top_terms"])
                )

            for document in cluster["documents"][:8]:
                title = document.get("title") or "Untitled Document"
                rank = document.get("rank")
                score = document.get("score")
                stance = document.get("stance") or "N/A"

                st.markdown(f"**Rank {rank}: {title}**")

                details = f"Score: {score:.4f}" if score is not None else "Score: N/A"
                details += f" | Stance: {stance}"

                st.caption(details)


def render_dataset_info():
    dataset_info = load_json_file(DATASET_INFO_PATH)

    if dataset_info is None:
        st.warning("Dataset info file was not found.")
        return

    st.subheader("Dataset Information")

    if isinstance(dataset_info, dict):
        useful_keys = [
            "dataset_id",
            "hf_dataset_id",
            "documents_count",
            "queries_count",
            "qrels_count",
            "unique_qrel_queries_count",
            "processed_docs_count",
            "processed_queries_count",
        ]

        shown_any = False
        cols = st.columns(4)
        metric_index = 0

        for key in useful_keys:
            if key in dataset_info:
                with cols[metric_index % 4]:
                    st.metric(key.replace("_", " ").title(), dataset_info[key])
                metric_index += 1
                shown_any = True

        if not shown_any:
            st.json(dataset_info)
    else:
        st.json(dataset_info)


def render_evaluation_summary():
    summary = load_json_file(EVALUATION_SUMMARY_PATH)

    if summary is None:
        st.warning("Evaluation summary file was not found.")
        return

    if isinstance(summary, dict):
        rows = [summary]
    else:
        rows = summary

    if not rows:
        st.warning("Evaluation summary is empty.")
        return

    df = pd.DataFrame(rows)

    selected_columns = [
        "model",
        "precision_at_10",
        "recall_at_100",
        "map_at_100",
        "ndcg_at_10",
        "mean_time_seconds",
        "evaluated_queries",
        "failed_queries_count",
    ]

    existing_columns = [column for column in selected_columns if column in df.columns]

    st.subheader("Evaluation Summary")
    st.dataframe(df[existing_columns], use_container_width=True)


def render_charts():
    chart_paths = [
        ("Precision@10", PRECISION_CHART_PATH),
        ("Recall@100", RECALL_CHART_PATH),
        ("MAP@100", MAP_CHART_PATH),
        ("nDCG@10", NDCG_CHART_PATH),
        ("Mean Query Time", TIME_CHART_PATH),
    ]

    for title, path in chart_paths:
        st.subheader(title)

        if Path(path).exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"Chart not found: {path}")


def main():
    st.title("🔎 Information Retrieval System — IR Project 2026")
    st.caption(
        "TF-IDF, BM25, Hybrid Retrieval, Query Refinement, and Evaluation Dashboard"
    )

    sidebar = st.sidebar

    sidebar.header("Search Settings")

    dataset_choice = sidebar.selectbox(
        "Dataset",
        ["BEIR / Webis Touché 2020"],
        index=0,
    )

    execution_mode = sidebar.radio(
        "Execution Mode",
        ["Basic Search", "Search with Extra Features"],
        index=0,
    )

    if execution_mode == "Basic Search":
        available_models = {
            "tfidf": MODEL_OPTIONS["tfidf"],
            "bm25": MODEL_OPTIONS["bm25"],
        }
    else:
        available_models = {
            "hybrid_serial": MODEL_OPTIONS["hybrid_serial"],
            "hybrid_parallel": MODEL_OPTIONS["hybrid_parallel"],
            "bm25_refined_corrected": MODEL_OPTIONS["bm25_refined_corrected"],
            "bm25_refined_expanded": MODEL_OPTIONS["bm25_refined_expanded"],
            "bm25": MODEL_OPTIONS["bm25"],
            "tfidf": MODEL_OPTIONS["tfidf"],
        }

    model_label = sidebar.selectbox(
        "Retrieval Model",
        list(available_models.values()),
    )

    model_key = [
        key for key, label in available_models.items()
        if label == model_label
    ][0]

    top_k = sidebar.slider(
        "Top K Results",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    sidebar.divider()
    sidebar.subheader("Result Analysis")

    enable_result_clustering = sidebar.checkbox(
        "Enable Result Clustering",
        value=False,
        help="Group the top retrieved documents into topic-based clusters.",
    )

    cluster_count = sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
    )

    cluster_result_count = sidebar.slider(
        "Results to Cluster",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
    )
    enable_topic_detection = sidebar.checkbox(
    "Enable Topic Detection",
    value=False,
    help="Extract the main topic terms from the query and top retrieved documents.",
)

    topic_terms_count = sidebar.slider(
    "Topic Terms Count",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)
    sidebar.divider()
    sidebar.subheader("BM25 Parameters")

    k1 = sidebar.slider(
        "k1",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Controls term frequency saturation.",
    )

    b = sidebar.slider(
        "b",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Controls document length normalization.",
    )

    sidebar.divider()
    sidebar.subheader("Hybrid Parameters")

    hybrid_serial_alpha = sidebar.slider(
        "Hybrid Serial Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Higher alpha means more BM25 influence.",
    )

    hybrid_serial_candidates = sidebar.slider(
        "Hybrid Serial BM25 Candidates",
        min_value=20,
        max_value=300,
        value=100,
        step=20,
    )

    hybrid_parallel_bm25_weight = sidebar.slider(
        "Hybrid Parallel BM25 Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
    )

    hybrid_parallel_tfidf_weight = round(1.0 - hybrid_parallel_bm25_weight, 2)

    sidebar.caption(f"TF-IDF Weight: {hybrid_parallel_tfidf_weight}")

    rrf_k = sidebar.slider(
        "RRF k",
        min_value=10,
        max_value=100,
        value=60,
        step=5,
    )

    query_text = st.text_input(
        "Enter your search query",
        value="Should teachers get tenure?",
        placeholder="Type a natural language query...",
    )

    tabs = st.tabs(
        [
            "Search",
            "Query Refinement",
            "Dataset",
            "Evaluation",
            "Charts",
        ]
    )

    with tabs[0]:
        st.subheader("Search")

        st.write("Selected Dataset:", dataset_choice)
        st.write("Selected Model:", model_label)

        if st.button("Run Search", type="primary"):
            if not query_text.strip():
                st.error("Please enter a query.")
            else:
                with st.spinner("Searching..."):
                    try:
                        results = search_with_model(
                            model_key=model_key,
                            query_text=query_text,
                            top_k=top_k,
                            k1=k1,
                            b=b,
                            hybrid_serial_alpha=hybrid_serial_alpha,
                            hybrid_serial_candidates=hybrid_serial_candidates,
                            hybrid_parallel_bm25_weight=hybrid_parallel_bm25_weight,
                            hybrid_parallel_tfidf_weight=hybrid_parallel_tfidf_weight,
                            rrf_k=rrf_k,
                        )

                        st.success(f"Retrieved {len(results)} results.")

                        for index, result in enumerate(results, start=1):
                            render_result_card(result, fallback_rank=index)



                        render_topic_detection(
    query_text=query_text,
    results=results,
    topic_terms_count=topic_terms_count,
    model_label=model_label,
)   

                        if enable_result_clustering and results:
                            st.markdown("---")
                            render_result_clusters(
                                results=results,
                                cluster_count=cluster_count,
                                cluster_result_count=cluster_result_count,
                            )

                    except Exception as error:
                        st.error("Search failed.")
                        st.exception(error)

    with tabs[1]:
        st.subheader("Query Refinement")

        refinement_service = load_query_refinement_service()

        if query_text.strip():
            refinement = refinement_service.refine_query(query_text)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Original Query")
                st.code(refinement["original_query"])

            with col2:
                st.markdown("#### Corrected Query")
                st.code(refinement["corrected_query"])

            with col3:
                st.markdown("#### Expanded Query")
                st.code(refinement["expanded_query"])

            st.markdown("#### Corrections")

            if refinement["corrections"]:
                st.dataframe(
                    pd.DataFrame(refinement["corrections"]),
                    use_container_width=True,
                )
            else:
                st.info("No spelling corrections were needed.")

            st.markdown("#### Added Terms")

            if refinement["added_terms"]:
                st.dataframe(
                    pd.DataFrame(refinement["added_terms"]),
                    use_container_width=True,
                )
            else:
                st.info("No synonym expansion terms were added.")

            st.markdown("#### Suggestions")

            if refinement["suggestions"]:
                for suggestion in refinement["suggestions"]:
                    st.code(suggestion)
            else:
                st.info("No suggestions generated.")
        else:
            st.info("Enter a query to see refinement suggestions.")

    with tabs[2]:
        render_dataset_info()

    with tabs[3]:
        render_evaluation_summary()

    with tabs[4]:
        render_charts()


if __name__ == "__main__":
    main()