import json

import matplotlib.pyplot as plt

from config import (
    EVALUATION_SUMMARY_PATH,
    PRECISION_CHART_PATH,
    RECALL_CHART_PATH,
    MAP_CHART_PATH,
    NDCG_CHART_PATH,
    TIME_CHART_PATH,
)


MODEL_DISPLAY_NAMES = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "hybrid_serial": "Hybrid\nSerial",
    "hybrid_parallel": "Hybrid\nParallel",
    "bm25_refined_corrected": "BM25 Refined\nCorrected",
    "bm25_refined_expanded": "BM25 Refined\nExpanded",
}


def load_evaluation_summary():
    with open(EVALUATION_SUMMARY_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return [data]

    return data


def get_model_label(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", "\n").upper())


def create_bar_chart(summary_rows, metric_key, title, ylabel, output_path):
    models = [get_model_label(row["model"]) for row in summary_rows]
    values = [float(row.get(metric_key, 0.0)) for row in summary_rows]

    plt.figure(figsize=(14, 7))

    bars = plt.bar(models, values)

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Retrieval Model", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.xticks(rotation=0, ha="center", fontsize=10)
    plt.yticks(fontsize=10)

    max_value = max(values) if values else 0.0
    upper_limit = max_value * 1.25 if max_value > 0 else 1.0
    plt.ylim(0, upper_limit)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (upper_limit * 0.02),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved chart:", output_path)


def generate_evaluation_charts():
    print("=" * 70)
    print("GENERATING EVALUATION CHARTS")
    print("=" * 70)

    summary_rows = load_evaluation_summary()

    create_bar_chart(
        summary_rows=summary_rows,
        metric_key="precision_at_10",
        title="Precision@10 Comparison",
        ylabel="Precision@10",
        output_path=PRECISION_CHART_PATH,
    )

    create_bar_chart(
        summary_rows=summary_rows,
        metric_key="recall_at_100",
        title="Recall@100 Comparison",
        ylabel="Recall@100",
        output_path=RECALL_CHART_PATH,
    )

    create_bar_chart(
        summary_rows=summary_rows,
        metric_key="map_at_100",
        title="MAP@100 Comparison",
        ylabel="MAP@100",
        output_path=MAP_CHART_PATH,
    )

    create_bar_chart(
        summary_rows=summary_rows,
        metric_key="ndcg_at_10",
        title="nDCG@10 Comparison",
        ylabel="nDCG@10",
        output_path=NDCG_CHART_PATH,
    )

    create_bar_chart(
        summary_rows=summary_rows,
        metric_key="mean_time_seconds",
        title="Mean Query Time Comparison",
        ylabel="Mean Query Time Seconds",
        output_path=TIME_CHART_PATH,
    )

    print("\nEvaluation charts generated successfully.")