import json
import sqlite3
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    DOCUMENT_STORE_DB_PATH,
    EMBEDDING_MATRIX_PATH,
    EMBEDDING_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)


class EmbeddingSearchService:

    def __init__(self):
        with open(EMBEDDING_METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.model_name = self.metadata.get("model_name", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(self.model_name)

        self.embedding_matrix = np.load(EMBEDDING_MATRIX_PATH, mmap_mode="r")
        self.docs_count = int(self.metadata["docs_count"])

    def _get_documents_by_row_ids(self, row_ids: List[int]) -> Dict[int, Dict]:
        if not row_ids:
            return {}

        placeholders = ",".join(["?"] * len(row_ids))

        connection = sqlite3.connect(DOCUMENT_STORE_DB_PATH)
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT row_id, doc_id, title, stance, url, original_text
            FROM documents
            WHERE row_id IN ({placeholders})
            """,
            row_ids,
        )

        rows = cursor.fetchall()
        connection.close()

        return {
            int(row["row_id"]): {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "stance": row["stance"],
                "url": row["url"],
                "original_text": row["original_text"],
            }
            for row in rows
        }

    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        query_text = " ".join(str(query_text).split())

        if not query_text:
            return []

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        scores = self.embedding_matrix.dot(query_embedding)

        if scores.max() <= 0:
            return []

        top_k = min(top_k, self.docs_count)

        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        row_ids = [int(index) for index in ranked_indices]
        documents = self._get_documents_by_row_ids(row_ids)

        results = []

        for rank, row_id in enumerate(row_ids, start=1):
            doc = documents.get(row_id, {})

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[row_id]),
                    "row_id": row_id,
                    "doc_id": doc.get("doc_id", ""),
                    "title": doc.get("title", ""),
                    "stance": doc.get("stance", ""),
                    "url": doc.get("url", ""),
                    "original_text": doc.get("original_text", ""),
                    "model_name": self.model_name,
                }
            )

        return results