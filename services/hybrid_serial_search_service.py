import sqlite3
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DOCUMENT_STORE_DB_PATH, EMBEDDING_MODEL_NAME
from services.bm25_search_service import BM25SearchService


class HybridSerialSearchService:

    def __init__(self):
        self.bm25_service = BM25SearchService()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _get_documents_by_row_ids(self, row_ids: List[int]) -> Dict[int, Dict]:
        if not row_ids:
            return {}

        placeholders = ",".join(["?"] * len(row_ids))

        connection = sqlite3.connect(DOCUMENT_STORE_DB_PATH)
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT row_id, doc_id, title, stance, url, original_text, cleaned_text
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
                "cleaned_text": row["cleaned_text"],
            }
            for row in rows
        }

    def _prepare_embedding_text(self, document: Dict) -> str:
        title = document.get("title", "")
        original_text = document.get("original_text", "")

        text = f"{title}. {original_text}".strip()
        text = " ".join(text.split())

        return text[:1200]

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        bm25_candidates: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
        alpha: float = 0.5,
    ) -> List[Dict]:
        bm25_results = self.bm25_service.search(
            query_text=query_text,
            top_k=bm25_candidates,
            k1=k1,
            b=b,
        )

        if not bm25_results:
            return []

        row_ids = [result["row_id"] for result in bm25_results]
        documents_by_row_id = self._get_documents_by_row_ids(row_ids)

        candidate_texts = []
        valid_candidates = []

        for bm25_result in bm25_results:
            row_id = bm25_result["row_id"]
            document = documents_by_row_id.get(row_id)

            if document is None:
                continue

            candidate_texts.append(self._prepare_embedding_text(document))
            valid_candidates.append((bm25_result, document))

        if not valid_candidates:
            return []

        query_embedding = self.embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        document_embeddings = self.embedding_model.encode(
            candidate_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedding_scores = document_embeddings.dot(query_embedding)

        bm25_scores = np.array(
            [candidate[0]["score"] for candidate in valid_candidates],
            dtype=np.float32,
        )

        if bm25_scores.max() > bm25_scores.min():
            bm25_normalized = (bm25_scores - bm25_scores.min()) / (
                bm25_scores.max() - bm25_scores.min()
            )
        else:
            bm25_normalized = np.ones_like(bm25_scores)

        embedding_normalized = (embedding_scores + 1.0) / 2.0

        hybrid_scores = alpha * bm25_normalized + (1.0 - alpha) * embedding_normalized

        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            bm25_result, document = valid_candidates[int(index)]

            results.append(
                {
                    "rank": rank,
                    "score": float(hybrid_scores[index]),
                    "hybrid_score": float(hybrid_scores[index]),
                    "bm25_score": float(bm25_result["score"]),
                    "embedding_score": float(embedding_scores[index]),
                    "row_id": bm25_result["row_id"],
                    "doc_id": document.get("doc_id", ""),
                    "title": document.get("title", ""),
                    "stance": document.get("stance", ""),
                    "url": document.get("url", ""),
                    "original_text": document.get("original_text", ""),
                    "k1": k1,
                    "b": b,
                    "alpha": alpha,
                    "bm25_candidates": bm25_candidates,
                }
            )

        return results