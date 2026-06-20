from typing import Dict, List

from services.bm25_search_service import BM25SearchService
from services.tfidf_search_service import TFIDFSearchService


class HybridParallelSearchService:

    def __init__(self):
        self.bm25_service = BM25SearchService()
        self.tfidf_service = TFIDFSearchService()

    def _create_or_get_record(self, fused_results: Dict[str, Dict], result: Dict) -> Dict:
        doc_id = str(result.get("doc_id", ""))

        if doc_id not in fused_results:
            fused_results[doc_id] = {
                "doc_id": doc_id,
                "row_id": result.get("row_id"),
                "title": result.get("title", ""),
                "stance": result.get("stance", ""),
                "url": result.get("url", ""),
                "original_text": result.get("original_text", ""),
                "score": 0.0,
                "fusion_score": 0.0,
                "bm25_rank": None,
                "bm25_score": None,
                "tfidf_rank": None,
                "tfidf_score": None,
            }

        return fused_results[doc_id]

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        bm25_top_k: int = 100,
        tfidf_top_k: int = 100,
        rrf_k: int = 60,
        bm25_weight: float = 0.8,
        tfidf_weight: float = 0.2,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Dict]:
        bm25_results = self.bm25_service.search(
            query_text=query_text,
            top_k=bm25_top_k,
            k1=k1,
            b=b,
        )

        tfidf_results = self.tfidf_service.search(
            query_text=query_text,
            top_k=tfidf_top_k,
        )

        fused_results = {}

        for rank, result in enumerate(bm25_results, start=1):
            record = self._create_or_get_record(fused_results, result)

            rrf_score = bm25_weight / (rrf_k + rank)

            record["score"] += rrf_score
            record["fusion_score"] = record["score"]
            record["bm25_rank"] = rank
            record["bm25_score"] = result.get("score")

        for rank, result in enumerate(tfidf_results, start=1):
            record = self._create_or_get_record(fused_results, result)

            rrf_score = tfidf_weight / (rrf_k + rank)

            record["score"] += rrf_score
            record["fusion_score"] = record["score"]
            record["tfidf_rank"] = rank
            record["tfidf_score"] = result.get("score")

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item["fusion_score"],
            reverse=True,
        )

        final_results = []

        for rank, result in enumerate(ranked_results[:top_k], start=1):
            result["rank"] = rank
            result["fusion_method"] = "weighted_rrf"
            result["rrf_k"] = rrf_k
            result["bm25_weight"] = bm25_weight
            result["tfidf_weight"] = tfidf_weight
            result["k1"] = k1
            result["b"] = b

            final_results.append(result)

        return final_results