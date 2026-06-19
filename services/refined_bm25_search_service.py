from typing import Dict, List

from services.bm25_search_service import BM25SearchService
from services.query_refinement_service import QueryRefinementService


class RefinedBM25SearchService:
    """
    BM25 search with Query Refinement.

    Modes:
    - corrected: uses the spelling-corrected query only.
    - expanded: uses the synonym-expanded query.
    """

    def __init__(self, mode: str = "corrected"):
        if mode not in {"corrected", "expanded"}:
            raise ValueError("mode must be either 'corrected' or 'expanded'")

        self.mode = mode
        self.bm25_service = BM25SearchService()
        self.refinement_service = QueryRefinementService()

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Dict]:
        refinement = self.refinement_service.refine_query(query_text)

        if self.mode == "expanded":
            refined_query = refinement["expanded_query"]
        else:
            refined_query = refinement["corrected_query"]

        results = self.bm25_service.search(
            query_text=refined_query,
            top_k=top_k,
            k1=k1,
            b=b,
        )

        for result in results:
            result["original_query"] = query_text
            result["refined_query"] = refined_query
            result["refinement_mode"] = self.mode
            result["corrections"] = refinement["corrections"]
            result["added_terms"] = refinement["added_terms"]

        return results