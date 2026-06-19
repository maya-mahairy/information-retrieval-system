import difflib
from typing import Dict, List, Set

import joblib

from config import BM25_COUNT_VECTORIZER_PATH
from services.preprocessing_service import normalize_text, tokenize_text


class QueryRefinementService:
    """
    Conservative Query Refinement Service.

    It improves user queries using:
    1. Safe spelling correction.
    2. Controlled synonym expansion.
    3. Query suggestions.

    The correction is intentionally conservative to avoid damaging valid queries.
    """

    def __init__(self):
        self.vectorizer = joblib.load(BM25_COUNT_VECTORIZER_PATH)
        raw_vocabulary = set(self.vectorizer.vocabulary_.keys())

        # Keep only clean alphabetic words.
        # This prevents bad corrections like students -> students3.
        self.vocabulary = {
            word
            for word in raw_vocabulary
            if word.isalpha() and len(word) > 2
        }

        self.vocabulary_by_first_letter = self._group_vocabulary_by_first_letter()

        # Words that should never be auto-corrected.
        self.protected_terms = {
            "a", "an", "the",
            "is", "are", "am", "be", "being", "been",
            "was", "were",
            "do", "does", "did",
            "should", "would", "could", "can", "may", "might", "must",
            "get", "go", "going",
            "to", "of", "in", "on", "for", "with", "and", "or",
            "have", "has", "had",
            "not", "no", "yes",
            "students", "student",
            "teachers", "teacher",
            "educators", "educator",
            "professors", "professor",
            "college", "colleges",
            "school", "schools",
            "classroom", "classrooms",
            "tenure", "tenured",
            "reward", "rewarded",
            "armed",
        }

        # Manual corrections are safer than relying only on noisy dataset vocabulary.
        self.manual_corrections = {
            "techer": "teacher",
            "techers": "teachers",
            "teachr": "teacher",
            "teachrs": "teachers",
            "tenur": "tenure",
            "tenrue": "tenure",
            "studnts": "students",
            "studenst": "students",
            "profesor": "professor",
            "profesors": "professors",
            "proffesor": "professor",
            "proffesors": "professors",
            "collage": "college",
            "universty": "university",
            "shcool": "school",
            "scool": "school",
            "clasroom": "classroom",
        }

        self.synonym_map = {
            "teacher": ["educator", "instructor"],
            "teachers": ["educator", "instructor"],
            "educator": ["teacher", "instructor"],
            "educators": ["teacher", "instructor"],
            "instructor": ["teacher", "educator"],
            "instructors": ["teacher", "educator"],

            "student": ["pupil", "learner"],
            "students": ["pupil", "learner"],

            "school": ["classroom", "education"],
            "schools": ["classroom", "education"],
            "classroom": ["school", "education"],
            "classrooms": ["school", "education"],

            "tenure": ["job", "security", "protection"],
            "tenured": ["tenure", "job", "security", "protection"],

            "job": ["work", "employment"],
            "security": ["protection"],
            "protection": ["security"],

            "gun": ["firearm", "weapon", "armed"],
            "guns": ["firearm", "weapon", "armed"],
            "armed": ["gun", "firearm", "weapon"],

            "reward": ["incentive", "paid", "payment"],
            "rewarded": ["incentive", "paid", "payment"],
            "rewards": ["incentive", "paid", "payment"],

            "pay": ["salary", "payment", "paid"],
            "paid": ["pay", "salary", "payment"],
            "payment": ["pay", "paid"],

            "college": ["university", "education"],
            "colleges": ["university", "education"],
            "university": ["college", "education"],
            "professor": ["teacher", "instructor"],
            "professors": ["teacher", "instructor"],
        }

    def _group_vocabulary_by_first_letter(self) -> Dict[str, List[str]]:
        grouped = {}

        for word in self.vocabulary:
            if not word:
                continue

            first_letter = word[0]
            grouped.setdefault(first_letter, []).append(word)

        return grouped

    def _get_candidate_words(self, token: str) -> List[str]:
        if not token:
            return []

        first_letter = token[0]
        candidates = self.vocabulary_by_first_letter.get(first_letter, [])

        return [
            word
            for word in candidates
            if word.isalpha()
            and abs(len(word) - len(token)) <= 2
        ]

    def correct_token(self, token: str) -> str:
        """
        Corrects one token safely.

        Priority:
        1. Manual correction.
        2. Keep protected terms unchanged.
        3. Keep valid vocabulary terms unchanged.
        4. Try conservative fuzzy correction.
        """
        token = token.lower()

        if token in self.manual_corrections:
            return self.manual_corrections[token]

        if token in self.protected_terms:
            return token

        if token in self.vocabulary:
            return token

        if len(token) <= 4:
            return token

        candidates = self._get_candidate_words(token)

        if not candidates:
            return token

        matches = difflib.get_close_matches(
            token,
            candidates,
            n=1,
            cutoff=0.92,
        )

        if not matches:
            return token

        candidate = matches[0]

        # Extra safety: do not accept strange candidates.
        if not candidate.isalpha():
            return token

        if candidate in {"sshould", "students3", "teicher", "proffessors"}:
            return token

        return candidate

    def correct_query(self, query_text: str) -> Dict:
        normalized_query = normalize_text(query_text)
        tokens = tokenize_text(normalized_query)

        corrected_tokens = []
        corrections = []

        for token in tokens:
            corrected_token = self.correct_token(token)
            corrected_tokens.append(corrected_token)

            if corrected_token != token:
                corrections.append(
                    {
                        "original": token,
                        "corrected": corrected_token,
                    }
                )

        corrected_query = " ".join(corrected_tokens)

        return {
            "original_query": query_text,
            "normalized_query": normalized_query,
            "tokens": tokens,
            "corrected_tokens": corrected_tokens,
            "corrected_query": corrected_query,
            "corrections": corrections,
        }

    def expand_query(self, corrected_tokens: List[str]) -> Dict:
        """
        Adds controlled synonyms.

        We keep expansion limited to prevent query drift.
        """
        expanded_tokens = list(corrected_tokens)
        added_terms = []
        seen_terms: Set[str] = set(corrected_tokens)

        for token in corrected_tokens:
            synonyms = self.synonym_map.get(token, [])

            for synonym in synonyms:
                synonym_tokens = tokenize_text(normalize_text(synonym))

                for synonym_token in synonym_tokens:
                    if synonym_token not in seen_terms:
                        expanded_tokens.append(synonym_token)
                        added_terms.append(
                            {
                                "source_token": token,
                                "added_term": synonym_token,
                            }
                        )
                        seen_terms.add(synonym_token)

        expanded_query = " ".join(expanded_tokens)

        return {
            "expanded_tokens": expanded_tokens,
            "expanded_query": expanded_query,
            "added_terms": added_terms,
        }

    def refine_query(self, query_text: str) -> Dict:
        correction_result = self.correct_query(query_text)

        expansion_result = self.expand_query(
            correction_result["corrected_tokens"]
        )

        suggestions = []

        corrected_query = correction_result["corrected_query"]
        expanded_query = expansion_result["expanded_query"]

        if corrected_query and corrected_query != correction_result["normalized_query"]:
            suggestions.append(corrected_query)

        if expanded_query and expanded_query != corrected_query:
            suggestions.append(expanded_query)

        return {
            "original_query": query_text,
            "normalized_query": correction_result["normalized_query"],
            "corrected_query": corrected_query,
            "expanded_query": expanded_query,
            "tokens": correction_result["tokens"],
            "corrected_tokens": correction_result["corrected_tokens"],
            "expanded_tokens": expansion_result["expanded_tokens"],
            "corrections": correction_result["corrections"],
            "added_terms": expansion_result["added_terms"],
            "suggestions": suggestions,
        }