import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CATALOG_PATH = Path("data/shl_catalog.json")


def _load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        return []
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def _make_document(item: dict) -> str:
    """
    Flatten all searchable fields into one string for TF-IDF indexing.
    """
    keys = item.get("keys", [])
    keys_str = " ".join(keys) if isinstance(keys, list) else ""
    job_levels = item.get("job_levels", [])
    levels_str = " ".join(job_levels) if isinstance(job_levels, list) else ""

    parts = [
        item.get("name", ""),
        item.get("description", ""),
        keys_str,
        levels_str,
    ]
    return " ".join(p for p in parts if p).lower()


def _get_url(item: dict) -> str:
    """Catalog has either 'url' or 'link' (handles both)"""
    return item.get("url") or item.get("link") or ""


def _get_test_type(item: dict) -> str:
    """Catalog has either 'test_type' or 'status' for the type code."""
    return item.get("test_type") or item.get("status") or ""


class CatalogRetriever:
    def __init__(self):
        self.items = _load_catalog()
        self._docs = [_make_document(item) for item in self.items]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams: "java developer" scores as a unit
            min_df=1,
            stop_words="english",
        )
        # Only fit if documents are present
        if self._docs:
            self._matrix = self._vectorizer.fit_transform(self._docs)
        else:
            self._matrix = None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        TF-IDF cosine similarity search.
        Returns up to top_k items with a '_score' field appended.
        Falls back to empty list if catalog is empty.
        """
        if self._matrix is None or not query.strip():
            return []

        q_vec = self._vectorizer.transform([query.lower()])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            if scores[idx] < 0.01:   # skip near-zero matches
                break
            item = self.items[idx]
            results.append({
                **item,
                "url": _get_url(item),         
                "test_type": _get_test_type(item),  
                "_score": float(scores[idx]),
            })
            if len(results) >= top_k:
                break

        return results

    def get_by_name(self, name: str) -> dict | None:
        """Exact then fuzzy name match — used for compare queries."""
        name_lower = name.lower().strip()
        for item in self.items:
            if item.get("name", "").lower() == name_lower:
                return item
        for item in self.items:
            if name_lower in item.get("name", "").lower():
                return item
        return None

    def format_for_prompt(self, items: list[dict]) -> str:
        """
        Serialise retrieved items for injection into the groq system prompt.
        """
        lines = []
        for i, item in enumerate(items, 1):
            keys = item.get("keys", [])
            keys_str = ", ".join(keys[:3]) if isinstance(keys, list) else ""
            lines.append(
                f"{i}. {item.get('name')} "
                f"| test_type={item.get('test_type', '')} "
                f"| keys={keys_str} "
                f"| url={item.get('url', '')}"
            )
        return "\n".join(lines)

    def valid_urls(self) -> set[str]:
        """Set of all known catalog URLs(used for hallucination whitelist.)"""
        return {_get_url(item) for item in self.items if _get_url(item)}

    def all_names(self) -> list[str]:
        return [item.get("name", "") for item in self.items]