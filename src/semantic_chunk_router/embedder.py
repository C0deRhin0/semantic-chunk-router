import hashlib
import re
from typing import Dict, List, Tuple

import numpy as np


class MockEmbedder:
    """
    A lightweight mock embedder that generates reproducible pseudo-embeddings
    from normalized word occurrences and a small set of semantic aliases.

    This is intended for local testing and offline fallback. It is not a
    substitute for production embedding models, but it preserves enough lexical
    overlap for deterministic chunking and routing tests.
    """
    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
    _SEMANTIC_ALIASES: Dict[str, Tuple[str, ...]] = {
        "api": ("technical", "software"),
        "component": ("ui", "interface"),
        "css": ("ui", "style", "layout"),
        "database": ("data", "sql", "schema"),
        "deploy": ("deployment", "technical", "software"),
        "deployed": ("deployment", "technical", "software"),
        "deployment": ("technical", "software"),
        "html": ("ui", "interface"),
        "index": ("database", "data"),
        "k8s": ("kubernetes", "deployment", "technical", "software"),
        "kubernetes": ("deployment", "technical", "software"),
        "layout": ("ui", "interface"),
        "prod": ("production", "deployment", "technical"),
        "production": ("deployment", "technical"),
        "query": ("support", "request"),
        "schema": ("database", "data"),
        "software": ("technical",),
        "sql": ("database", "data"),
        "style": ("ui", "css"),
        "technical": ("software",),
        "ui": ("interface", "layout"),
    }
    _STOP_WORDS = {
        "a",
        "an",
        "and",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "we",
    }

    def __init__(self, dimension: int = 1536):
        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero")
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension

        vec = np.zeros(self.dimension, dtype=float)
        for term, weight in self._term_weights(text).items():
            vec[self._hash_index(term)] += weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def _term_weights(self, text: str) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for raw_token in self._TOKEN_PATTERN.findall(text.lower()):
            token = self._normalize_token(raw_token)
            if token in self._STOP_WORDS:
                continue
            for term in self._expanded_terms(token):
                weights[term] = weights.get(term, 0.0) + 1.0
        return weights

    def _expanded_terms(self, token: str) -> Tuple[str, ...]:
        aliases = self._SEMANTIC_ALIASES.get(token, ())
        return (token, *aliases)

    def _normalize_token(self, token: str) -> str:
        if token in self._SEMANTIC_ALIASES:
            return token

        if token.endswith("ies") and len(token) > 4:
            singular = f"{token[:-3]}y"
            if singular in self._SEMANTIC_ALIASES:
                return singular
            return token

        if token.endswith("es") and len(token) > 4:
            singular = token[:-2]
            if singular in self._SEMANTIC_ALIASES:
                return singular

        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            singular = token[:-1]
            if singular in self._SEMANTIC_ALIASES:
                return singular

        return token

    def _hash_index(self, term: str) -> int:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") % self.dimension
