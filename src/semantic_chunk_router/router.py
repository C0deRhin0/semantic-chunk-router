import numpy as np
from typing import List, Dict, Any, Optional
from .embedder import MockEmbedder

class Route:
    def __init__(self, name: str, description: str, threshold: float = 0.75):
        self.name = name
        self.description = description
        self.threshold = threshold

class SemanticRouter:
    """
    Routes text chunks to defined destination categories based on semantic matching
    against route descriptions.
    """
    def __init__(self, routes: List[Route], embedder=None):
        self.routes = routes
        self.embedder = embedder or MockEmbedder()
        self._route_embeddings = {}
        self._embed_routes()

    def _embed_routes(self):
        for route in self.routes:
            self._route_embeddings[route.name] = self.embedder.embed_text(route.description)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def route_chunk(self, chunk_text: str) -> Optional[str]:
        if not chunk_text or not self.routes:
            return None

        chunk_emb = self.embedder.embed_text(chunk_text)
        best_route = None
        best_score = -1.0

        for route in self.routes:
            route_emb = self._route_embeddings[route.name]
            score = self._cosine_similarity(chunk_emb, route_emb)
            if score >= route.threshold:
                if score > best_score:
                    best_score = score
                    best_route = route.name

        return best_route
