import hashlib
import numpy as np
from typing import List, Union

class MockEmbedder:
    """
    A lightweight, zero-dependency mock embedder that generates reproducible
    pseudo-embeddings based on word occurrences and hashing.
    Used for local testing and offline fallback.
    """
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension
        # Generate a stable random representation using SHA-256
        hash_digest = hashlib.sha256(text.encode('utf-8')).digest()
        # Seed generator with hash integer
        seed = int.from_bytes(hash_digest[:4], byteorder='big')
        rng = np.random.default_rng(seed)
        vec = rng.normal(0.0, 1.0, self.dimension)
        # Normalize vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
