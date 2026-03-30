import re
import numpy as np
from typing import List, Dict, Any, Optional
from .embedder import MockEmbedder
from .exceptions import ConfigurationError

class SemanticChunker:
    """
    Splits text documents into semantically coherent chunks based on the similarity
    of adjacent sentences using embeddings.
    """
    def __init__(
        self,
        embedder=None,
        similarity_threshold: float = 0.82,
        max_chunk_size: int = 1000,
        min_sentences_per_chunk: int = 1
    ):
        self.embedder = embedder or MockEmbedder()
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk

        if not (0.0 <= similarity_threshold <= 1.0):
            raise ConfigurationError("Similarity threshold must be between 0.0 and 1.0")

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple rule-based sentence splitter
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        embeddings = self.embedder.embed_documents(sentences)
        
        chunks = []
        current_sentences = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            sentence_len = len(sentences[i])

            should_split = (
                sim < self.similarity_threshold and 
                len(current_sentences) >= self.min_sentences_per_chunk
            ) or (
                current_len + sentence_len > self.max_chunk_size
            )

            if should_split:
                chunks.append({
                    "text": " ".join(current_sentences),
                    "sentence_count": len(current_sentences),
                    "character_count": current_len
                })
                current_sentences = [sentences[i]]
                current_len = sentence_len
            else:
                current_sentences.append(sentences[i])
                current_len += sentence_len + 1 # count space

        if current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "sentence_count": len(current_sentences),
                "character_count": current_len
            })

        return chunks
