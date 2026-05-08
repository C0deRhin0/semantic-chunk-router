# Semantic Chunk Router

A Python library to split large documents into semantically coherent chunks based on sentence embedding similarity and route them to different downstream LLMs or Vector DB paths.

## Key Features

- **Semantic Chunker**: Automatically groups sentences based on cosine similarity thresholds.
- **Semantic Router**: Routes chunked text to specific destinations based on route descriptions.
- **Zero-Dependency Mock Embedder**: Includes a fallback embedder for testing without external API keys.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from semantic_chunk_router import SemanticChunker, SemanticRouter
from semantic_chunk_router.router import Route

# 1. Initialize chunker
chunker = SemanticChunker(similarity_threshold=0.82)
chunks = chunker.chunk_text("This is an LLM agent tutorial. We will configure routing. Oncall notifications will be configured separately.")

# 2. Route chunks
routes = [
    Route("llm_agent", "Topics related to agent orchestration and large language models"),
    Route("oncall", "Oncall notification setups, paging, and system incidents")
]
router = SemanticRouter(routes)

for chunk in chunks:
    destination = router.route_chunk(chunk["text"])
    print(f"Chunk: {chunk['text'][:40]}... -> Routed to: {destination}")
```

<!-- Last updated evaluation statistics on 2026-04-13 - Dataset size: 3 -->
<!-- Last updated evaluation statistics on 2026-05-08 - Dataset size: 11 -->