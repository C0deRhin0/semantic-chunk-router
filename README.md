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
