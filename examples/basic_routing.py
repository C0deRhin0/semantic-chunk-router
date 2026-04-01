from semantic_chunk_router import SemanticChunker, SemanticRouter
from semantic_chunk_router.router import Route

# Setup router
chunker = SemanticChunker(similarity_threshold=0.8)
routes = [
    Route("database", "Databases, SQL queries, schema updates"),
    Route("ui", "HTML, CSS styles, UI components")
]
router = SemanticRouter(routes)

text = "Database indexes must be optimized. Update the CSS files to match UI layout specs."
chunks = chunker.chunk_text(text)

for c in chunks:
    rt = router.route_chunk(c["text"])
    print(f"Chunk: {c['text']} -> Target: {rt}")
