import pytest

from semantic_chunk_router import MockEmbedder, Route, SemanticRouter


class FrozenEmbedder:
    """Known vectors isolate routing decisions from model quality and network I/O."""

    def embed_text(self, text):
        return {
            "Technical software deployment": [1.0, 0.0],
            "Customer support": [0.0, 1.0],
            "Deploy Kubernetes": [0.95, 0.1],
            "Mixed query": [3.0, 4.0],
            "Equally relevant": [1.0, 1.0],
            "No signal": [0.0, 0.0],
        }[text]


def routes(tech_threshold=0.75, support_threshold=0.75):
    return [
        Route("tech", "Technical software deployment", tech_threshold),
        Route("support", "Customer support", support_threshold),
    ]


def test_router_no_routes():
    assert SemanticRouter([]).route_chunk("hello") is None


def test_basic_routing_with_an_injected_embedder():
    router = SemanticRouter(routes(), embedder=FrozenEmbedder())
    assert router.route_chunk("Deploy Kubernetes") == "tech"


def test_highest_scoring_eligible_route_wins():
    router = SemanticRouter(routes(0.5, 0.5), embedder=FrozenEmbedder())
    assert router.route_chunk("Mixed query") == "support"


def test_each_route_uses_its_own_threshold():
    router = SemanticRouter(routes(0.6, 0.9), embedder=FrozenEmbedder())
    assert router.route_chunk("Mixed query") == "tech"


def test_no_eligible_route_returns_none():
    router = SemanticRouter(routes(), embedder=FrozenEmbedder())
    assert router.route_chunk("Equally relevant") is None


def test_tied_routes_keep_declaration_order():
    router = SemanticRouter(routes(0.5, 0.5), embedder=FrozenEmbedder())
    assert router.route_chunk("Equally relevant") == "tech"


@pytest.mark.parametrize("text", ["", "No signal"])
def test_empty_or_zero_signal_does_not_route(text):
    router = SemanticRouter(routes(), embedder=FrozenEmbedder())
    assert router.route_chunk(text) is None


def test_offline_embedder_routes_an_identical_description():
    description = "Exact document route description"
    router = SemanticRouter([Route("exact", description)])
    assert router.route_chunk(description) == "exact"


def test_mock_vectors_are_repeatable_without_changing_global_random_state():
    import numpy as np

    state = np.random.get_state()
    embedder = MockEmbedder(dimension=16)
    assert embedder.embed_text("same input") == embedder.embed_text("same input")
    actual = np.random.get_state()
    assert state[0] == actual[0]
    np.testing.assert_array_equal(state[1], actual[1])
    assert state[2:] == actual[2:]
