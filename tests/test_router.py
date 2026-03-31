import pytest
from semantic_chunk_router import SemanticRouter, Route

def test_router_no_routes():
    router = SemanticRouter([])
    assert router.route_chunk("hello") is None

def test_basic_routing():
    routes = [
        Route("tech", "Issues related to technical software deployment"),
        Route("support", "General customer support queries")
    ]
    router = SemanticRouter(routes)
    result = router.route_chunk("We need to configure the kubernetes deployments on prod.")
    assert result == "tech"
