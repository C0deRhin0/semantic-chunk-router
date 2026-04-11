class SemanticRouterError(Exception):
    """Base exception for semantic chunk router."""
    pass

class EmbeddingError(SemanticRouterError):
    """Raised when embedding generation fails."""
    pass

class ConfigurationError(SemanticRouterError):
    """Raised when configuration is invalid."""
    pass


class RoutingValidationError(SemanticRouterError):
    """Raised when route definitions are invalid."""
    pass
