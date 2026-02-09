"""Exception classes for statistical analysis."""


class StatisticalError(Exception):
    """Base exception for statistical analysis errors."""

    pass


class InsufficientDataError(StatisticalError):
    """Raised when there is insufficient data for a statistical calculation."""

    def __init__(self, message: str, min_required: int | None = None) -> None:
        self.min_required = min_required
        super().__init__(message)
