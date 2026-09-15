class DomainError(Exception):
    """Base exception for all domain-level errors."""


class AnalysisNotFoundError(DomainError):
    def __init__(self, analysis_id: str) -> None:
        super().__init__(f"Analysis '{analysis_id}' was not found")
        self.analysis_id = analysis_id


class StaleEpochError(DomainError):
    def __init__(self, expected_epoch: int, current_epoch: int) -> None:
        super().__init__(
            f"Expected epoch {expected_epoch} does not match active epoch {current_epoch}"
        )
        self.expected_epoch = expected_epoch
        self.current_epoch = current_epoch


class InvalidEpochError(DomainError):
    def __init__(self, epoch: int) -> None:
        super().__init__(f"Epoch must be an integer >= 1, got {epoch}")
        self.epoch = epoch


class InvalidTargetError(DomainError):
    def __init__(self, candidate_id: str) -> None:
        super().__init__(f"Target candidate '{candidate_id}' is invalid or has expired")
        self.candidate_id = candidate_id


class TargetAmbiguousError(DomainError):
    def __init__(self, message: str = "Tracking target has become ambiguous") -> None:
        super().__init__(message)


class CursorExpiredError(DomainError):
    def __init__(self, requested_cursor: str, oldest_cursor: str | None = None) -> None:
        super().__init__(
            f"Requested observation cursor '{requested_cursor}' has expired from buffer"
        )
        self.requested_cursor = requested_cursor
        self.oldest_cursor = oldest_cursor
