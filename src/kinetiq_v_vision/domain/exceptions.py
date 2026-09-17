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


class AnalysisNotTrackingError(DomainError):
    """Raised when frame tracking is attempted while the analysis is not in
    the TRACKING state (e.g. after a stream restart cleared the target and
    reset the analysis to AWAITING_SELECTION, or the analysis is PAUSED)."""

    def __init__(self, analysis_id: str, current_state: str) -> None:
        super().__init__(
            f"Analysis '{analysis_id}' is not in TRACKING state (current: {current_state}); "
            "target must be (re)confirmed via select_target before tracking can proceed"
        )
        self.analysis_id = analysis_id
        self.current_state = current_state


class CursorExpiredError(DomainError):
    def __init__(self, requested_cursor: str, oldest_cursor: str | None = None) -> None:
        super().__init__(
            f"Requested observation cursor '{requested_cursor}' has expired from buffer"
        )
        self.requested_cursor = requested_cursor
        self.oldest_cursor = oldest_cursor
