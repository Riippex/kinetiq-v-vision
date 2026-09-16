from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreateAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    source_id: str
    exercise_key: str
    exercise_version: int = 1
    idempotency_key: str | None = None


class CreateAnalysisResponse(BaseModel):
    analysis_id: str
    session_id: str
    epoch: int
    state: str


class SelectTargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    expected_epoch: int
    idempotency_key: str | None = None


class SelectTargetResponse(BaseModel):
    target_person_id: str
    epoch: int
    state: str


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    session_id: str
    epoch: int
    state: str
    last_valid_at: str | None = None


class CandidateDTO(BaseModel):
    candidate_id: str
    bbox: list[float]
    confidence: float
    detected_at: str


class CandidateListResponse(BaseModel):
    candidates: list[CandidateDTO]


class RepetitionDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repetition_index: int
    start_timestamp: str
    end_timestamp: str
    confidence: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    form_flags: list[str] = Field(default_factory=list)


class HoldDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elapsed_seconds: float = Field(ge=0.0)
    is_holding: bool
    confidence: float = Field(ge=0.0, le=1.0)
    stability_score: float = Field(ge=0.0, le=1.0)
    form_flags: list[str] = Field(default_factory=list)


class ObservationDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    epoch: int = Field(ge=1)
    sequence: int = Field(ge=1)
    timestamp_utc: str
    target_person_id: str
    exercise_key: str
    exercise_version: int = Field(ge=1)
    tracking_state: str
    visibility_state: str
    reason_code: str
    repetitions: list[RepetitionDTO] | None = None
    hold: HoldDTO | None = None


class ObservationsPageResponse(BaseModel):
    observations: list[dict[str, Any]]
    next_cursor: str | None = None
    has_more: bool = False
