from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from kinetiq_v_vision.application.use_cases.create_analysis import (
    CreateAnalysisCommand,
    CreateAnalysisUseCase,
)
from kinetiq_v_vision.application.use_cases.poll_observations import (
    PollObservationsQuery,
    PollObservationsUseCase,
)
from kinetiq_v_vision.application.use_cases.select_target import (
    SelectTargetCommand,
    SelectTargetUseCase,
)
from kinetiq_v_vision.application.use_cases.stop_analysis import StopAnalysisUseCase
from kinetiq_v_vision.domain.entities import Observation
from kinetiq_v_vision.interfaces.rest.dto import (
    AnalysisStatusResponse,
    CandidateDTO,
    CandidateListResponse,
    CreateAnalysisRequest,
    CreateAnalysisResponse,
    HoldDTO,
    ObservationDTO,
    ObservationsPageResponse,
    RepetitionDTO,
    SelectTargetRequest,
    SelectTargetResponse,
)

router = APIRouter(prefix="/v1/analyses", tags=["analyses"])


def get_create_use_case() -> CreateAnalysisUseCase:
    raise NotImplementedError("Dependency injected at app factory")


def get_select_use_case() -> SelectTargetUseCase:
    raise NotImplementedError("Dependency injected at app factory")


def get_poll_use_case() -> PollObservationsUseCase:
    raise NotImplementedError("Dependency injected at app factory")


def get_stop_use_case() -> StopAnalysisUseCase:
    raise NotImplementedError("Dependency injected at app factory")


def get_repository():
    raise NotImplementedError("Dependency injected at app factory")


def _serialize_observation(obs: Observation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": obs.session_id,
        "epoch": obs.epoch,
        "sequence": obs.sequence,
        "timestamp_utc": obs.timestamp_utc.isoformat(),
        "target_person_id": obs.target_person_id,
        "exercise_key": obs.exercise_key,
        "exercise_version": obs.exercise_version,
        "tracking_state": obs.tracking_state.value,
        "visibility_state": obs.visibility_state.value,
        "reason_code": obs.reason_code.value,
    }

    if obs.repetitions is not None:
        payload["repetitions"] = [
            {
                "repetition_index": r.repetition_index,
                "start_timestamp": r.start_timestamp.isoformat(),
                "end_timestamp": r.end_timestamp.isoformat(),
                "confidence": r.confidence,
                "quality_score": r.quality_score,
                "form_flags": r.form_flags,
            }
            for r in obs.repetitions
        ]

    if obs.hold is not None:
        payload["hold"] = {
            "elapsed_seconds": obs.hold.elapsed_seconds,
            "is_holding": obs.hold.is_holding,
            "confidence": obs.hold.confidence,
            "stability_score": obs.hold.stability_score,
            "form_flags": obs.hold.form_flags,
        }

    return payload


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=CreateAnalysisResponse,
)
def create_analysis(
    request: CreateAnalysisRequest,
    use_case: CreateAnalysisUseCase = Depends(get_create_use_case),
) -> CreateAnalysisResponse:
    command = CreateAnalysisCommand(
        session_id=request.session_id,
        source_id=request.source_id,
        exercise_key=request.exercise_key,
        exercise_version=request.exercise_version,
        idempotency_key=request.idempotency_key,
    )
    analysis = use_case.execute(command)
    return CreateAnalysisResponse(
        analysis_id=analysis.analysis_id,
        session_id=analysis.session_id,
        epoch=analysis.epoch,
        state=analysis.state.value,
    )


@router.post(
    "/{analysis_id}/target",
    status_code=status.HTTP_200_OK,
    response_model=SelectTargetResponse,
)
def select_target(
    analysis_id: str,
    request: SelectTargetRequest,
    use_case: SelectTargetUseCase = Depends(get_select_use_case),
) -> SelectTargetResponse:
    command = SelectTargetCommand(
        analysis_id=analysis_id,
        candidate_id=request.candidate_id,
        expected_epoch=request.expected_epoch,
        idempotency_key=request.idempotency_key,
    )
    analysis = use_case.execute(command)
    return SelectTargetResponse(
        target_person_id=analysis.target_person_id or request.candidate_id,
        epoch=analysis.epoch,
        state=analysis.state.value,
    )


@router.get(
    "/{analysis_id}",
    status_code=status.HTTP_200_OK,
    response_model=AnalysisStatusResponse,
)
def get_analysis_status(
    analysis_id: str,
    repo=Depends(get_repository),
) -> AnalysisStatusResponse:
    analysis = repo.get_by_id(analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' was not found",
        )
    return AnalysisStatusResponse(
        analysis_id=analysis.analysis_id,
        session_id=analysis.session_id,
        epoch=analysis.epoch,
        state=analysis.state.value,
        last_valid_at=analysis.last_valid_at.isoformat() if analysis.last_valid_at else None,
    )


@router.get(
    "/{analysis_id}/candidates",
    status_code=status.HTTP_200_OK,
    response_model=CandidateListResponse,
)
def get_candidates(
    analysis_id: str,
    repo=Depends(get_repository),
) -> CandidateListResponse:
    analysis = repo.get_by_id(analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' was not found",
        )
    candidate_dtos = [
        CandidateDTO(
            candidate_id=c.candidate_id,
            bbox=[c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height],
            confidence=c.confidence,
            detected_at=c.detected_at.isoformat(),
        )
        for c in analysis.candidates.values()
    ]
    return CandidateListResponse(candidates=candidate_dtos)


@router.get(
    "/{analysis_id}/observations",
    status_code=status.HTTP_200_OK,
    response_model=ObservationsPageResponse,
)
def get_observations(
    analysis_id: str,
    after: str | None = Query(None, description="Cursor in '{epoch}:{sequence}' format"),
    limit: int = Query(50, ge=1, le=100),
    use_case: PollObservationsUseCase = Depends(get_poll_use_case),
) -> ObservationsPageResponse:
    query = PollObservationsQuery(
        analysis_id=analysis_id,
        after_cursor=after,
        limit=limit,
    )
    result = use_case.execute(query)
    serialized = [_serialize_observation(obs) for obs in result.observations]
    return ObservationsPageResponse(
        observations=serialized,
        next_cursor=result.next_cursor,
        has_more=result.has_more,
    )


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def stop_analysis(
    analysis_id: str,
    use_case: StopAnalysisUseCase = Depends(get_stop_use_case),
) -> None:
    use_case.execute(analysis_id)
