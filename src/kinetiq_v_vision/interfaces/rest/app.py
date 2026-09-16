import uuid

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from kinetiq_v_vision.domain.exceptions import (
    AnalysisNotFoundError,
    CursorExpiredError,
    InvalidEpochError,
    InvalidTargetError,
    StaleEpochError,
)
from kinetiq_v_vision.interfaces.rest.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kinetiq V Vision Engine",
        version="0.1.0",
        description="Independent REST service for real-time movement, pose, and exercise analysis",
    )

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = corr_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = corr_id
        return response

    @app.exception_handler(StaleEpochError)
    async def stale_epoch_handler(
        request: Request, exc: StaleEpochError
    ) -> JSONResponse:
        corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {
                    "code": "STALE_EPOCH",
                    "message": str(exc),
                    "correlation_id": corr_id,
                    "retryable": False,
                    "details": {
                        "expected_epoch": exc.expected_epoch,
                        "current_epoch": exc.current_epoch,
                    },
                }
            },
        )

    @app.exception_handler(CursorExpiredError)
    async def cursor_expired_handler(
        request: Request, exc: CursorExpiredError
    ) -> JSONResponse:
        corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_410_GONE,
            content={
                "error": {
                    "code": "CURSOR_EXPIRED",
                    "message": str(exc),
                    "correlation_id": corr_id,
                    "retryable": False,
                    "details": {
                        "requested_cursor": exc.requested_cursor,
                        "oldest_cursor": exc.oldest_cursor,
                    },
                }
            },
        )

    @app.exception_handler(AnalysisNotFoundError)
    async def not_found_handler(
        request: Request, exc: AnalysisNotFoundError
    ) -> JSONResponse:
        corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": "ANALYSIS_NOT_FOUND",
                    "message": str(exc),
                    "correlation_id": corr_id,
                    "retryable": False,
                    "details": {"analysis_id": exc.analysis_id},
                }
            },
        )

    @app.exception_handler(InvalidTargetError)
    async def invalid_target_handler(
        request: Request, exc: InvalidTargetError
    ) -> JSONResponse:
        corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "INVALID_TARGET",
                    "message": str(exc),
                    "correlation_id": corr_id,
                    "retryable": False,
                    "details": {"candidate_id": exc.candidate_id},
                }
            },
        )

    @app.exception_handler(InvalidEpochError)
    async def invalid_epoch_handler(
        request: Request, exc: InvalidEpochError
    ) -> JSONResponse:
        corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "INVALID_EPOCH",
                    "message": str(exc),
                    "correlation_id": corr_id,
                    "retryable": False,
                    "details": {"epoch": exc.epoch},
                }
            },
        )

    @app.get("/health", tags=["system"])
    async def health_check():
        return {"status": "ok", "service": "kinetiq-v-vision"}

    app.include_router(router)
    return app
