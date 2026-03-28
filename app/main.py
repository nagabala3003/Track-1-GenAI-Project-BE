import logging
import os
from collections.abc import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routes.ask import router as ask_router
from app.routes.generate import router as generate_router
from app.routes.summarize import router as summarize_router

from app.agent import trim_text_for_log
from app.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger: logging.Logger = logging.getLogger(__name__)


app = FastAPI(title="GenAI Text Processing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_incoming_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    body_bytes: bytes = await request.body()
    body_text: str = body_bytes.decode("utf-8", errors="ignore") if body_bytes else ""

    if body_text:
        logger.info(
            "Incoming request: %s %s body=%s",
            request.method,
            request.url.path,
            trim_text_for_log(body_text),
        )
    else:
        logger.info("Incoming request: %s %s", request.method, request.url.path)

    return await call_next(request)


@app.get("/health")
def read_health() -> dict[str, str]:
    """Health check endpoint for deployment readiness."""
    return {"status": "healthy"}


@app.get("/")
def read_root() -> dict[str, str]:
    """Return API health for basic browser checks."""
    return {"status": "ok"}


@app.get("/favicon.ico")
def read_favicon() -> Response:
    """Silence browser favicon requests."""
    return Response(status_code=204)

def build_error_message_from_validation(
    validation_error: RequestValidationError,
) -> str:
    error_messages: list[str] = []
    for error in validation_error.errors():
        message: object = error.get("msg")
        if isinstance(message, str) and message:
            error_messages.append(message)
    return "; ".join(error_messages) if error_messages else "Invalid request."


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    error_message: str = build_error_message_from_validation(exc)
    logger.warning(
        "Request validation failed for %s: %s",
        request.url.path,
        error_message,
    )
    return JSONResponse(
        status_code=422,
        content={"status": "error", "data": None, "error": error_message},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    logger.warning("HTTP error for %s: %s", request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "data": None, "error": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled error for %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "data": None, "error": "Internal server error"},
    )

app.include_router(summarize_router)
app.include_router(ask_router)
app.include_router(generate_router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
