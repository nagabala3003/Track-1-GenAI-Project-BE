import logging

from fastapi import APIRouter

from app.agent import GeminiApiResponse, generate_response, trim_text_for_log
from app.models.request_models import QuestionRequest


logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask")
def ask(req: QuestionRequest) -> GeminiApiResponse:
    logger.info(
        "Incoming request: /ask text=%s question=%s",
        trim_text_for_log(req.text),
        trim_text_for_log(req.question),
    )
    prompt = (
        "Answer the question based on the context below:\n\n"
        f"{req.text}\n\nQuestion: {req.question}"
    )
    return generate_response(prompt)
