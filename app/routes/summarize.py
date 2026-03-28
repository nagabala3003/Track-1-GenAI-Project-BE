import logging

from fastapi import APIRouter

from app.agent import GeminiApiResponse, generate_response, trim_text_for_log
from app.models.request_models import TextRequest


logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/summarize")
def summarize(req: TextRequest) -> GeminiApiResponse:
    logger.info("Incoming request: /summarize text=%s", trim_text_for_log(req.text))
    prompt = f"Summarize the following text in 3 concise lines:\n\n{req.text}"
    return generate_response(prompt)
