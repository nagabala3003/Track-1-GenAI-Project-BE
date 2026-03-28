import logging

from fastapi import APIRouter

from app.agent import GeminiApiResponse, generate_response, trim_text_for_log
from app.models.request_models import PromptRequest


logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate")
def generate(req: PromptRequest) -> GeminiApiResponse:
    logger.info("Incoming request: /generate prompt=%s", trim_text_for_log(req.prompt))
    prompt = f"Generate a helpful response for the following prompt:\n\n{req.prompt}"
    return generate_response(prompt)
