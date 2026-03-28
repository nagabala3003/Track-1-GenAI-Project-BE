from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_MIN_STRING_LENGTH: Final[int] = 1


class TextRequest(BaseModel):
    """Request model for text summarization."""

    model_config: ConfigDict = ConfigDict(str_strip_whitespace=True, extra="forbid")
    text: str = Field(min_length=DEFAULT_MIN_STRING_LENGTH)

    @field_validator("text")
    @classmethod
    def validate_text_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text must not be empty.")
        return value


class QuestionRequest(BaseModel):
    """Request model for question answering from provided text context."""

    model_config: ConfigDict = ConfigDict(str_strip_whitespace=True, extra="forbid")
    text: str = Field(min_length=DEFAULT_MIN_STRING_LENGTH)
    question: str = Field(min_length=DEFAULT_MIN_STRING_LENGTH)

    @field_validator("text")
    @classmethod
    def validate_context_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text must not be empty.")
        return value

    @field_validator("question")
    @classmethod
    def validate_question_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must not be empty.")
        return value


class PromptRequest(BaseModel):
    """Request model for general content generation."""

    model_config: ConfigDict = ConfigDict(str_strip_whitespace=True, extra="forbid")
    prompt: str = Field(min_length=DEFAULT_MIN_STRING_LENGTH)

    @field_validator("prompt")
    @classmethod
    def validate_prompt_is_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("prompt must not be empty.")
        return value
