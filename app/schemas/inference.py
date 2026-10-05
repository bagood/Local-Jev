from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    choices: list[str] = Field(
        ...,
        min_length=2,
        max_length=26,
        description="Choices to score, in display order.",
        examples=[["Good", "Bad"]],
    )
    system_prompt: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class ChoiceScore(BaseModel):
    choice: str
    probability: float


class InferenceResponse(BaseModel):
    predicted_choice: str
    scores: list[ChoiceScore]
