from pydantic import BaseModel, Field


class Citation(BaseModel):
    meeting_id: str
    quote: str


class AnswerItem(BaseModel):
    question_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class CitationRequest(BaseModel):
    """What the LLM returns for each citation: only WHICH turn, plus a short hint of the claim.
    The quote itself is copied by code (services/citation_service.py)."""
    turn_id: int
    hint: str = ""


class LLMAnswer(BaseModel):
    answer: str
    citations: list[CitationRequest] = Field(default_factory=list)
    enough_evidence: bool = True
    search_query: str = ""  # if evidence was missing: what to search for next


class AskRequest(BaseModel):
    question: str
    team: str | None = None
