from typing import TypedDict


class GraphState(TypedDict, total=False):
    question: str
    team: str
    query: str               # what we search for (the question, or a reformulated one)
    evidence: str            # text shown to the LLM
    valid_turn_ids: list[int]
    meetings_present: list[str]
    draft: dict              # LLM reply (answer + citation requests)
    answer: str
    citations: list[dict]    # final, code-verified citations
    errors: list[str]        # fed back to the LLM on retry
    loops: int
    status: str              # retrieving | answering | done | unknown
