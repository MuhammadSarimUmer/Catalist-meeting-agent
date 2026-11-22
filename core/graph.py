"""The question-answering agent as a LangGraph state machine (same style as the receipt agent).

    route_team -> retrieve -> answer -> verify --ok--------------> END
                     ^                     |
                     +---- retry (max N) --+   (bad citations, or the model asked for more evidence)
"""
import re

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import ValidationError

from config.settings import settings
from core.schemas import LLMAnswer
from core.state import GraphState
from services.citation_service import clean_text, make_citation
from services.llm_service import LLMError, call_json
from services.retrieval_service import format_evidence, retrieve_chunks, retrieve_notes, chunk_turns
from services.storage_service import meetings_of_team

TEAM_RE = re.compile(r"\b((?:ES|IS|TS)\d{4})\b")

SYSTEM = """You answer questions about the meetings of ONE design team (they design a TV remote control).
You are given transcript excerpts in meeting order (a = first meeting, then b, c, d) plus short memory notes.
Some meetings may be missing from the corpus; the list of available meetings is given.

How to answer:
- Answer in 1 to 3 plain sentences. Name the speaker letter (and the role if the text makes it clear).
- Ideas change over time. Compare the meetings in order and report what was FIRST, what CHANGED, and what
  is FINAL, with the reason given. The latest meeting wins when the team changed its mind.
- Use only the given text. Never invent figures or names.
- Never write turn ids or brackets like [123] inside "answer"; ids belong only in "citations".
- Read each line's speaker label. A reply comes from a DIFFERENT speaker than the one before it, so do not
  credit one speaker with both a proposal and the objection to it unless the lines show that.
- Say "not in the corpus" for a meeting that is missing, instead of guessing what happened there.
- Cite 1 to 3 turns. Cite only turns that show an id like [123]. Turns without an id cannot be cited.
  For each citation give "hint": a few words of the claim that turn supports.
- If the excerpts do not contain the answer, set enough_evidence=false and put a better search phrase
  in search_query (words people would actually say in the meeting).
Return JSON only:
{"answer":"...","citations":[{"turn_id":123,"hint":"..."}],"enough_evidence":true,"search_query":""}"""


def _team_from(question: str, given: str | None) -> str | None:
    m = TEAM_RE.search(question)
    return m.group(1) if m else given


@traceable(name="route_team")
def route_team(state: GraphState) -> dict:
    team = _team_from(state["question"], state.get("team"))
    if not team or not meetings_of_team(team):
        return {"status": "unknown", "answer": "I could not tell which team this question is about.",
                "citations": []}
    return {"team": team, "query": state["question"], "loops": 0, "errors": [], "status": "retrieving",
            "meetings_present": meetings_of_team(team)}


@traceable(name="retrieve")
def retrieve(state: GraphState) -> dict:
    wider = state.get("loops", 0)  # each retry looks at more evidence
    chunks = retrieve_chunks(state["team"], state["query"], k=8 + 4 * wider, per_meeting=2 + wider)
    notes = retrieve_notes(state["team"], state["query"])
    valid = [t["id"] for c in chunks for t in chunk_turns(c) if t["citable"]]
    return {"evidence": format_evidence(chunks, notes), "valid_turn_ids": valid, "status": "answering"}


@traceable(name="answer")
def answer(state: GraphState) -> dict:
    present = ", ".join(state["meetings_present"])
    user = (f"Team: {state['team']}\nMeetings available: {present}\n\nQUESTION: {state['question']}\n\n"
            f"EXCERPTS:\n{state['evidence']}")
    if state.get("errors"):
        user += "\n\nFIX THESE PROBLEMS FROM YOUR LAST TRY:\n- " + "\n- ".join(state["errors"])
    try:
        raw = call_json(settings.answer_model, SYSTEM, user, max_out=900,
                        fallback=settings.extract_model)
        draft = LLMAnswer.model_validate(raw)
    except (LLMError, ValidationError) as e:
        return {"draft": {}, "errors": [f"bad reply: {e}"], "status": "verifying"}
    return {"draft": draft.model_dump(), "status": "verifying"}


@traceable(name="verify")
def verify(state: GraphState) -> dict:
    """Pure code: copy the quotes out of the cited turns and check them."""
    draft = state.get("draft") or {}
    loops = state.get("loops", 0)
    if not draft:
        if loops + 1 >= settings.max_agent_loops:
            return {"answer": "The model did not return a usable answer.", "citations": [], "status": "done"}
        return {"errors": state["errors"], "loops": loops + 1, "status": "retrying"}

    citations, errors, seen = [], [], set()
    for c in draft.get("citations", []):
        if c["turn_id"] not in state["valid_turn_ids"]:
            errors.append(f"turn {c['turn_id']} is not a citable turn in the excerpts")
            continue
        cite = make_citation(c["turn_id"], c.get("hint", "") + " " + draft["answer"], state["team"])
        if cite and (cite["meeting_id"], cite["quote"]) not in seen:
            seen.add((cite["meeting_id"], cite["quote"]))
            citations.append(cite)
    citations = citations[:3]

    if draft.get("enough_evidence", True) and citations:
        return {"answer": clean_text(draft["answer"]), "citations": citations, "status": "done"}

    if loops + 1 >= settings.max_agent_loops:  # out of tries: be honest instead of guessing
        text = draft.get("answer") or "The transcripts do not contain enough evidence to answer this."
        return {"answer": clean_text(text), "citations": citations, "status": "done"}

    errors = errors or ["give at least one valid citation"]
    query = draft.get("search_query") or state["query"]
    return {"errors": errors, "query": query, "loops": loops + 1, "status": "retrying"}


def after_route(state: GraphState) -> str:
    return END if state["status"] == "unknown" else "retrieve"


def after_verify(state: GraphState) -> str:
    return END if state["status"] == "done" else "retrieve"


_graph = StateGraph(GraphState)
_graph.add_node("route_team", route_team)
_graph.add_node("retrieve", retrieve)
_graph.add_node("answer", answer)
_graph.add_node("verify", verify)
_graph.add_edge(START, "route_team")
_graph.add_conditional_edges("route_team", after_route, {"retrieve": "retrieve", END: END})
_graph.add_edge("retrieve", "answer")
_graph.add_edge("answer", "verify")
_graph.add_conditional_edges("verify", after_verify, {"retrieve": "retrieve", END: END})
agent = _graph.compile()


def ask(question: str, team: str | None = None) -> dict:
    out = agent.invoke({"question": question, "team": team or ""})
    return {"answer": out.get("answer", ""), "citations": out.get("citations", []),
            "team": out.get("team"), "loops": out.get("loops", 0)}
