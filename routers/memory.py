from fastapi import APIRouter, HTTPException

from core.graph import ask
from core.schemas import AskRequest
from services.brief_service import build_brief
from services.storage_service import get_conn

router = APIRouter()


@router.post("/ask")
def ask_question(req: AskRequest) -> dict:
    """Ask about one team's meetings. The team is read from the question (e.g. 'TS3012') or the `team` field."""
    result = ask(req.question, req.team)
    if not result["team"]:
        raise HTTPException(400, result["answer"])
    return result


@router.get("/brief/{team}")
def brief(team: str) -> dict:
    return {"team": team, "brief": build_brief(team)}


@router.get("/notes/{team}")
def notes(team: str) -> list[dict]:
    """The memory notes of a team in meeting order, for inspection."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT n.meeting_id, n.turn_id, n.kind, n.speaker, n.text FROM note n "
            "JOIN meeting m ON m.id=n.meeting_id WHERE n.team=? ORDER BY m.letter, n.turn_id", (team,)).fetchall()
    return [dict(r) for r in rows]
