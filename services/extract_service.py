"""LLM step of ingestion: read a window of a meeting, write short NOTES (the memory).

A note = one decision / proposal / objection / action / number, pointing at the turn that says it.
Notes are what the brief is built from, and extra evidence for answering questions.
The LLM does the understanding; plain code validates every note before it is stored.
"""
from pydantic import BaseModel, ValidationError

from config.settings import settings
from services.llm_service import LLMError, call_json
from services.storage_service import get_conn

KINDS = {"decision", "proposal", "objection", "action", "number", "change"}

SYSTEM = """You turn a noisy meeting transcript into short memory notes. The meeting is a design team
(project manager, industrial designer, user interface designer, marketing) designing a TV remote control.
Each line is: [turn_id] Speaker X: words.

Write a note ONLY for things that matter later:
- decision: the team settles on something ("we'll use rubber")
- proposal: someone suggests an idea, even if not accepted
- objection: someone disagrees, or gives a reason against
- action: someone says they WILL do or check something (who, what)
- number: a price, cost, budget, percentage or date
- change: the team drops or replaces an earlier idea

Rules:
- text: plain words, max 30 words, keep names, figures and the reason given.
- speaker: the speaker label from the line, e.g. "Speaker B".
- turn_id: the turn where it is best stated. Prefer a long turn over a short "Yeah" turn.
- Skip small talk, jokes, tool training and filler. If nothing matters, return an empty list.
Return JSON only: {"notes":[{"kind":"decision","speaker":"Speaker A","text":"...","turn_id":123}]}"""


class Note(BaseModel):
    kind: str
    speaker: str
    text: str
    turn_id: int


def make_windows(turns: list[dict]) -> list[list[dict]]:
    """Group consecutive turns into windows of ~window_words words. Tiny turns ("Hmm.") are skipped."""
    windows: list[list[dict]] = []
    cur: list[dict] = []
    words = 0
    for t in turns:
        if t["n_words"] < settings.min_extract_turn_words:
            continue
        if cur and words + t["n_words"] > settings.window_words:
            windows.append(cur)
            cur, words = [], 0
        cur.append(t)
        words += t["n_words"]
    if cur:
        windows.append(cur)
    return windows


def window_text(window: list[dict]) -> str:
    return "\n".join(f"[{t['id']}] {t['speaker']}: {t['text']}" for t in window)


def validate_notes(raw: dict | list, window: list[dict]) -> list[Note]:
    """Keep only well-formed notes whose turn is in this window and can be cited (>= 6 words)."""
    citable = {t["id"] for t in window if t["citable"]}
    items = raw if isinstance(raw, list) else raw.get("notes", [])  # models sometimes skip the wrapper
    good: list[Note] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            n = Note.model_validate(item)
        except ValidationError:
            continue
        n.kind = n.kind.lower().strip()
        if n.kind in KINDS and n.turn_id in citable and n.text.strip():
            good.append(n)
    return good


def extract_meeting(meeting_id: str, refresh: bool = False) -> int:
    """Extract notes for one meeting. Returns how many were stored. Safe to re-run (cached)."""
    with get_conn() as conn:
        turns = [dict(r) for r in conn.execute(
            "SELECT * FROM turn WHERE meeting_id=? ORDER BY idx", (meeting_id,))]
        team = turns[0]["team"]
        all_notes: list[Note] = []
        failed = 0
        for window in make_windows(turns):
            try:
                raw = call_json(settings.extract_model, SYSTEM, window_text(window), max_out=2000,
                                fallback=settings.extract_fallback_model, refresh=refresh)
            except LLMError:
                failed += 1
                continue
            all_notes += validate_notes(raw, window)

        if failed:  # do not store a half-finished meeting: it would look finished next time
            raise LLMError(f"{meeting_id}: {failed} window(s) failed")
        conn.execute("DELETE FROM note WHERE meeting_id=?", (meeting_id,))
        conn.executemany(
            "INSERT INTO note (meeting_id, team, turn_id, kind, speaker, text) VALUES (?,?,?,?,?,?)",
            [(meeting_id, team, n.turn_id, n.kind, n.speaker, n.text.strip()) for n in all_notes])
    return len(all_notes)
