"""The pre-meeting brief for one team: decisions, what changed, open actions, open disagreements.

The LLM reads the team's memory notes (a few thousand tokens, NOT the transcripts) in meeting order and
writes bullets that point at note ids. Code then turns each note id into a verified quote.
Bullets without a valid citation are dropped, so nothing in the brief is unsupported.
"""
from config.settings import settings
from services.citation_service import clean_text, make_citation
from services.llm_service import call_json
from services.storage_service import get_conn, meetings_of_team

SYSTEM = """You write a briefing for a design team to read BEFORE their next meeting. You are given the memory
notes of their earlier meetings, in order. Each note: [note_id] meeting | kind | speaker | text.

Write four sections, each a list of bullets:
1. "Decided": what the team has settled on (the latest position on each topic).
2. "What changed": ideas that were dropped or replaced, with the reason, and which meeting.
3. "Open actions": things someone promised to do or check, with the owner (speaker letter) and whether a later
   note shows it was done. Only list what looks unfinished.
4. "Open disagreements": objections or conflicts that no later note resolves.

Rules: every bullet is 1-2 sentences, names the meeting letter, and lists 1-2 note_ids that support it.
Use only the notes. Do not invent anything. Skip a section's bullets rather than guess.
Return JSON only:
{"sections":[{"title":"Decided","bullets":[{"text":"...","note_ids":[12,15]}]}]}"""

TITLES = ["Decided", "What changed", "Open actions", "Open disagreements"]


def build_brief(team: str, max_notes: int = 220) -> str:
    """Brief for the team's NEXT meeting = everything before the last meeting. The last meeting is the
    one being prepared for, so its notes must not be used (that would be reading the future)."""
    letters = meetings_of_team(team)
    before = letters[:-1]  # all meetings except the final one
    with get_conn() as conn:
        notes = [dict(r) for r in conn.execute(
            "SELECT n.id, n.meeting_id, n.turn_id, n.kind, n.speaker, n.text FROM note n "
            "JOIN meeting m ON m.id=n.meeting_id WHERE n.team=? AND m.letter IN (%s) "
            "ORDER BY m.letter, n.turn_id" % ",".join("?" * len(before)), (team, *before))] if before else []
    if not notes:
        return f"No notes found for {team}. Run scripts.extract first."
    notes = notes[:max_notes]
    by_id = {n["id"]: n for n in notes}

    lines = [f"[{n['id']}] {n['meeting_id']} | {n['kind']} | {n['speaker']} | {n['text']}" for n in notes]
    user = (f"Team {team}. Meetings covered by these notes: {', '.join(before)}.\n"
            f"The team is about to hold its final meeting ({letters[-1]}).\n\nNOTES:\n" + "\n".join(lines))
    raw = call_json(settings.brief_model, SYSTEM, user, max_out=3000, fallback=settings.answer_model)

    out = [f"BRIEF FOR TEAM {team}", ""]
    sections = {s.get("title", ""): s.get("bullets", []) for s in raw.get("sections", [])}
    for title in TITLES:
        bullets = []
        for b in sections.get(title, []):
            cites = []
            for nid in b.get("note_ids", [])[:2]:
                n = by_id.get(nid)
                c = n and make_citation(n["turn_id"], b["text"] + " " + n["text"], team)
                if c:
                    cites.append(f'[{c["meeting_id"]}: "{c["quote"]}"]')
            if cites and b.get("text"):  # drop anything we cannot back with a verified quote
                bullets.append(f"- {clean_text(b['text'].strip())} {' '.join(cites)}")
        out.append(f"{title.upper()}")
        out += bullets or ["- (nothing supported by the notes)"]
        out.append("")
    return "\n".join(out).strip()
