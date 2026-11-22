"""Find the evidence for a question. All plain code.

1. Team first: a question names its team, so we only look at that team's 3-4 meetings.
2. Two searches: meaning (embeddings) and keywords (FTS5). Merged with Reciprocal Rank Fusion.
3. Fairness rule: every meeting of the team contributes its best chunks, so "what changed over
   time" questions see the early AND the late discussion, not just the loudest meeting.
4. Everything is returned in meeting order (a, b, c, d).
"""
import re

import numpy as np

from config.settings import settings
from services.citation_service import STOP, _keywords
from services.index_service import embed
from services.storage_service import fts_normalise, get_conn

_team_cache: dict[str, dict] = {}


def load_team(team: str) -> dict:
    """Chunks + their vectors for one team (a few hundred rows, so brute force is exact and instant)."""
    if team not in _team_cache:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT c.id, c.meeting_id, m.letter, c.first_turn, c.last_turn, c.text, c.embedding "
                "FROM chunk c JOIN meeting m ON m.id=c.meeting_id WHERE c.team=? ORDER BY m.letter, c.id",
                (team,)).fetchall()
        _team_cache[team] = {
            "rows": [dict(r) for r in rows],
            "vectors": (np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                        if rows else np.zeros((0, 384), dtype=np.float32)),
        }
    return _team_cache[team]


def _fts_query(question: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", fts_normalise(question)) if w not in STOP and len(w) > 2]
    return " OR ".join(sorted(set(words))) if words else ""


def _keyword_turn_ids(team: str, question: str, limit: int = 40) -> list[int]:
    q = _fts_query(question)
    if not q:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT t.id FROM turn_fts f JOIN turn t ON t.id=f.rowid "
            "WHERE turn_fts MATCH ? AND t.team=? ORDER BY bm25(turn_fts) LIMIT ?", (q, team, limit)).fetchall()
    return [r["id"] for r in rows]


def retrieve_chunks(team: str, query: str, k: int | None = None, per_meeting: int | None = None) -> list[dict]:
    k = k or settings.chunks_per_question
    per_meeting = per_meeting or settings.chunks_per_meeting
    data = load_team(team)
    rows = data["rows"]
    if not rows:
        return []

    # (a) meaning search
    sims = data["vectors"] @ embed([query])[0]
    vec_rank = list(np.argsort(-sims))

    # (b) keyword search -> map matching turns to the chunk that contains them
    hit_turns = _keyword_turn_ids(team, query)
    kw_rank: list[int] = []
    for tid in hit_turns:
        for i, r in enumerate(rows):
            if r["first_turn"] <= tid <= r["last_turn"] and i not in kw_rank:
                kw_rank.append(i)

    # (c) Reciprocal Rank Fusion: score = sum of 1 / (60 + rank) over both lists
    score: dict[int, float] = {}
    for ranking in (vec_rank[:40], kw_rank[:40]):
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(score, key=score.get, reverse=True)

    # (d) fairness: best chunks overall + best chunks of every meeting
    chosen = list(ordered[:k])
    count: dict[str, int] = {}
    for i in ordered:
        mid = rows[i]["meeting_id"]
        count[mid] = count.get(mid, 0) + 1
        if count[mid] <= per_meeting and i not in chosen:
            chosen.append(i)

    return [rows[i] for i in sorted(chosen)]  # rows are already in meeting order


def chunk_turns(chunk: dict) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, meeting_id, speaker, text, citable FROM turn WHERE id BETWEEN ? AND ? ORDER BY id",
            (chunk["first_turn"], chunk["last_turn"])).fetchall()
    return [dict(r) for r in rows]


def retrieve_notes(team: str, query: str, limit: int = 12) -> list[dict]:
    """Notes of the team ranked by word overlap with the query, returned in meeting order."""
    want = _keywords(query)
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT n.id, n.meeting_id, n.turn_id, n.kind, n.speaker, n.text, m.letter "
            "FROM note n JOIN meeting m ON m.id=n.meeting_id WHERE n.team=?", (team,))]
    scored = [(len(want & _keywords(r["text"])), r) for r in rows]
    top = [r for s, r in sorted(scored, key=lambda x: -x[0])[:limit] if s > 0]
    return sorted(top, key=lambda r: (r["letter"], r["turn_id"]))


def format_evidence(chunks: list[dict], notes: list[dict]) -> str:
    """Text shown to the LLM. Only citable turns (>= 6 words) get an id the model may cite."""
    out: list[str] = []
    current = None
    seen: set[int] = set()  # a very long turn is split over several chunks; show it once
    for c in chunks:
        if c["meeting_id"] != current:
            current = c["meeting_id"]
            out.append(f"\n=== MEETING {current} ===")
        for t in chunk_turns(c):
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            tag = f"[{t['id']}] " if t["citable"] else "      "
            out.append(f"{tag}{t['speaker']}: {t['text']}")
        out.append("   ...")
    if notes:
        out.append("\n=== MEMORY NOTES (summaries, in meeting order; cite the turn id, not the note) ===")
        for n in notes:
            out.append(f"{n['meeting_id']} [{n['turn_id']}] {n['kind']} ({n['speaker']}): {n['text']}")
    return "\n".join(out)
