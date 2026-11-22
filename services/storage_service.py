"""SQLite storage: meetings, turns, chunks (with embeddings), notes, and an FTS5 keyword index.

Plain sqlite3 on purpose: bulk inserts are fast and FTS5 needs raw SQL anyway.
At scale this file is the one to swap for Postgres.
"""
import re
import sqlite3
from pathlib import Path

from config.settings import settings

MIN_CITE_WORDS = 6  # the brief: a quote must be at least 6 words, inside one turn

SCHEMA = """
CREATE TABLE IF NOT EXISTS meeting (
    id TEXT PRIMARY KEY, team TEXT, letter TEXT, n_turns INTEGER, source_hash TEXT
);
CREATE TABLE IF NOT EXISTS turn (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT, team TEXT, idx INTEGER, start TEXT, speaker TEXT,
    text TEXT, n_words INTEGER, citable INTEGER
);
CREATE INDEX IF NOT EXISTS ix_turn_meeting ON turn(meeting_id, idx);
CREATE INDEX IF NOT EXISTS ix_turn_team ON turn(team);
CREATE TABLE IF NOT EXISTS chunk (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT, team TEXT, first_turn INTEGER, last_turn INTEGER,
    text TEXT, embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_chunk_team ON chunk(team);
CREATE TABLE IF NOT EXISTS note (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT, team TEXT, turn_id INTEGER, kind TEXT, speaker TEXT, text TEXT
);
CREATE INDEX IF NOT EXISTS ix_note_team ON note(team);
CREATE VIRTUAL TABLE IF NOT EXISTS turn_fts USING fts5(fts_text, tokenize='porter');
"""


def get_conn() -> sqlite3.Connection:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def fts_normalise(text: str) -> str:
    """'L_C_D_' -> 'lcd' so keyword search matches how people say it."""
    return text.lower().replace("_", "")


def meeting_hash(meeting_id: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT source_hash FROM meeting WHERE id=?", (meeting_id,)).fetchone()
    return row["source_hash"] if row else None


def delete_meeting(conn: sqlite3.Connection, meeting_id: str) -> None:
    ids = [r["id"] for r in conn.execute("SELECT id FROM turn WHERE meeting_id=?", (meeting_id,))]
    conn.executemany("DELETE FROM turn_fts WHERE rowid=?", [(i,) for i in ids])
    for table in ("turn", "chunk", "note"):
        conn.execute(f"DELETE FROM {table} WHERE meeting_id=?", (meeting_id,))
    conn.execute("DELETE FROM meeting WHERE id=?", (meeting_id,))


def save_meeting(conn: sqlite3.Connection, meeting: dict) -> list[dict]:
    """Insert meeting + turns + FTS rows. Returns turns with their database ids."""
    delete_meeting(conn, meeting["id"])
    conn.execute(
        "INSERT INTO meeting VALUES (?,?,?,?,?)",
        (meeting["id"], meeting["team"], meeting["letter"], len(meeting["turns"]), meeting["source_hash"]),
    )
    saved = []
    for t in meeting["turns"]:
        cur = conn.execute(
            "INSERT INTO turn (meeting_id, team, idx, start, speaker, text, n_words, citable) VALUES (?,?,?,?,?,?,?,?)",
            (meeting["id"], meeting["team"], t["idx"], t["start"], t["speaker"], t["text"],
             t["n_words"], int(t["n_words"] >= MIN_CITE_WORDS)),
        )
        conn.execute("INSERT INTO turn_fts (rowid, fts_text) VALUES (?,?)", (cur.lastrowid, fts_normalise(t["text"])))
        saved.append({**t, "id": cur.lastrowid})
    return saved


def save_chunks(conn: sqlite3.Connection, meeting: dict, chunks: list[dict], vectors) -> None:
    for c, v in zip(chunks, vectors):
        conn.execute(
            "INSERT INTO chunk (meeting_id, team, first_turn, last_turn, text, embedding) VALUES (?,?,?,?,?,?)",
            (meeting["id"], meeting["team"], c["first_turn"], c["last_turn"], c["text"], v.tobytes()),
        )


def meetings_of_team(team: str) -> list[str]:
    """Meeting letters actually present, in order (a < b < c < d). Gaps are real: IS1002 has no 'a'."""
    with get_conn() as conn:
        rows = conn.execute("SELECT letter FROM meeting WHERE team=? ORDER BY letter", (team,)).fetchall()
    return [r["letter"] for r in rows]
