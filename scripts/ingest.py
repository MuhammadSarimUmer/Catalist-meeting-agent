"""Step 1 of the pipeline: parse every transcript, index it (keywords + embeddings). No LLM.

    python -m scripts.ingest              # all meetings, skips unchanged ones
    python -m scripts.ingest --limit 3    # quick test
Safe to re-run: a meeting whose file hash has not changed is skipped (idempotent).
"""
import argparse
import time
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
from pathlib import Path

from config.settings import settings
from services import storage_service as db
from services.index_service import build_chunks, embed
from services.parse_service import parse_transcript


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db.init_db()
    files = sorted(Path(settings.transcripts_dir).glob("*.txt"))
    if args.limit:
        files = files[: args.limit]

    start, done, skipped, n_chunks = time.time(), 0, 0, 0
    for f in files:
        meeting = parse_transcript(f)
        if db.meeting_hash(meeting["id"]) == meeting["source_hash"]:
            skipped += 1
            continue
        with db.get_conn() as conn:
            turns = db.save_meeting(conn, meeting)
            chunks = build_chunks(turns)
            db.save_chunks(conn, meeting, chunks, embed([c["text"] for c in chunks]))
        done += 1
        n_chunks += len(chunks)
        print(f"{meeting['id']}: {len(turns)} turns, {len(chunks)} chunks")

    secs = time.time() - start
    print(f"\ningested={done} skipped={skipped} chunks={n_chunks} in {secs:.0f}s")


if __name__ == "__main__":
    main()
