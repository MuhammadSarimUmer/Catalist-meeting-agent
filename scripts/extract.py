"""Step 2 of the pipeline: LLM turns each meeting into notes. Resumable and cached.

    python -m scripts.extract                    # all meetings not done yet
    python -m scripts.extract --team TS3012      # one team only
    python -m scripts.extract --limit 2          # quick test
Every LLM reply is cached in data/cache, so after a crash or a quota stop you just run it again.
"""
import argparse
import time

import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
from services import storage_service as db
from services.extract_service import extract_meeting
from services.llm_service import LLMError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", nargs="*", default=[], help="meeting ids to extract again from scratch")
    args = ap.parse_args()

    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT m.id FROM meeting m WHERE (?='' OR m.team=?) "
            "AND NOT EXISTS (SELECT 1 FROM note n WHERE n.meeting_id=m.id) ORDER BY m.id",
            (args.team, args.team)).fetchall()
    todo = [r["id"] for r in rows if r["id"] not in args.redo] + list(args.redo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} meeting(s) to extract")

    start = time.time()
    for i, mid in enumerate(todo, 1):
        try:
            n = extract_meeting(mid, refresh=mid in args.redo)
            print(f"[{i}/{len(todo)}] {mid}: {n} notes  ({time.time() - start:.0f}s)", flush=True)
        except LLMError as e:
            print(f"[{i}/{len(todo)}] {mid}: FAILED, will retry on next run ({e})", flush=True)


if __name__ == "__main__":
    main()
