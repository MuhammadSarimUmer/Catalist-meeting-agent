"""Read the transcripts fast, to write and check your own eval answers by hand.

    python -m scripts.find ES2005 battery            # every turn of team ES2005 mentioning 'battery'
    python -m scripts.find ES2005 "power source" --context 1
Prints: meeting, turn id, speaker, text. (Plain keyword search, no LLM, so you can trust it.)
"""
import argparse

import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
from services.storage_service import fts_normalise, get_conn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("team")
    ap.add_argument("word")
    ap.add_argument("--context", type=int, default=0, help="also show N turns before/after")
    args = ap.parse_args()

    needle = fts_normalise(args.word)
    with get_conn() as conn:
        hits = conn.execute(
            "SELECT t.id, t.meeting_id, t.idx FROM turn t WHERE t.team=? AND lower(replace(t.text,'_','')) LIKE ? "
            "ORDER BY t.meeting_id, t.idx", (args.team, f"%{needle}%")).fetchall()
        print(f"{len(hits)} turn(s) in {args.team} mention '{args.word}'\n")
        for h in hits:
            rows = conn.execute(
                "SELECT id, idx, speaker, text, n_words FROM turn WHERE meeting_id=? AND idx BETWEEN ? AND ? ORDER BY idx",
                (h["meeting_id"], h["idx"] - args.context, h["idx"] + args.context)).fetchall()
            for r in rows:
                mark = ">>" if r["id"] == h["id"] else "  "
                cite = "" if r["n_words"] >= 6 else " (too short to cite)"
                print(f"{mark} {h['meeting_id']} turn {r['id']} {r['speaker']}{cite}: {r['text']}")
            print()


if __name__ == "__main__":
    main()
