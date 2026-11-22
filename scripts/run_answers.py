"""Answer the 25 questions and write the TS3012 brief -> answers.json

    python -m scripts.run_answers
    python -m scripts.run_answers --only Q05 Q13     # a few questions
    python -m scripts.run_answers --no-brief
Questions already answered in answers.json are kept unless you pass --redo.
"""
import argparse
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
from pathlib import Path

from core.graph import ask
from services.brief_service import build_brief

QUESTIONS = Path("data/questions.json")
OUT = Path("answers.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-brief", action="store_true")
    args = ap.parse_args()

    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"answers": [], "brief": ""}
    done = {a["question_id"]: a for a in result["answers"]}

    for q in spec["questions"]:
        qid = q["question_id"]
        if args.only and qid not in args.only:
            continue
        if qid in done and not args.redo and not args.only:
            continue
        r = ask(q["question"], q["team"])
        done[qid] = {"question_id": qid, "answer": r["answer"], "citations": r["citations"]}
        print(f"{qid} ({q['team']}) loops={r['loops']} cites={len(r['citations'])}\n  {r['answer']}\n", flush=True)
        ordered = [done[k["question_id"]] for k in spec["questions"] if k["question_id"] in done]
        OUT.write_text(json.dumps({"answers": ordered, "brief": result.get("brief", "")}, indent=2), encoding="utf-8")

    if not args.no_brief:
        result = json.loads(OUT.read_text(encoding="utf-8"))
        result["brief"] = build_brief(spec["brief_task"]["team"])
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("brief written")


if __name__ == "__main__":
    main()
