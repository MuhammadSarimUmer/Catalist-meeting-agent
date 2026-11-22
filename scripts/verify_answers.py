"""Independent check of answers.json, the way the grader will do it.

Re-reads the raw transcript FILES (not our database) and confirms, for every citation:
  - the meeting exists and belongs to the question's team
  - the quote is >= 6 words and sits inside ONE speaker turn (case/spacing ignored)
    python -m scripts.verify_answers
"""
import json
import re
import sys
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
from pathlib import Path

HEADER = re.compile(r"^\d\d:\d\d:\d\d Speaker \w$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def turns_of(meeting_id: str) -> list[str]:
    raw = Path(f"data/transcripts/{meeting_id}.txt").read_text(encoding="utf-8").split("\n")
    out, i = [], 0
    while i < len(raw):
        if HEADER.match(raw[i].strip()):
            i += 1
            body = []
            while i < len(raw) and raw[i].strip() and not HEADER.match(raw[i].strip()):
                body.append(raw[i].strip())
                i += 1
            out.append(norm(" ".join(body)))
        else:
            i += 1
    return out


def main(path: str = "answers.json") -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    teams = {q["question_id"]: q["team"] for q in json.loads(Path("data/questions.json").read_text("utf-8"))["questions"]}
    total = bad = 0
    for a in data["answers"]:
        if not 1 <= len(a["citations"]) <= 3:
            print(f"{a['question_id']}: has {len(a['citations'])} citations (need 1-3)")
            bad += 1
        for c in a["citations"]:
            total += 1
            mid, quote = c["meeting_id"], c["quote"]
            problems = []
            if not mid.startswith(teams.get(a["question_id"], "")):
                problems.append("meeting not in the question's team")
            elif not Path(f"data/transcripts/{mid}.txt").exists():
                problems.append("meeting file missing")
            else:
                if len(quote.split()) < 6:
                    problems.append("fewer than 6 words")
                if not any(norm(quote) in t for t in turns_of(mid)):
                    problems.append("not found inside a single turn")
            if problems:
                bad += 1
                print(f"{a['question_id']} {mid}: {', '.join(problems)}: {quote[:70]}")
    print(f"\n{len(data['answers'])} answers, {total} citations checked, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
