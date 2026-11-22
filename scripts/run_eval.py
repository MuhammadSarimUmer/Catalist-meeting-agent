"""Score the agent on YOUR OWN hand-checked questions (evals/my_questions.jsonl).

    python -m scripts.run_eval
What it measures:
  citations_valid : do all quotes exist, >= 6 words, inside one turn?   (automatic, exact)
  facts_found     : share of your `must_contain` key facts that appear in the answer (automatic, crude)
  verdict         : YOU read answer vs gold and mark right / partial / wrong (the real judgment)
Output: evals/results.json. Fill in "verdict" and "why", then copy the numbers into evals/REPORT.md.
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
from pathlib import Path

from core.graph import ask
from scripts.verify_answers import norm, turns_of

SRC = Path("evals/my_questions.jsonl")
OUT = Path("evals/results.json")


def main() -> None:
    questions = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q["gold_answer"] != "TODO"]
    if not questions:
        print("No questions with a gold_answer yet. Fill in the gold_answer fields in evals/my_questions.jsonl first.")
        return

    old = {r["id"]: r for r in json.loads(OUT.read_text("utf-8"))} if OUT.exists() else {}
    results = []
    for q in questions:
        r = ask(q["question"], q["team"])
        valid = all(len(c["quote"].split()) >= 6 and any(norm(c["quote"]) in t for t in turns_of(c["meeting_id"]))
                    for c in r["citations"]) and bool(r["citations"])
        hits = [k for k in q["must_contain"] if k.lower() in r["answer"].lower()]
        facts = len(hits) / len(q["must_contain"]) if q["must_contain"] else None
        results.append({
            "id": q["id"], "team": q["team"], "question": q["question"], "gold_answer": q["gold_answer"],
            "agent_answer": r["answer"], "citations": r["citations"], "loops": r["loops"],
            "citations_valid": valid, "facts_found": facts,
            "verdict": old.get(q["id"], {}).get("verdict", "TODO"), "why": old.get(q["id"], {}).get("why", "")})
        print(f"{q['id']} valid={valid} facts={facts}\n  gold : {q['gold_answer']}\n  agent: {r['answer']}\n")
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n = len(results)
    print(f"citations valid: {sum(r['citations_valid'] for r in results)}/{n}")
    scored = [r["facts_found"] for r in results if r["facts_found"] is not None]
    if scored:
        print(f"average key facts found: {sum(scored) / len(scored):.0%}")
    print("Now open evals/results.json and set verdict (right/partial/wrong) + why for each.")


if __name__ == "__main__":
    main()
