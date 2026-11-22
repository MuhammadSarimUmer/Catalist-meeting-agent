# Meeting Memory Agent: notes for the Catalist team

A short guide for reviewers. For setup details see `README.md`; for design, evals and honest failures see `WRITEUP.md` and `evals/REPORT.md`.

## What it does
- Reads 138 AMI meeting transcripts and builds a searchable memory.
- Answers multi-meeting questions with 1-3 citations (meeting id plus an exact quote of at least 6 words from one speaker turn).
- Writes a pre-meeting brief for Team TS3012, using only the meetings before its final one.

## How it works (short)
1. **Ingest** (`scripts/ingest.py`): parse transcripts into speaker turns, split into chunks, embed locally, store in SQLite with a keyword index.
2. **Notes** (`scripts/extract.py`): an LLM writes short notes (decisions, objections, actions, numbers, changes), each tied to a real turn. Code rejects any note that does not point to a citable turn.
3. **Ask** (`core/graph.py`): find the team, retrieve evidence (embeddings + keywords, every meeting represented), the LLM answers with turn ids, then **code** copies the quote from the transcript and verifies it. If verification fails, it retries.
4. **Verify** (`scripts/verify_answers.py`): re-reads the raw transcripts and checks every citation independently.

## Key design choices
- Team-first filtering: a question about one team never sees another team's meetings.
- The LLM never writes quotes. Code copies them, so quotes are exact.
- Free-tier only (Gemini, Groq, local MiniLM embeddings), with rate limiting, caching and fallback models.
- Plain SQLite and plain functions: simple to read and easy to swap for Postgres later.

## Run it
```
pip install -r requirements.txt
copy .env.example .env        # then add your own keys
python -m scripts.ingest
python -m scripts.extract
python -m scripts.run_answers
python -m scripts.verify_answers
uvicorn main:app --reload     # API docs at /docs
```

## Deliverables
| Item | Where |
|---|---|
| Answers and brief | `answers.json` |
| Write-up | `WRITEUP.md` |
| Own evals and failures | `evals/` |
| Run instructions | `README.md` |

## Known limits
Quotes are guaranteed to exist, but not guaranteed to fully support the claim. Details and fixes are in `WRITEUP.md`.
