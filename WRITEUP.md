# Catalist build test: write-up

**Candidate:** Muhammad | **Corpus:** 138 AMI meetings (676,745 words, 48,148 turns) | **Code:** this repo

## What I built
An agent that ingests the transcripts into a searchable memory, answers questions about one team's meetings with exact citations, and writes a pre-meeting brief. `answers.json` was produced by the agent, and an independent script (`scripts/verify_answers.py`) re-reads the raw transcript files and confirms every citation (25 answers, 51 citations, 0 problems).

## Architecture
1. **Ingest (code, no LLM):** parse turns, mark turns of 6+ words as citable, chunk to ~150 words, embed (MiniLM, CPU), build a keyword index (SQLite FTS5). Hash per file, so re-runs skip unchanged meetings. ~10 min for the slice.
2. **Notes (LLM, once per meeting):** Gemini reads ~1,500-word windows and writes short notes (decision, proposal, objection, action, number, change), each tied to a turn. Code rejects any note whose turn is not in the window or not citable.
3. **Ask (LangGraph):** route to the team named in the question, retrieve from that team only (meaning + keyword search merged by rank, and every meeting contributes so the agent sees how an idea changed), answer, verify, and loop with wider evidence if citations are missing or evidence is thin (max 3).
4. **Brief:** notes from the meetings *before the last one* -> LLM drafts four sections -> code attaches a verified quote to each bullet and drops any bullet it cannot back.

**What is current:** evidence is always shown in meeting order (a, b, c, d) and the latest meeting wins when the team changed its mind. Missing meetings (`IS1002` has no a, `IS1005` has no d) are listed in the prompt so the model says "not in the corpus" instead of guessing.

## Model vs code
- **Code:** parsing, citable-turn rule, chunking, embeddings, team routing, keyword search, rank merging, meeting ordering, **copying and checking quotes**, rate limiting, caching. Reason: exact, testable, free, and the grader is an exact-match check, so a model must not judge its own quotes. Half of all turns (50.5%) are under 6 words and cannot be cited, which makes a free-text "quote" from the model risky.
- **LLM:** finding decisions/actions/objections in messy speech, composing 1-3 sentence answers and the brief, and deciding that evidence is not enough. Reason: these need language understanding.

## Tools and where they broke
| Tool | What happened |
|---|---|
| Groq `gpt-oss` models | 8,000 tokens/min: bursts hit 429. Groq counts *requested* output tokens, so uncapped calls were counted double. `gpt-oss-20b` returned empty JSON until reasoning effort was lowered. Used for answers (few tokens per call). |
| Gemini `3.1-flash-lite` | Fast (15 req/min), but the free tier has a **500 requests/day per model** cap. The ~490 extraction windows used it up near the end; calls fell back to Groq, which was slower and wrote different notes. Fixed with a fallback chain (second Gemini model first) and by re-extracting the affected meetings. |
| Hugging Face hosted embeddings | 402 Payment Required. Gemini embeddings: 100 texts/min (~1 h). Used MiniLM on CPU instead (~10 min, no quota). |
| MiniLM | 256-token input limit, so chunks are <= 150 words. |
| LLM JSON | Sometimes the wrong shape (bare list) or fancy characters; validated with Pydantic and cleaned in code. |

## Honest limits
Quotes are always exact, but a quote can be only loosely relevant to the claim. Short decisions ("Agreed.") cannot be cited directly. Speaker attribution can be wrong.

## Evaluation results (details in `evals/REPORT.md`)
I wrote 13 questions on teams outside the 25 graded ones, with gold answers taken from the raw transcripts (supporting turn ids are stored with each question; the gold answers were prepared with an AI assistant's help and are meant to be re-checked).
- Citations valid: **13/13** (exact, automatic).
- Answer correctness (read by hand): **8 right, 3 partial, 2 wrong** (about 62% fully right).
- Three failures: **(1)** E08 said "palm-shaped" when the team chose a banana shape, so the quote was real but did not support the claim; **(2)** E11 claimed a power plan was settled when the last meeting still discussed both battery and kinetic; **(3)** E12/E13 gave the right final fact but missed earlier meetings (a research promise, a kinetic consensus).
- Takeaway: verified quotes do not mean correct answers. The next fixes are a claim-vs-quote check and a "tentative vs agreed" flag on notes.

## Scale: 1,000 and 10,000 meetings
- Answering touches one team's meetings, so it stays fast regardless of corpus size.
- Ingest measured at ~4.3 s per meeting on CPU (~72 min per 1,000). Notes: ~3.5 LLM windows per meeting, so ~3,500 calls per 1,000 meetings. That is 7 model-days on the free tier, so production needs a paid tier or batch API. (Dollar cost: not measured, estimate only.)
- Production: queue of workers (parse -> embed -> notes), Postgres + pgvector, batch APIs, extract notes only for active teams, monitor citation-failure rate and 429 rates, treat transcript text as untrusted input.
