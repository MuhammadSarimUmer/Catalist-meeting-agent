# 🧠 Catalist Meeting Memory Agent

An agent that turns 138 real meeting transcripts (AMI corpus) into **memory you can ask questions of**, answers with **exact, code-verified citations**, and writes a **pre-meeting brief** for a team.

Built with **LangGraph** as a real state machine (route, retrieve, answer, verify, retry). The LLM understands the language; plain code does everything exact: parsing, search, ordering, and copying and checking quotes.

## Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langgraph&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq" />
  <img src="https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini" />
  <img src="https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic" />
</p>

## How it works

1. **Ingest (code, no LLM):** parse each transcript into speaker turns, chunk them (~150 words), embed on CPU (`all-MiniLM-L6-v2`), and build a keyword index (SQLite FTS5). A file hash makes re-runs skip unchanged meetings.
2. **Extract (LLM, once per meeting):** Gemini reads ~1,500-word windows and writes short **notes** (decision, proposal, objection, action, number, change), each pointing at the turn that says it. Code drops any note it cannot verify.
3. **Ask (LangGraph):** the team is read from the question. Retrieval searches only that team's meetings by **meaning and keywords**, merges the rankings, and guarantees every meeting contributes, so the agent sees an idea *change over time*. The model answers from the evidence in meeting order and names the **turn ids** that support it.
4. **Verify (code):** for each turn id, code **copies** the best matching span out of the transcript and checks it (>= 6 words, exact, inside one turn). Missing citations or thin evidence send the run back to retrieve with wider evidence (max 3 tries).
5. **Brief:** notes from every meeting *before the last one* go to the model; code attaches a verified quote to each bullet and drops bullets it cannot back.

See [PLAN.md](PLAN.md) for the reasoning, measured limits, the full decision log and interview prep.

## Project structure

```
Catalist/
├── main.py                  # FastAPI app
├── config/settings.py       # Pydantic settings (.env)
├── core/                    # schemas, state, LangGraph agent
├── services/                # parse, storage, index, llm, extract, retrieval, citation, brief
├── routers/                 # /health, /ask, /brief/{team}, /notes/{team}
├── scripts/                 # ingest, extract, run_answers, verify_answers, find, run_eval, probes
├── evals/                   # my_questions.jsonl, results.json, REPORT.md
└── data/                    # transcripts, questions.json, SQLite db, LLM cache (generated)
```

## Running locally

```bash
python -m venv cenv
.\cenv\Scripts\activate          # macOS/Linux: source cenv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # add GROQ_API_KEY and GOOGLE_API_KEY

# put the unzipped transcripts in data/transcripts/ and questions.json in data/

python -m scripts.ingest         # ~10 min, no LLM, safe to re-run
python -m scripts.extract        # LLM notes, resumable (Ctrl+C is safe, results are cached)
python -m scripts.run_answers    # 25 answers + TS3012 brief -> answers.json
python -m scripts.verify_answers # independent citation check against the raw files
```

Optional API: `uvicorn main:app --reload`, then `POST /ask {"question": "...", "team": "TS3012"}` or `GET /brief/TS3012`. Docs at `/docs`.

Handy: `python -m scripts.find ES2005 battery --context 1` prints every turn of a team that mentions a word (for writing your own eval answers).

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | required | [console.groq.com](https://console.groq.com) |
| `GOOGLE_API_KEY` | | [aistudio.google.com](https://aistudio.google.com) |
| `EXTRACT_MODEL` | `gemini-3.1-flash-lite` | Bulk note extraction (15 requests/min free) |
| `EXTRACT_FALLBACK_MODEL` | `openai/gpt-oss-20b` | Used if the primary fails |
| `EXTRACT_RPM` | `12` | Stay under the free limit |
| `ANSWER_MODEL` | `openai/gpt-oss-120b` | Answers (few tokens per call) |
| `EMBED_PROVIDER` | `local` | `local` = CPU, `gemini` = API (100 texts/min free) |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 256-token input limit |

## Limitations

See [PLAN.md section 13](PLAN.md#13-known-weaknesses-honest-list). In short: quotes are always exact but can be only loosely relevant; short turns (< 6 words) cannot be cited; free-tier limits make extraction slow; each question must name one team.
