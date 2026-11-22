"""Measure Groq rate limits against realistic extraction windows.

Run from the Catalist folder:
    python -m scripts.probe_limits
    python -m scripts.probe_limits --models openai/gpt-oss-120b llama-3.1-8b-instant --burst 8

What it answers:
  1. How many tokens does one ~window of transcript really cost (input + output + reasoning)?
  2. What are the account's TPM/RPM/TPD limits (read from response headers)?
  3. Do we hit 429s when firing windows back to back, and how long is retry-after?
  4. Projection: how long would the whole corpus / 1,000 meetings take on this model?
"""
import argparse
import json
import time
from pathlib import Path

import groq

from config.settings import settings

CORPUS_TOKENS = 880_000      # from the brief (~676k words)
CORPUS_MEETINGS = 138

SYSTEM = (
    "You extract structured memory from a noisy meeting transcript. "
    "Return JSON: {\"items\": [{\"kind\": \"decision|proposal|objection|action|number\", "
    "\"speaker\": \"Speaker X\", \"statement\": \"short paraphrase\"}]}. "
    "Only include items clearly present in the text."
)

HEADERS = [
    "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
    "retry-after",
]


def load_windows(words_per_window: int, n: int) -> list[str]:
    """Slice real meetings into full-size word windows (plain code, no LLM)."""
    windows: list[str] = []
    for f in sorted(Path(settings.transcripts_dir).glob("*d.txt")):
        words = f.read_text(encoding="utf-8").split("\n\n\n", 1)[-1].split()
        for i in range(len(words) // words_per_window):
            windows.append(" ".join(words[i * words_per_window:(i + 1) * words_per_window]))
            if len(windows) == n:
                return windows
    return windows


args_sleep = 0.0
args_max_out = 1500  # set from CLI in main(); capping it lowers the tokens Groq reserves per request


def call(client: groq.Groq, model: str, window: str) -> dict:
    t0 = time.time()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": window}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=args_max_out,
            **({"reasoning_effort": "low"} if "gpt-oss" in model else {}),
        )
        resp = raw.parse()
        u = resp.usage
        return {
            "ok": True,
            "seconds": round(time.time() - t0, 2),
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
            "headers": {h: raw.headers.get(h) for h in HEADERS if raw.headers.get(h)},
        }
    except groq.RateLimitError as e:
        return {
            "ok": False,
            "error": "429 rate limit",
            "seconds": round(time.time() - t0, 2),
            "message": str(e)[:300],
            "headers": {h: e.response.headers.get(h) for h in HEADERS if e.response.headers.get(h)},
        }
    except groq.APIError as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)[:300],
                "seconds": round(time.time() - t0, 2), "headers": {}}


def probe_model(client: groq.Groq, model: str, windows: list[str]) -> dict:
    print(f"\n=== {model} ===")
    results = []
    start = time.time()
    for i, w in enumerate(windows, 1):
        r = call(client, model, w)
        results.append(r)
        if args_sleep:
            time.sleep(args_sleep)
        if r["ok"]:
            print(f"  #{i} ok   {r['seconds']:>5}s  in={r['prompt_tokens']} out={r['completion_tokens']}")
        else:
            print(f"  #{i} FAIL {r['error']}  retry-after={r['headers'].get('retry-after')}  {r['message'][:120]}")
        if r["headers"] and i == 1:
            print("     limits:", r["headers"])
    wall = time.time() - start

    ok = [r for r in results if r["ok"]]
    summary = {"model": model, "calls": len(results), "ok": len(ok), "rate_limited": sum(
        1 for r in results if r.get("error") == "429 rate limit"), "wall_seconds": round(wall, 1)}
    if ok:
        in_avg = sum(r["prompt_tokens"] for r in ok) / len(ok)
        out_avg = sum(r["completion_tokens"] for r in ok) / len(ok)
        tok_per_call = in_avg + out_avg
        calls_needed = CORPUS_TOKENS / in_avg
        observed_tpm = sum(r["total_tokens"] for r in ok) / (wall / 60)
        summary.update({
            "avg_in": round(in_avg), "avg_out": round(out_avg),
            "tokens_per_call": round(tok_per_call),
            "calls_for_full_corpus": round(calls_needed),
            "total_tokens_full_corpus": round(calls_needed * tok_per_call),
            "observed_tokens_per_min": round(observed_tpm),
            "est_minutes_full_corpus_at_observed_rate": round(calls_needed * tok_per_call / max(observed_tpm, 1), 1),
            "est_minutes_1000_meetings": round(
                calls_needed * tok_per_call / max(observed_tpm, 1) * 1000 / CORPUS_MEETINGS, 1),
        })
    print("  summary:", json.dumps(summary, indent=2))
    return {"summary": summary, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
    ap.add_argument("--max-out", type=int, default=1500)
    ap.add_argument("--sleep", type=float, default=0, help="seconds between calls (0 = burst)")
    ap.add_argument("--words", type=int, default=2000, help="words per window (~1.35 tokens/word)")
    ap.add_argument("--burst", type=int, default=6, help="back-to-back calls per model")
    args = ap.parse_args()

    global args_max_out
    args_max_out = args.max_out
    global args_sleep
    args_sleep = args.sleep
    # max_retries=0 so 429s show up instead of being silently retried by the SDK
    client = groq.Groq(api_key=settings.groq_api_key, max_retries=0)
    windows = load_windows(args.words, args.burst)

    out = {m: probe_model(client, m, windows) for m in args.models}
    Path("data/probe_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved data/probe_results.json")


if __name__ == "__main__":
    main()
