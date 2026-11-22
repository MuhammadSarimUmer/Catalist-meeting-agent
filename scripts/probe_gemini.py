"""Probe Gemini for the same extraction job, plus one whole-meeting call.

    python -m scripts.probe_gemini --burst 8 --words 1500
"""
import argparse
import time
from pathlib import Path

from google import genai
from google.genai import types

from config.settings import settings
from scripts.probe_limits import SYSTEM, load_windows


def call(client: genai.Client, model: str, text: str) -> dict:
    t0 = time.time()
    try:
        r = client.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        u = r.usage_metadata
        return {"ok": True, "seconds": round(time.time() - t0, 2),
                "in": u.prompt_token_count, "out": u.candidates_token_count,
                "json_ok": r.text.strip().startswith("{")}
    except Exception as e:  # noqa: BLE001 - probe script, we want to see everything
        return {"ok": False, "seconds": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {str(e)[:250]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", type=int, default=8)
    ap.add_argument("--words", type=int, default=1500)
    args = ap.parse_args()

    client = genai.Client(api_key=settings.google_api_key, http_options=types.HttpOptions(api_version="v1beta"))
    model = settings.gemini_model
    print(f"=== {model}: {args.burst} back-to-back windows of {args.words} words ===")
    for i, w in enumerate(load_windows(args.words, args.burst), 1):
        print(f"  #{i}", call(client, model, w))

    big = max(Path(settings.transcripts_dir).glob("*d.txt"), key=lambda p: p.stat().st_size)
    print(f"=== whole meeting {big.stem} in ONE call ===")
    print(" ", call(client, model, big.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
