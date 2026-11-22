"""One place for every LLM call: JSON mode, rate limiting, disk cache, retries, fallback.

Why one place: free tiers have hard limits (measured: Gemini 15 requests/min, Groq 8,000 tokens/min).
Keeping the plumbing here means the rest of the code just calls `call_json(...)`.
"""
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path

import groq
from google import genai
from google.genai import types

from config.settings import settings


class LLMError(Exception):
    pass


class RateLimiter:
    """Sliding 60-second window. Waits (instead of failing with 429) when a budget is used up."""

    def __init__(self, rpm: int, tpm: int | None = None):
        self.rpm, self.tpm = rpm, tpm
        self.events: deque[tuple[float, int]] = deque()  # (time, tokens)

    def wait(self, tokens: int) -> None:
        while True:
            now = time.time()
            while self.events and now - self.events[0][0] > 60:
                self.events.popleft()
            used = sum(t for _, t in self.events)
            ok_rpm = len(self.events) < self.rpm
            ok_tpm = self.tpm is None or used + tokens <= self.tpm or not self.events
            if ok_rpm and ok_tpm:
                self.events.append((now, tokens))
                return
            time.sleep(max(0.5, 60 - (now - self.events[0][0]) + 0.1))


_limiters = {
    "gemini": RateLimiter(settings.extract_rpm),
    "groq": RateLimiter(settings.groq_rpm, settings.groq_tpm),
}
_clients: dict = {}
_exhausted: set[str] = set()  # models whose DAILY quota is gone: do not try them again this run


def _provider(model: str) -> str:
    return "gemini" if model.startswith("gemini") else "groq"


def _estimate_tokens(system: str, user: str, max_out: int) -> int:
    return int(len((system + user).split()) * 1.8) + max_out


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def _call_gemini(model: str, system: str, user: str, max_out: int) -> str:
    if "gemini" not in _clients:
        _clients["gemini"] = genai.Client(
            api_key=settings.google_api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=90000,
                                           retry_options=types.HttpRetryOptions(attempts=1)),
        )
    r = _clients["gemini"].models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            temperature=0, max_output_tokens=max_out),
    )
    return r.text or ""


def _call_groq(model: str, system: str, user: str, max_out: int) -> str:
    if "groq" not in _clients:
        _clients["groq"] = groq.Groq(api_key=settings.groq_api_key, max_retries=0, timeout=90)
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    # max_completion_tokens matters: Groq counts the reserved output against the per-minute budget
    r = _clients["groq"].chat.completions.create(
        model=model, temperature=0, max_completion_tokens=max_out,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
    return r.choices[0].message.content or ""


def _cache_path(model: str, system: str, user: str) -> Path:
    key = hashlib.sha256(f"{model}|{system}|{user}".encode()).hexdigest()
    return Path(settings.cache_dir) / f"{key}.json"


def _one_model(model: str, system: str, user: str, max_out: int, attempts: int = 4) -> dict:
    if model in _exhausted:
        raise LLMError(f"{model}: daily quota used up")
    provider = _provider(model)
    last: Exception | None = None
    for attempt in range(attempts):
        _limiters[provider].wait(_estimate_tokens(system, user, max_out))
        try:
            text = (_call_gemini if provider == "gemini" else _call_groq)(model, system, user, max_out)
            return _parse_json(text)
        except json.JSONDecodeError as e:  # model returned bad JSON: try once more
            last = e
        except Exception as e:  # noqa: BLE001 - 429 / 503 / timeouts: wait and retry
            last = e
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():  # a daily cap will not clear in a minute
                _exhausted.add(model)
                break
            m = re.search(r"retry[^0-9]{0,15}(\d+(?:\.\d+)?)", msg, re.I)
            transient = any(x in msg for x in ("429", "503", "500", "timed out", "Timeout", "overloaded"))
            if not transient:
                break
            time.sleep(min(60.0, float(m.group(1)) + 1 if m else 5 * (attempt + 1)))
    raise LLMError(f"{model}: {last}")


def call_json(model: str, system: str, user: str, max_out: int = 1500,
              fallback: str | None = None, refresh: bool = False) -> dict:
    """Returns the model's JSON reply as a dict. Cached on disk, so re-runs are free.

    `fallback` = one model or a comma-separated chain, tried in order if the first one fails.
    `refresh=True` ignores the cached reply and asks again (the new reply replaces it).
    """
    path = _cache_path(model, system, user)
    if not refresh and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    chain = [model] + [m.strip() for m in (fallback or "").split(",") if m.strip()]
    last: LLMError | None = None
    for m in chain:
        try:
            result = _one_model(m, system, user, max_out)
            break
        except LLMError as e:
            last = e
    else:
        raise last  # every model in the chain failed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result
