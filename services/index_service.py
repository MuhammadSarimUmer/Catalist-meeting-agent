"""Chunking and embeddings. All deterministic code, no LLM."""
import time

import numpy as np

from config.settings import settings

_model = None  # loaded once, reused


def build_chunks(turns: list[dict], max_words: int | None = None) -> list[dict]:
    """Group consecutive turns into chunks of <= max_words words.

    A single very long turn is split into several chunks (all pointing at the same turn)
    so nothing is cut off by the embedding model's input limit.
    """
    max_words = max_words or settings.chunk_words
    chunks: list[dict] = []
    cur: list[dict] = []
    cur_words = 0

    def flush() -> None:
        nonlocal cur, cur_words
        if cur:
            chunks.append({
                "first_turn": cur[0]["id"],
                "last_turn": cur[-1]["id"],
                "text": " ".join(f"{t['speaker']}: {t['text']}" for t in cur),
            })
        cur, cur_words = [], 0

    for t in turns:
        words = t["text"].split()
        if len(words) > max_words:
            flush()
            for i in range(0, len(words), max_words):
                chunks.append({
                    "first_turn": t["id"], "last_turn": t["id"],
                    "text": f"{t['speaker']}: " + " ".join(words[i:i + max_words]),
                })
            continue
        if cur_words + len(words) > max_words:
            flush()
        cur.append(t)
        cur_words += len(words)
    flush()
    return chunks


def _local_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:  # use the downloaded copy first: no network call, no timeout
            _model = SentenceTransformer(settings.embed_model, device="cpu", local_files_only=True)
        except OSError:  # first run on a new machine: download it (uses HF_TOKEN if set)
            _model = SentenceTransformer(settings.embed_model, device="cpu")
    return _model


def _embed_gemini(texts: list[str]) -> np.ndarray:
    """Gemini free tier = 100 texts/minute, so send 100 at a time and wait."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key, http_options=types.HttpOptions(api_version="v1beta"))
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        r = client.models.embed_content(
            model="gemini-embedding-001", contents=texts[i:i + 100],
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        out += [e.values for e in r.embeddings]
        if i + 100 < len(texts):
            time.sleep(62)
    return np.array(out, dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    """Returns L2-normalised float32 vectors, so cosine similarity is just a dot product."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    if settings.embed_provider == "gemini":
        vecs = _embed_gemini(texts)
    else:
        vecs = _local_model().encode(texts, batch_size=64, show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
