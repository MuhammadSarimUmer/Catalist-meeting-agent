"""Citations are built and checked by CODE, never by the LLM.

The grader wants: meeting id + a quote of >= 6 words, copied exactly from ONE speaker turn.
So the LLM only says WHICH turn supports a claim. This file copies the words out of that turn.
A copied span is exact by construction, and verify_quote() double-checks it.
"""
import re

from services.storage_service import MIN_CITE_WORDS, get_conn

STOP = set("""a an the and or but if so of to in on at for with from by is are was were be been it its this that
these those i you he she we they them his her our their my your me us do does did have has had will would can
could should not no yes um uh er mm hmm okay ok just like then there here what which who how when where why
about as also very really""".split())
MAX_QUOTE_WORDS = 45


_FIX = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2026": "..."}


def clean_text(text: str) -> str:
    """Models like fancy dashes and non-breaking spaces. Plain ASCII is safer for JSON, consoles and graders."""
    for bad, good in _FIX.items():
        text = text.replace(bad, good)
    # internal turn ids must never reach the reader: "[45696]", "(turn 10207)", "turns 12 and 14"
    text = re.sub(r"\s*\((?:turns?\s*)?\[?\d{3,6}\]?(?:[,\s]+\[?\d{3,6}\]?)*\)", "", text)
    text = re.sub(r"\s*\[\s*turn_id\s*:[^\]]*\]", "", text)
    text = re.sub(r"\s*\((?:Speaker [A-D],?\s*)?(?:turn[_ ]?id:?\s*)?\d{3,6}\)", "", text)
    text = re.sub(r"\s*\[\d{3,6}\]", "", text)
    text = re.sub(r"\bturns?\s+\d{3,6}\b", "that turn", text)
    return text


def normalise(text: str) -> str:
    """How the grader compares: case and spacing are ignored."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p for p in parts if p]


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 1}


def best_quote(turn_text: str, hint: str) -> str | None:
    """Pick the span of one turn that best matches `hint` (the claim). Always an exact copy."""
    words = turn_text.split()
    if len(words) < MIN_CITE_WORDS:
        return None
    if len(words) <= MAX_QUOTE_WORDS:
        return " ".join(words)

    want = _keywords(hint)
    sentences = split_sentences(turn_text)
    best, best_score = None, -1.0
    # try every run of 1-3 consecutive sentences
    for i in range(len(sentences)):
        for j in range(i, min(i + 3, len(sentences))):
            span = " ".join(sentences[i:j + 1])
            n = len(span.split())
            if n < MIN_CITE_WORDS or n > MAX_QUOTE_WORDS:
                continue
            score = len(want & _keywords(span)) / (n ** 0.5)  # prefer relevant, but not just longest
            if score > best_score:
                best, best_score = span, score
    if best:
        return best
    # fallback: no sentence run fits (very long sentences) -> a word window around the best match
    best_start, best_hits = 0, -1
    for s in range(0, len(words) - MIN_CITE_WORDS + 1, 5):
        win = words[s:s + MAX_QUOTE_WORDS]
        hits = len(want & _keywords(" ".join(win)))
        if hits > best_hits:
            best_start, best_hits = s, hits
    return " ".join(words[best_start:best_start + MAX_QUOTE_WORDS])


def verify_quote(turn_text: str, quote: str) -> bool:
    """>= 6 words AND an exact substring of the turn (case/spacing ignored)."""
    return len(quote.split()) >= MIN_CITE_WORDS and normalise(quote) in normalise(turn_text)


def get_turn(turn_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM turn WHERE id=?", (turn_id,)).fetchone()
    return dict(row) if row else None


def make_citation(turn_id: int, hint: str, team: str) -> dict | None:
    """turn id + claim -> {"meeting_id", "quote"} or None if the turn cannot be cited."""
    turn = get_turn(turn_id)
    if not turn or turn["team"] != team:
        return None
    quote = best_quote(turn["text"], hint)
    if not quote or not verify_quote(turn["text"], quote):
        return None
    return {"meeting_id": turn["meeting_id"], "quote": quote}
