"""Plain-code transcript parser. No LLM: the format is fixed, so code is faster and exact."""
import hashlib
import re
from pathlib import Path

HEADER = re.compile(r"^(\d\d:\d\d:\d\d) Speaker (\w)$")


def parse_transcript(path: Path) -> dict:
    """File name 'ES2008b.txt' -> meeting id 'ES2008b', team 'ES2008', letter 'b'."""
    raw = path.read_text(encoding="utf-8")
    meeting_id = path.stem
    turns: list[dict] = []

    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        m = HEADER.match(lines[i].strip())
        if not m:
            i += 1
            continue
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip() and not HEADER.match(lines[i].strip()):
            body.append(lines[i].strip())
            i += 1
        text = " ".join(body)
        turns.append({
            "idx": len(turns),
            "start": m.group(1),
            "speaker": f"Speaker {m.group(2)}",
            "text": text,
            "n_words": len(text.split()),
        })

    return {
        "id": meeting_id,
        "team": meeting_id[:-1],
        "letter": meeting_id[-1],
        "source_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        "turns": turns,
    }
