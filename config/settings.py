from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    google_api_key: str | None = None
    hf_token: str | None = None  # only needed to download a model the first time

    # Bulk extraction: Gemini free tier (15 RPM measured). Groq is the fallback (8,000 TPM).
    extract_model: str = "gemini-3.1-flash-lite"
    extract_fallback_model: str = "gemini-3.5-flash-lite,openai/gpt-oss-20b"  # tried in order
    extract_rpm: int = 12  # stay under the measured 15 RPM

    # Answers: few tokens per call, fits Groq's cap. Brief reads many notes, so it uses Gemini.
    answer_model: str = "openai/gpt-oss-120b"
    brief_model: str = "gemini-3.1-flash-lite"

    # Groq free tier: 8,000 tokens/min. Keep a safety margin.
    groq_tpm: int = 7000
    groq_rpm: int = 25

    # Embeddings: "local" = sentence-transformers on CPU (no quota),
    # "gemini" = API (100 texts/min on the free tier). HF hosted API returned 402.
    embed_provider: str = "local"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_words: int = 150  # MiniLM truncates input at 256 word-pieces

    # Extraction windows
    window_words: int = 1500
    min_extract_turn_words: int = 3

    # Retrieval
    chunks_per_question: int = 8
    chunks_per_meeting: int = 2
    max_agent_loops: int = 3

    transcripts_dir: str = "data/transcripts"
    db_path: str = "data/catalist.db"
    cache_dir: str = "data/cache"

    # extra="ignore": the shared .env may hold unrelated keys (LangSmith, etc.)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
