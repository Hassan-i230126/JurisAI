"""
Juris AI — Configuration Module
All configuration via environment variables with sensible defaults.
Centralized configuration prevents magic strings scattered through the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ─── Base Paths ───────────────────────────────────────────────────────────────
# PROJECT_ROOT is the JurisAI/ directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Ollama Configuration ─────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "phi4-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")

# ─── ChromaDB Configuration ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(PROJECT_ROOT / "data" / "chroma_db")
)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "juris_legal_corpus")

# ─── SQLite Configuration ────────────────────────────────────────────────────
SQLITE_DB_PATH: str = os.getenv(
    "SQLITE_DB_PATH",
    str(PROJECT_ROOT / "data" / "juris.db")
)

# ─── Dataset Paths (relative to PROJECT_ROOT) ────────────────────────────────
DATASET_LAWS_JSON: str = os.getenv(
    "DATASET_LAWS_JSON",
    str(PROJECT_ROOT / "datasets" / "ayesha_jadoon_pdf_data.json")
)
DATASET_PPC_MARKDOWN: str = os.getenv(
    "DATASET_PPC_MARKDOWN",
    str(PROJECT_ROOT / "datasets" / "Pakistan_Penal_Court_markdown.md")
)
# Auto-detect the judgment folder name (handles typo variants)
_judgment_dir_default = str(PROJECT_ROOT / "datasets" / "Supreme_judgments")
if not Path(_judgment_dir_default).exists():
    _alt = PROJECT_ROOT / "datasets" / "Supreme_jugdments"
    if _alt.exists():
        _judgment_dir_default = str(_alt)

DATASET_JUDGMENTS_DIR: str = os.getenv(
    "DATASET_JUDGMENTS_DIR",
    _judgment_dir_default
)

# ─── RAG Configuration ───────────────────────────────────────────────────────
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_DISTANCE_THRESHOLD: float = float(os.getenv("RAG_DISTANCE_THRESHOLD", "1.4"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ─── LLM Generation Parameters ───────────────────────────────────────────────
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_NUM_THREADS: int = int(os.getenv("LLM_NUM_THREADS", "8"))

# ─── Conversation Configuration ──────────────────────────────────────────────
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "4"))

# ─── Cache Configuration ─────────────────────────────────────────────────────
EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "200"))

# ─── Logging Configuration ───────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR: str = os.getenv("LOG_DIR", str(PROJECT_ROOT / "data" / "logs"))

# ─── Server Configuration ────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# ─── Indexing Configuration ──────────────────────────────────────────────────
INDEXING_BATCH_SIZE: int = int(os.getenv("INDEXING_BATCH_SIZE", "16"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
JUDGMENT_BATCH_SIZE: int = int(os.getenv("JUDGMENT_BATCH_SIZE", "200"))
MAX_JUDGMENT_CHARS: int = int(os.getenv("MAX_JUDGMENT_CHARS", "6000"))

# ─── Criminal Law Filter Keywords ────────────────────────────────────────────
CRIMINAL_FILTER_KEYWORDS = [
    "Penal", "Criminal", "Evidence", "Qanun-e-Shahadat", "Anti-Terrorism",
    "Narcotic", "Offences", "Hudood", "Zina", "Qazf", "Prohibition",
    "Juvenile", "Bail", "Extradition", "Explosive", "Firearms",
    "Kidnapping", "Corruption", "Prevention of"
]

# ─── Judgment Content Filter Keywords ─────────────────────────────────────────
JUDGMENT_FILTER_KEYWORDS = [
    "criminal", "murder", "robbery", "theft", "bail", "acquittal",
    "conviction", "302", "304", "392", "penal code", "ppc", "crpc",
    "anti-terrorism", "narcotics", "accused", "appellant"
]

# ─── Ensure data directories exist ────────────────────────────────────────────
def ensure_directories() -> None:
    """Create required data directories if they don't exist."""
    dirs = [
        Path(CHROMA_PERSIST_DIR),
        Path(SQLITE_DB_PATH).parent,
        Path(LOG_DIR),
        PROJECT_ROOT / "data" / "processed",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
