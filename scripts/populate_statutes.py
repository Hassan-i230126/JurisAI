"""
Juris AI — Statute Population Script
Populates the SQLite statutes table from the JSON and PPC datasets.
Also rebuilds the FTS5 index for keyword search.

Usage:
    cd JurisAI
    python scripts/populate_statutes.py
"""

import sys
import json
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import (
    SQLITE_DB_PATH,
    DATASET_LAWS_JSON,
    DATASET_PPC_MARKDOWN,
    CRIMINAL_FILTER_KEYWORDS,
    ensure_directories,
)
from app.utils.logger import setup_logger
from loguru import logger


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the statutes table and FTS5 virtual table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS statutes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            act TEXT NOT NULL,
            section_number TEXT,
            title TEXT,
            text TEXT,
            punishment TEXT,
            cognizability TEXT,
            bailable TEXT,
            triable_by TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS statutes_fts USING fts5(
            title, text, content='statutes', content_rowid='id'
        )
    """)
    conn.commit()


def is_criminal(title: str, text: str) -> bool:
    """Check if content is criminal-law-relevant."""
    combined = (title + " " + text).lower()
    return any(kw.lower() in combined for kw in CRIMINAL_FILTER_KEYWORDS)


def populate_from_json(conn: sqlite3.Connection) -> int:
    """
    Populate statutes from the Pakistan Laws JSON dataset.
    
    Returns number of records inserted.
    """
    logger.info("Loading statutes from JSON: {}", DATASET_LAWS_JSON)

    try:
        with open(DATASET_LAWS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Failed to load JSON: {}", e)
        return 0

    # Normalize to list
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        first_val = next(iter(raw.values()))
        records = first_val if isinstance(first_val, list) else list(raw.values())
    else:
        logger.error("Unexpected JSON structure")
        return 0

    count = 0
    for record in records:
        if not isinstance(record, dict):
            continue

        # Extract fields defensively
        title = ""
        text = ""
        section = ""
        punishment = ""

        for key in ["title", "act_name", "law_name", "name", "heading"]:
            if key in record and record[key]:
                title = str(record[key])
                break

        for key in ["content", "text", "body", "law_text", "description", "full_text"]:
            if key in record and record[key]:
                text = str(record[key])
                break

        for key in ["section_number", "section", "section_no", "clause"]:
            if key in record and record[key]:
                section = str(record[key])
                break

        for key in ["punishment", "penalty", "sentence"]:
            if key in record and record[key]:
                punishment = str(record[key])
                break

        if not text or len(text) < 50:
            continue

        if not is_criminal(title, text):
            continue

        # Detect act abbreviation
        act = "PPC"
        title_lower = title.lower()
        if "criminal procedure" in title_lower or "crpc" in title_lower:
            act = "CrPC"
        elif "shahadat" in title_lower or "evidence" in title_lower or "qso" in title_lower:
            act = "QSO"
        elif "anti-terrorism" in title_lower or "ata" in title_lower:
            act = "ATA"

        conn.execute(
            """INSERT INTO statutes (act, section_number, title, text, punishment, 
               cognizability, bailable, triable_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (act, section, title, text[:5000], punishment, "", "", "")
        )
        count += 1

    conn.commit()
    logger.info("Inserted {} records from JSON dataset", count)
    return count


def populate_from_ppc_markdown(conn: sqlite3.Connection) -> int:
    """
    Populate statutes from the PPC Markdown file.
    
    Returns number of records inserted.
    """
    logger.info("Loading statutes from PPC Markdown: {}", DATASET_PPC_MARKDOWN)

    try:
        with open(DATASET_PPC_MARKDOWN, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error("PPC Markdown file not found")
        return 0

    # Split on section headers
    section_pattern = r'^#{1,3}\s+[Ss]ection\s+(\d+[\w-]*)\s*[.:\-—]*\s*(.*?)$'
    parts = re.split(section_pattern, content, flags=re.MULTILINE)

    count = 0

    # parts: [preamble, sec_num, sec_title, sec_text, sec_num, sec_title, sec_text, ...]
    for i in range(1, len(parts) - 2, 3):
        section_num = parts[i].strip()
        section_title = parts[i + 1].strip() if i + 1 < len(parts) else ""
        section_text = parts[i + 2].strip() if i + 2 < len(parts) else ""

        if not section_text or len(section_text) < 30:
            continue

        # Extract punishment from text
        punishment = ""
        punishment_match = re.search(
            r'(?:punish|sentence|imprison|fine|death|shall be)\w*[^.]*\.',
            section_text, re.IGNORECASE
        )
        if punishment_match:
            punishment = punishment_match.group(0)[:300]

        # Check if already exists (dedup)
        existing = conn.execute(
            "SELECT id FROM statutes WHERE act = 'PPC' AND section_number = ?",
            (section_num,)
        ).fetchone()

        if existing:
            # Update with PPC markdown content (more authoritative)
            conn.execute(
                "UPDATE statutes SET title = ?, text = ?, punishment = ? WHERE id = ?",
                (section_title, section_text[:5000], punishment, existing[0])
            )
        else:
            conn.execute(
                """INSERT INTO statutes (act, section_number, title, text, punishment,
                   cognizability, bailable, triable_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("PPC", section_num, section_title, section_text[:5000], punishment, "", "", "")
            )
            count += 1

    conn.commit()
    logger.info("Inserted {} new PPC sections from Markdown", count)
    return count


def rebuild_fts_index(conn: sqlite3.Connection) -> None:
    """Rebuild the FTS5 full-text search index."""
    logger.info("Rebuilding FTS5 index...")

    # Clear existing FTS data
    try:
        conn.execute("DELETE FROM statutes_fts")
    except Exception:
        pass

    # Rebuild from statutes table
    conn.execute("""
        INSERT INTO statutes_fts(rowid, title, text)
        SELECT id, title, text FROM statutes
    """)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM statutes_fts").fetchone()[0]
    logger.info("FTS5 index rebuilt with {} entries", count)


def main():
    """Run the statute population pipeline."""
    setup_logger()
    ensure_directories()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   Juris AI — Populate Statutes               ║")
    logger.info("╚══════════════════════════════════════════════╝")

    conn = sqlite3.connect(SQLITE_DB_PATH)

    try:
        # Create schema
        create_schema(conn)

        # Check existing count
        existing = conn.execute("SELECT COUNT(*) FROM statutes").fetchone()[0]
        logger.info("Existing statutes: {}", existing)

        if existing > 100:
            logger.info("Statutes already populated ({}). Delete juris.db to re-populate.", existing)
            rebuild_fts_index(conn)
            return

        # Populate from both sources
        json_count = populate_from_json(conn)
        ppc_count = populate_from_ppc_markdown(conn)

        # Rebuild FTS index
        rebuild_fts_index(conn)

        # Summary
        total = conn.execute("SELECT COUNT(*) FROM statutes").fetchone()[0]
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║   Statute Population Complete                 ║")
        logger.info("╠══════════════════════════════════════════════╣")
        logger.info("║ From JSON dataset:    {:>6} records           ║", json_count)
        logger.info("║ From PPC Markdown:    {:>6} records           ║", ppc_count)
        logger.info("║ Total in database:    {:>6} records           ║", total)
        logger.info("╚══════════════════════════════════════════════╝")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
