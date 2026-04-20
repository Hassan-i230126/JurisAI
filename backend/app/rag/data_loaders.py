"""
Juris AI — Data Loaders
Loaders for each dataset format: JSON laws, PPC markdown, SC judgments.
All paths resolved via config.py — datasets are in the project root.
"""

import os
import re
import json
import glob
from typing import List, Dict, Optional, Tuple, Generator

from loguru import logger

from app.config import (
    DATASET_LAWS_JSON,
    DATASET_PPC_MARKDOWN,
    DATASET_JUDGMENTS_DIR,
    CRIMINAL_FILTER_KEYWORDS,
    JUDGMENT_FILTER_KEYWORDS,
    MAX_JUDGMENT_CHARS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema Probe Utility
# ═══════════════════════════════════════════════════════════════════════════════

def probe_json_schema() -> None:
    """
    Inspect the top-level structure of the laws JSON dataset.
    
    Prints the schema to the terminal for runtime confirmation
    before the indexing pipeline begins. Called at the top of
    run_indexer.py before the main pipeline.
    """
    logger.info("Probing JSON schema of: {}", DATASET_LAWS_JSON)

    try:
        with open(DATASET_LAWS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.error("Dataset file not found: {}", DATASET_LAWS_JSON)
        return
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in {}: {}", DATASET_LAWS_JSON, e)
        return

    if isinstance(raw, list):
        logger.info("Top-level structure: LIST with {} elements", len(raw))
        if raw:
            first = raw[0]
            if isinstance(first, dict):
                logger.info("First element keys: {}", list(first.keys()))
                for key in list(first.keys())[:5]:
                    val = first[key]
                    preview = str(val)[:200] if val else "None"
                    logger.info("  {} ({}): {}", key, type(val).__name__, preview)
            else:
                logger.info("First element type: {} — value: {}", type(first).__name__, str(first)[:200])
    elif isinstance(raw, dict):
        logger.info("Top-level structure: DICT with {} keys", len(raw))
        first_key = next(iter(raw.keys()))
        first_val = raw[first_key]
        logger.info("First key: '{}' — value type: {}", first_key, type(first_val).__name__)
        if isinstance(first_val, dict):
            logger.info("First value keys: {}", list(first_val.keys()))
            for key in list(first_val.keys())[:5]:
                v = first_val[key]
                logger.info("  {} ({}): {}", key, type(v).__name__, str(v)[:200] if v else "None")
        elif isinstance(first_val, list):
            logger.info("First value is a list with {} elements", len(first_val))
            if first_val and isinstance(first_val[0], dict):
                logger.info("First list element keys: {}", list(first_val[0].keys()))
    else:
        logger.warning("Unexpected top-level type: {}", type(raw).__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Field Extraction Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(record: dict) -> str:
    """
    Extract the main text content from a record.
    Tries multiple candidate field names since the schema may vary.
    """
    for key in ["content", "text", "body", "law_text", "description", "full_text"]:
        if key in record and record[key]:
            return str(record[key])
    return ""


def extract_title(record: dict) -> str:
    """
    Extract the title/act name from a record.
    Tries multiple candidate field names.
    """
    for key in ["title", "act_name", "law_name", "name", "heading"]:
        if key in record and record[key]:
            return str(record[key])
    return "Unknown Act"


def extract_section(record: dict) -> str:
    """
    Extract the section number from a record.
    Tries multiple candidate field names.
    """
    for key in ["section_number", "section", "section_no", "clause"]:
        if key in record and record[key]:
            return str(record[key])
    return ""


def is_criminal_relevant_text(title: str, text: str) -> bool:
    """
    Check if a record's content is relevant to criminal law.
    
    Uses case-insensitive keyword matching against the filter list.
    """
    combined = (title + " " + text).lower()
    return any(kw.lower() in combined for kw in CRIMINAL_FILTER_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 1 — Pakistan Laws JSON
# ═══════════════════════════════════════════════════════════════════════════════

def load_laws_json() -> List[Dict]:
    """
    Load and filter the Pakistan Laws JSON dataset.
    
    Handles both JSON arrays and dict-of-records structures.
    Filters to criminal law content and skips short/empty records.
    
    Returns:
        List of dicts with keys: text, metadata
    """
    logger.info("Loading Dataset 1: Pakistan Laws JSON from {}", DATASET_LAWS_JSON)

    try:
        with open(DATASET_LAWS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.error("Dataset 1 file not found: {}", DATASET_LAWS_JSON)
        return []
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in Dataset 1: {}", e)
        return []

    # Normalize to a flat list of dicts regardless of top-level shape
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        first_val = next(iter(raw.values()))
        if isinstance(first_val, list):
            records = first_val
        else:
            records = list(raw.values())
    else:
        raise ValueError(f"Unrecognised JSON structure in {DATASET_LAWS_JSON}")

    logger.info("Dataset 1: {} total records loaded", len(records))

    # Filter and extract
    result = []
    skipped_empty = 0
    skipped_short = 0
    skipped_irrelevant = 0

    for record in records:
        if not isinstance(record, dict):
            continue

        text = extract_text(record)
        title = extract_title(record)
        section = extract_section(record)

        # Skip empty text
        if not text:
            skipped_empty += 1
            continue

        # Skip very short text (< 80 characters)
        if len(text) < 80:
            skipped_short += 1
            continue

        # Filter to criminal law content
        if not is_criminal_relevant_text(title, text):
            skipped_irrelevant += 1
            continue

        metadata = {
            "source": "ayesha_jadoon_dataset",
            "act": title,
            "section": section,
            "doc_type": "statute",
        }

        result.append({"text": text, "metadata": metadata})

    logger.info(
        "Dataset 1 filtered: {} relevant records | "
        "Skipped: {} empty, {} short, {} irrelevant",
        len(result), skipped_empty, skipped_short, skipped_irrelevant
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 2 — Pakistan Penal Code Markdown
# ═══════════════════════════════════════════════════════════════════════════════

def load_ppc_markdown() -> List[Dict]:
    """
    Load and split the PPC Markdown dataset into section-level chunks.
    
    Primary split: on section headers using regex.
    Fallback: double newlines if regex yields < 10 splits.
    
    Returns:
        List of dicts with keys: text, metadata
    """
    logger.info("Loading Dataset 2: PPC Markdown from {}", DATASET_PPC_MARKDOWN)

    try:
        with open(DATASET_PPC_MARKDOWN, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error("Dataset 2 file not found: {}", DATASET_PPC_MARKDOWN)
        return []

    logger.info("Dataset 2: {} characters loaded", len(content))

    # Primary split: section headers
    section_pattern = r'^#{1,3}\s+[Ss]ection\s+(\d+[\w-]*)'
    splits = re.split(section_pattern, content, flags=re.MULTILINE)

    result = []

    if len(splits) > 10:
        # splits alternates: [preamble, sec_num, text, sec_num, text, ...]
        # First element is everything before the first section header
        preamble = splits[0].strip()
        if preamble and len(preamble) >= 80:
            result.append({
                "text": preamble[:3000],
                "metadata": {
                    "source": "ppc_markdown",
                    "act": "Pakistan Penal Code",
                    "section": "",
                    "doc_type": "statute",
                }
            })

        # Process section pairs
        for i in range(1, len(splits) - 1, 2):
            section_num = splits[i].strip()
            section_text = splits[i + 1].strip() if i + 1 < len(splits) else ""

            if not section_text or len(section_text) < 30:
                continue

            # Prepend the section number to the text for context
            full_text = f"Section {section_num}\n{section_text}"

            result.append({
                "text": full_text,
                "metadata": {
                    "source": "ppc_markdown",
                    "act": "Pakistan Penal Code",
                    "section": section_num,
                    "doc_type": "statute",
                }
            })
    else:
        # Fallback: split on double newlines
        logger.warning("Section regex yielded < 10 splits, falling back to paragraph split")
        paragraphs = content.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if len(para) < 150:
                continue

            # Try to extract section number from the paragraph
            sec_match = re.search(r'[Ss]ection\s+(\d+[\w-]*)', para)
            section_num = sec_match.group(1) if sec_match else ""

            result.append({
                "text": para,
                "metadata": {
                    "source": "ppc_markdown",
                    "act": "Pakistan Penal Code",
                    "section": section_num,
                    "doc_type": "statute",
                }
            })

    logger.info("Dataset 2: {} sections/chunks extracted", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 3 — Supreme Court Judgments
# ═══════════════════════════════════════════════════════════════════════════════

def is_criminal_relevant(filepath: str) -> bool:
    """
    Check if a judgment file is relevant to criminal law.
    
    Reads only the first 500 bytes to avoid loading full files into memory.
    
    Args:
        filepath: Path to the judgment text file.
        
    Returns:
        True if the file contains criminal law keywords.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(500).lower()
        return any(kw in sample for kw in JUDGMENT_FILTER_KEYWORDS)
    except Exception:
        return False


def extract_judgment_metadata(text: str, filename: str) -> Dict[str, str]:
    """
    Extract metadata from a judgment's text and filename.
    
    Args:
        text: The judgment text (first 500 chars used for extraction).
        filename: The original filename.
        
    Returns:
        Dict with source, doc_type, year, case_ref, filename.
    """
    year_match = re.search(r'\b(19|20)\d{2}\b', text[:500])
    year = year_match.group(0) if year_match else "unknown"

    # Extract case number from filename: "C.A_supreme (42).txt" → "CA-42"
    num_match = re.search(r'\((\d+)\)', filename)
    case_ref = f"CA-{num_match.group(1)}" if num_match else filename

    return {
        "source": "sc_judgments",
        "doc_type": "judgment",
        "year": year,
        "case_ref": case_ref,
        "filename": os.path.basename(filename),
    }


def load_judgments_batch(batch_files: List[str]) -> List[Dict]:
    """
    Load a batch of judgment files.
    
    Each file is filtered for criminal relevance, then the first
    MAX_JUDGMENT_CHARS characters are read.
    
    Args:
        batch_files: List of file paths to process.
        
    Returns:
        List of dicts with keys: text, metadata
    """
    result = []

    for filepath in batch_files:
        try:
            # Content-based filter (reads first 500 bytes)
            if not is_criminal_relevant(filepath):
                continue

            # Full read (truncated)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(MAX_JUDGMENT_CHARS)

            if len(text) < 80:
                continue

            metadata = extract_judgment_metadata(text, filepath)
            result.append({"text": text, "metadata": metadata})

        except Exception as e:
            logger.warning("Failed to load judgment {}: {}", filepath, str(e))
            continue

    return result


def get_judgment_files() -> List[str]:
    """
    Get all judgment text files from the judgments directory.
    
    Returns:
        Sorted list of file paths.
    """
    pattern = os.path.join(DATASET_JUDGMENTS_DIR, "*.txt")
    all_files = glob.glob(pattern)
    logger.info("Found {} judgment files in {}", len(all_files), DATASET_JUDGMENTS_DIR)
    return sorted(all_files)
