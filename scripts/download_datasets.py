"""
Juris AI — Dataset Download Instructions & Automation
Instructions for fetching all 3 datasets.

Since the datasets are already present in the project root:
- ayesha_jadoon_pdf_data.json
- Pakistan_Penal_Court_markdown.md
- Supreme_judgments/

This script verifies their presence and reports status.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directories to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


from app.config import (
    DATASET_LAWS_JSON,
    DATASET_PPC_MARKDOWN,
    DATASET_JUDGMENTS_DIR,
)


def check_datasets():
    """Verify that all required datasets are present."""
    print("=" * 60)
    print("  Juris AI — Dataset Verification")
    print("=" * 60)
    print()

    all_ok = True

    # Dataset 1: Pakistan Laws JSON
    print("Dataset 1: Pakistan Laws JSON")
    print(f"  Path: {DATASET_LAWS_JSON}")
    if os.path.exists(DATASET_LAWS_JSON):
        size_mb = os.path.getsize(DATASET_LAWS_JSON) / (1024 * 1024)
        print(f"  Status: ✓ Found ({size_mb:.1f} MB)")

        # Quick schema check
        try:
            with open(DATASET_LAWS_JSON, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                print(f"  Structure: JSON array with {len(raw)} elements")
            elif isinstance(raw, dict):
                print(f"  Structure: JSON object with {len(raw)} keys")
        except Exception as e:
            print(f"  Warning: Could not parse JSON — {e}")
    else:
        print("  Status: ✗ NOT FOUND")
        print("  Action: Download from HuggingFace or place the file manually.")
        all_ok = False
    print()

    # Dataset 2: PPC Markdown
    print("Dataset 2: Pakistan Penal Code Markdown")
    print(f"  Path: {DATASET_PPC_MARKDOWN}")
    if os.path.exists(DATASET_PPC_MARKDOWN):
        size_kb = os.path.getsize(DATASET_PPC_MARKDOWN) / 1024
        print(f"  Status: ✓ Found ({size_kb:.0f} KB)")
    else:
        print("  Status: ✗ NOT FOUND")
        print("  Action: Download from GitHub repo or place the file manually.")
        all_ok = False
    print()

    # Dataset 3: Supreme Court Judgments
    print("Dataset 3: Supreme Court Judgments")
    print(f"  Path: {DATASET_JUDGMENTS_DIR}")
    if os.path.isdir(DATASET_JUDGMENTS_DIR):
        import glob
        txt_files = glob.glob(os.path.join(DATASET_JUDGMENTS_DIR, "*.txt"))
        print(f"  Status: ✓ Found ({len(txt_files)} .txt files)")
    else:
        print("  Status: ✗ NOT FOUND")
        print("  Action: Place Supreme Court judgment .txt files in the directory.")
        all_ok = False
    print()

    print("=" * 60)
    if all_ok:
        print("  All datasets verified. Ready to index.")
        print("  Run: python scripts/run_indexer.py")
    else:
        print("  Some datasets are missing. Please download them.")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    check_datasets()
