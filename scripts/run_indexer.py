"""
Juris AI — Indexer Entry Point
Preprocesses and indexes all 3 datasets into ChromaDB.

Usage:
    cd JurisAI
    python scripts/run_indexer.py
"""

import sys
import time
from pathlib import Path

# Add backend to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import CHROMA_COLLECTION_NAME, ensure_directories
from app.utils.logger import setup_logger
from app.rag.data_loaders import probe_json_schema
from app.rag.indexer import LegalIndexer

from loguru import logger


def main():
    """Run the full indexing pipeline."""
    # Initialize
    setup_logger()
    ensure_directories()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   Juris AI — Indexer                         ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # Step 1: Probe JSON schema for Dataset 1
    logger.info("Step 1: Probing Dataset 1 JSON schema...")
    probe_json_schema()
    print()

    # Step 2: Check existing collection
    indexer = LegalIndexer()
    existing_count = indexer.get_collection_size()
    logger.info("Existing collection '{}' has {} entries", CHROMA_COLLECTION_NAME, existing_count)

    if existing_count > 1000:
        logger.info(
            "Collection already has {} entries (> 1000). "
            "Skipping re-indexing. Delete data/chroma_db/ to force re-index.",
            existing_count
        )
        # Still run sanity check
        _run_sanity_check(indexer)
        return

    # Step 3: Run the full pipeline
    logger.info("Step 3: Running full indexing pipeline...")
    summary = indexer.run_full_pipeline()

    # Step 4: Sanity check
    _run_sanity_check(indexer)

    # Print final summary
    print("\n" + "=" * 60)
    print("  INDEXING COMPLETE")
    print("=" * 60)
    for key, val in summary.items():
        print(f"  {key}: {val}")
    print("=" * 60)


def _run_sanity_check(indexer: LegalIndexer):
    """Run a test query to verify the index works."""
    logger.info("Running sanity check query: 'murder under Pakistan Penal Code'")

    try:
        import ollama as ollama_client

        # Generate test embedding
        response = ollama_client.embeddings(
            model="bge-m3",
            prompt="murder under Pakistan Penal Code"
        )
        embedding = response["embedding"]

        # Query ChromaDB
        results = indexer.collection.query(
            query_embeddings=[embedding],
            n_results=3,
        )

        if results and results["documents"] and results["documents"][0]:
            logger.info("✓ Sanity check PASSED — {} results returned", len(results["documents"][0]))
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                source = meta.get("source", "unknown")
                section = meta.get("section", "")
                logger.info(
                    "  Result {}: source={} section={} text={}...",
                    i + 1, source, section, doc[:100]
                )
        else:
            logger.warning("✗ Sanity check returned no results")

    except Exception as e:
        logger.error("Sanity check failed: {}. Is Ollama running with bge-m3?", e)
        logger.info("Run: ollama serve && ollama pull bge-m3")


if __name__ == "__main__":
    main()
