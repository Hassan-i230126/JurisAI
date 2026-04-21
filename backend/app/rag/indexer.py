"""
Juris AI — Indexer
Offline pipeline: load datasets → chunk → embed → store in ChromaDB.
"""

import asyncio
import time
from typing import List, Set

import chromadb
import ollama as ollama_client
from loguru import logger

from app.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    JUDGMENT_BATCH_SIZE,
)
from app.models.schemas import DocumentChunk
from app.rag.data_loaders import (
    load_laws_json,
    load_ppc_markdown,
    load_judgments_batch,
    get_judgment_files,
)
from app.rag.chunker import chunk_documents, hash_text


class LegalIndexer:
    """
    Manages the full indexing pipeline for Juris AI.
    
    Pipeline order: Dataset 1 (Laws JSON) → Dataset 2 (PPC Markdown)
    → Dataset 3 (SC Judgments). Includes deduplication and progress logging.
    """

    def __init__(self):
        """Initialize the indexer with a persistent ChromaDB client."""
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_hashes: Set[str] = set()
        self._ppc_sections: Set[str] = set()  # Track PPC sections for dedup
        self._total_indexed = 0
        self._total_skipped = 0
        self._total_errors = 0

    def get_collection_size(self) -> int:
        """Return the number of documents currently in the collection."""
        return self.collection.count()

    def _embed_texts(self, texts: List[str], label: str = "") -> List[List[float]]:
        """
        Generate embeddings using Ollama's batch embed API.
        
        Uses ollama.embed() with list input — sends EMBEDDING_BATCH_SIZE texts
        per HTTP request instead of one-by-one, dramatically reducing overhead.
        Logs progress with ETA every 1000 chunks.
        
        Args:
            texts: List of text strings to embed.
            label: Label for progress logging (e.g. 'Dataset 1').
            
        Returns:
            List of embedding vectors (one per input text).
        """
        all_embeddings = []
        total = len(texts)
        start_time = time.time()
        last_log_milestone = -1

        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = []

            try:
                # We iterate individually to prevent memory thrashing on CPU.
                # Sending massive batches of 1000-char chunks causes 5+ minute lag spikes.
                for text in batch:
                    resp = ollama_client.embeddings(
                        model=EMBEDDING_MODEL,
                        prompt=text,
                    )
                    batch_embeddings.append(resp["embedding"])
                
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                  logger.error("Embedding failed during batch loop. Stopping to prevent zero-vector corruption: {}", e)
                  raise RuntimeError(f"Ollama Embedding API failed: {e}")

            # ── Progress logging every 1000 chunks ──
            done = min(i + len(batch), total)
            current_milestone = done // 1000
            if current_milestone > last_log_milestone or done >= total:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta_sec = remaining / rate if rate > 0 else 0
                logger.info(
                    "{} | Embedded {}/{} ({:.1f}%) | "
                    "Elapsed: {:.0f}s | Rate: {:.1f} chunks/s | ETA: {:.0f}s ({:.1f} min)",
                    label, done, total,
                    (done / total) * 100,
                    elapsed, rate, eta_sec, eta_sec / 60,
                )
                last_log_milestone = current_milestone

        return all_embeddings

    def _load_existing_hashes(self) -> None:
        """Load existing document hashes from the collection for deduplication."""
        try:
            count = self.collection.count()
            if count == 0:
                return

            # Get existing IDs (which are the hashes)
            results = self.collection.get(limit=count, include=[])
            if results and results["ids"]:
                self._indexed_hashes = set(results["ids"])
                logger.info("Loaded {} existing document hashes for deduplication", len(self._indexed_hashes))
        except Exception as e:
            logger.warning("Could not load existing hashes: {}", e)

    def _index_chunks(self, chunks: List[DocumentChunk], label: str) -> int:
        """
        Index a list of document chunks into ChromaDB.
        
        Skips duplicates based on SHA-256 hash.
        
        Args:
            chunks: List of DocumentChunk objects to index.
            label: Label for logging (e.g. 'Dataset 1').
            
        Returns:
            Number of new chunks indexed.
        """
        new_chunks = []
        for chunk in chunks:
            if chunk.chunk_hash not in self._indexed_hashes:
                new_chunks.append(chunk)
                self._indexed_hashes.add(chunk.chunk_hash)
            else:
                self._total_skipped += 1

        if not new_chunks:
            logger.info("{}: All chunks already indexed (dedup)", label)
            return 0

        logger.info("{}: Embedding {} new chunks...", label, len(new_chunks))

        # Embed in batches
        texts = [c.text for c in new_chunks]
        embeddings = self._embed_texts(texts, label=label)

        # Prepare ChromaDB upsert data
        ids = [c.chunk_hash for c in new_chunks]
        metadatas = [c.metadata for c in new_chunks]

        # Upsert in batches of 100 to avoid memory issues
        batch_size = 100
        indexed_count = 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            try:
                self.collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )
                indexed_count += len(batch_ids)
            except MemoryError:
                logger.error("{}: MemoryError at batch {} — skipping", label, i // batch_size)
                self._total_errors += len(batch_ids)
                continue
            except Exception as e:
                logger.error("{}: Error indexing batch {}: {}", label, i // batch_size, e)
                self._total_errors += len(batch_ids)
                continue

        self._total_indexed += indexed_count
        logger.info("{}: Indexed {} new chunks", label, indexed_count)
        return indexed_count

    def index_dataset_1(self) -> int:
        """
        Index Dataset 1 — Pakistan Laws JSON.
        
        Returns:
            Number of chunks indexed.
        """
        logger.info("═══ Indexing Dataset 1: Pakistan Laws JSON ═══")
        start = time.time()

        documents = load_laws_json()
        if not documents:
            logger.warning("Dataset 1: No documents to index")
            return 0

        chunks = chunk_documents(documents, label="Dataset 1")
        count = self._index_chunks(chunks, "Dataset 1")

        elapsed = time.time() - start
        logger.info("Dataset 1 complete in {:.1f}s | {} chunks indexed", elapsed, count)
        return count

    def index_dataset_2(self) -> int:
        """
        Index Dataset 2 — PPC Markdown.
        
        Tracks PPC section numbers for deduplication against Dataset 1.
        
        Returns:
            Number of chunks indexed.
        """
        logger.info("═══ Indexing Dataset 2: PPC Markdown ═══")
        start = time.time()

        documents = load_ppc_markdown()
        if not documents:
            logger.warning("Dataset 2: No documents to index")
            return 0

        # Track PPC sections from Dataset 2 for dedup priority
        for doc in documents:
            sec = doc["metadata"].get("section", "")
            if sec:
                self._ppc_sections.add(sec)

        chunks = chunk_documents(documents, label="Dataset 2")

        # Deduplication: skip PPC chunks from Dataset 1 if Dataset 2 has the same section
        # This is handled by the hash-based dedup — Dataset 2 chunks will have different
        # text and thus different hashes. But we also check the collection for existing
        # PPC sections from Dataset 1 and could remove them. For simplicity and data
        # integrity, we rely on the fact that Dataset 2 PPC content is more authoritative
        # and both are kept — the retriever will rank by distance.

        count = self._index_chunks(chunks, "Dataset 2")

        elapsed = time.time() - start
        logger.info("Dataset 2 complete in {:.1f}s | {} chunks indexed", elapsed, count)
        return count

    def index_dataset_3(self) -> int:
        """
        Index Dataset 3 — Supreme Court Judgments.
        
        Processes in batches of JUDGMENT_BATCH_SIZE files.
        Logs progress every 200 files.
        
        Returns:
            Number of chunks indexed.
        """
        logger.info("═══ Indexing Dataset 3: Supreme Court Judgments ═══")
        start = time.time()

        all_files = get_judgment_files()
        if not all_files:
            logger.warning("Dataset 3: No judgment files found")
            return 0

        total_chunks_indexed = 0
        processed = 0

        for i in range(0, len(all_files), JUDGMENT_BATCH_SIZE):
            batch_files = all_files[i:i + JUDGMENT_BATCH_SIZE]
            documents = load_judgments_batch(batch_files)

            if documents:
                chunks = chunk_documents(documents, label=f"Dataset 3 batch {i // JUDGMENT_BATCH_SIZE}")
                count = self._index_chunks(chunks, f"Dataset 3 batch {i // JUDGMENT_BATCH_SIZE}")
                total_chunks_indexed += count

            processed += len(batch_files)

            # Log progress every 200 files
            if processed % 200 == 0 or processed == len(all_files):
                logger.info(
                    "Dataset 3 progress: {}/{} files processed | {} chunks indexed so far",
                    processed, len(all_files), total_chunks_indexed
                )

        elapsed = time.time() - start
        logger.info("Dataset 3 complete in {:.1f}s | {} chunks indexed", elapsed, total_chunks_indexed)
        return total_chunks_indexed

    def run_full_pipeline(self) -> dict:
        """
        Run the complete indexing pipeline: Dataset 1 → 2 → 3.
        
        Returns:
            Summary dict with counts and timings.
        """
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║   Juris AI — Full Indexing Pipeline          ║")
        logger.info("╚══════════════════════════════════════════════╝")

        start = time.time()

        # Load existing hashes for deduplication
        self._load_existing_hashes()

        # Run pipeline in order
        d1_count = self.index_dataset_1()
        d2_count = self.index_dataset_2()
        d3_count = self.index_dataset_3()

        elapsed = time.time() - start
        total_in_collection = self.get_collection_size()

        summary = {
            "dataset_1_chunks": d1_count,
            "dataset_2_chunks": d2_count,
            "dataset_3_chunks": d3_count,
            "total_new_chunks": d1_count + d2_count + d3_count,
            "total_in_collection": total_in_collection,
            "total_skipped_dedup": self._total_skipped,
            "total_errors": self._total_errors,
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║   Indexing Pipeline Complete                 ║")
        logger.info("╠══════════════════════════════════════════════╣")
        logger.info("║ Dataset 1 (Laws JSON):    {:>6} chunks       ║", d1_count)
        logger.info("║ Dataset 2 (PPC Markdown): {:>6} chunks       ║", d2_count)
        logger.info("║ Dataset 3 (SC Judgments):  {:>6} chunks      ║", d3_count)
        logger.info("║ Total in collection:      {:>6} chunks       ║", total_in_collection)
        logger.info("║ Skipped (dedup):          {:>6}              ║", self._total_skipped)
        logger.info("║ Errors:                   {:>6}              ║", self._total_errors)
        logger.info("║ Time elapsed:           {:>6.1f}s            ║", elapsed)
        logger.info("╚══════════════════════════════════════════════╝")

        return summary
