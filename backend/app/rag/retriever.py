"""
Juris AI — Retriever Module
Query-time retrieval: embed query → search ChromaDB → return ranked chunks.
"""

import asyncio
import math
import time
from typing import List, Optional

import chromadb
import httpx
from loguru import logger

from app.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    RAG_TOP_K,
    RAG_DISTANCE_THRESHOLD,
)
from app.models.schemas import RetrievedChunk
from app.utils.cache import LRUCache


class LegalRetriever:
    """
    Query-time retrieval engine for Juris AI.
    
    Embeds queries using bge-m3 via Ollama, searches the ChromaDB
    vector store, and returns ranked, filtered chunks with citations.
    """

    def __init__(self, collection=None):
        """
        Initialize the retriever.
        
        Args:
            collection: Pre-initialized ChromaDB collection.
                       If None, creates a new client connection.
        """
        if collection:
            self.collection = collection
        else:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        self.embedding_cache = LRUCache()
        self._embedding_timeout = httpx.Timeout(connect=2.0, read=30.0, write=30.0, pool=10.0)
        logger.info(
            "LegalRetriever initialized | collection={} | docs={}",
            CHROMA_COLLECTION_NAME, self.collection.count()
        )

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query text using bge-m3 via Ollama.
        
        Checks the LRU cache first. If not cached, calls Ollama
        in a thread executor to avoid blocking the event loop.
        
        Args:
            text: The query text to embed.
            
        Returns:
            The embedding vector as a list of floats.
        """
        # Check cache first
        cached = self.embedding_cache.get(text)
        if cached is not None:
            logger.debug("Embedding cache hit for query: {}...", text[:50])
            return cached

        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": text,
        }

        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=self._embedding_timeout) as client:
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()

        embedding = body.get("embedding")
        if not embedding:
            raise RuntimeError("Embedding response missing 'embedding' field")

        if any((not math.isfinite(float(x))) for x in embedding):
            raise RuntimeError("Embedding vector contains non-finite values")

        # Cache the result
        self.embedding_cache.put(text, embedding)

        return embedding

    async def retrieve(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        doc_type_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Pipeline:
        1. Check cache for query embedding
        2. Generate embedding via bge-m3
        3. Query ChromaDB with optional doc_type filter
        4. Filter out chunks with distance > threshold
        5. Return sorted list of RetrievedChunk objects
        
        Args:
            query: The user's query text.
            top_k: Number of top results to retrieve (default from config).
            doc_type_filter: Optional filter: "statute" or "judgment".
            
        Returns:
            List of RetrievedChunk objects sorted by distance ascending.
            Empty list if no chunks pass the distance threshold.
        """
        start_time = time.time()

        try:
            # Step 1-2: Generate embedding
            embedding = await self.embed_query(query)

            # Step 3: Build query parameters
            query_params = {
                "query_embeddings": [embedding],
                "n_results": top_k,
            }

            if doc_type_filter:
                query_params["where"] = {"doc_type": doc_type_filter}

            # Execute ChromaDB query in executor to keep event loop responsive
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.query(**query_params),
            )

            if not results or not results["documents"] or not results["documents"][0]:
                latency_ms = (time.time() - start_time) * 1000
                logger.info(
                    "RAG retrieval | query={}... | chunks=0 | latency_ms={:.0f}",
                    query[:50], latency_ms
                )
                return []

            # Step 4-5: Process and filter results
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            chunks: List[RetrievedChunk] = []

            for doc, meta, dist in zip(documents, metadatas, distances):
                # Filter out chunks beyond the distance threshold
                if dist > RAG_DISTANCE_THRESHOLD:
                    continue

                # Build citation string
                citation = self._build_citation(meta, dist)

                chunk = RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    act=meta.get("act", ""),
                    section=meta.get("section", ""),
                    doc_type=meta.get("doc_type", ""),
                    distance=dist,
                    citation=citation,
                )
                chunks.append(chunk)

            # Sort by distance ascending (most relevant first)
            chunks.sort(key=lambda c: c.distance)

            latency_ms = (time.time() - start_time) * 1000
            top_dist = chunks[0].distance if chunks else -1
            logger.info(
                "RAG retrieval | query={}... | chunks={} | top_distance={:.3f} | latency_ms={:.0f}",
                query[:50], len(chunks), top_dist, latency_ms
            )

            return chunks

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "RAG retrieval failed | query={}... | error={} | latency_ms={:.0f}",
                query[:50], str(e), latency_ms
            )
            return []

    @staticmethod
    def _build_citation(metadata: dict, distance: float) -> str:
        """
        Build a formatted citation string from chunk metadata.
        
        Examples:
            - "PPC §302" (for statutes with section numbers)
            - "Pakistan Penal Code" (for statutes without section)
            - "SC Judgment CA-42 (2019)" (for judgments)
        """
        doc_type = metadata.get("doc_type", "")
        source = metadata.get("source", "")

        if doc_type == "statute":
            act = metadata.get("act", "Unknown Act")
            section = metadata.get("section", "")

            # Abbreviate common act names
            act_abbrev = {
                "Pakistan Penal Code": "PPC",
                "Code of Criminal Procedure": "CrPC",
                "Qanun-e-Shahadat Order": "QSO",
                "Anti-Terrorism Act": "ATA",
            }
            act_short = act_abbrev.get(act, act)

            if section:
                return f"{act_short} §{section}"
            else:
                return act_short

        elif doc_type == "judgment":
            case_ref = metadata.get("case_ref", "Unknown")
            year = metadata.get("year", "")
            if year and year != "unknown":
                return f"SC Judgment {case_ref} ({year})"
            else:
                return f"SC Judgment {case_ref}"

        else:
            return source or "Unknown Source"
