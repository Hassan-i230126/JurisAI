"""
Juris AI — Case Search Tool
Semantic search for Supreme Court criminal judgments via ChromaDB.
"""

import asyncio
from typing import Optional, List

import httpx
from loguru import logger

from app.tools.base import ToolBase
from app.models.schemas import ToolResult


class CaseSearchTool(ToolBase):
    """
    Searches Supreme Court criminal judgments using semantic similarity
    against the judgment subset of the ChromaDB vector index.
    
    Also performs a health check against the Supreme Court website
    to provide a supplementary link in results.
    """

    name = "case_search"
    description = "Search Supreme Court criminal judgments by topic, section, or keyword using semantic search."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query describing the case topic or legal issue"
            },
            "year_filter": {
                "type": "string",
                "description": "Optional year filter (e.g. '2019')"
            }
        },
        "required": ["query"]
    }

    def __init__(self, chroma_collection=None):
        """
        Initialize the case search tool.
        
        Args:
            chroma_collection: ChromaDB collection containing indexed documents.
        """
        self.collection = chroma_collection

    async def run(self, **kwargs) -> ToolResult:
        """
        Execute a judgment search.
        
        Args:
            query: The search query text.
            year_filter: Optional year to filter results.
            
        Returns:
            ToolResult with matching judgment snippets.
        """
        query = kwargs.get("query", "")
        year_filter = kwargs.get("year_filter")

        if not query:
            return ToolResult(
                success=False,
                data=None,
                formatted_text="Please provide a search query.",
                error_message="Missing query"
            )

        if not self.collection:
            return ToolResult(
                success=False,
                data=None,
                formatted_text="Case search index is not available. Please run the indexer first.",
                error_message="ChromaDB collection not initialized"
            )

        try:
            # Build the where filter for judgments only
            where_filter = {"doc_type": "judgment"}
            if year_filter:
                where_filter["year"] = year_filter

            # Perform semantic search using ChromaDB
            # We need to embed the query first using Ollama
            import ollama as ollama_client

            # Generate embedding in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            embedding_response = await loop.run_in_executor(
                None,
                lambda: ollama_client.embeddings(model="bge-m3", prompt=query)
            )
            query_embedding = embedding_response["embedding"]

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where=where_filter,
            )

            # Check Supreme Court website accessibility
            sc_note = await self._check_sc_website()

            # Format results
            if not results or not results["documents"] or not results["documents"][0]:
                return ToolResult(
                    success=True,
                    data=[],
                    formatted_text=f"No judgments found matching '{query}'.{sc_note}"
                )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            formatted_results = [f"**Supreme Court Judgment Search: '{query}'**\n"]
            result_data = []

            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                # Filter out results with high distance (not meaningfully related)
                if dist > 1.4:
                    continue

                case_ref = meta.get("case_ref", "Unknown")
                year = meta.get("year", "Unknown")
                filename = meta.get("filename", "Unknown")

                # Truncate document text for display
                snippet = doc[:500] + "..." if len(doc) > 500 else doc

                formatted_results.append(
                    f"**{i+1}. {case_ref} ({year})**\n"
                    f"File: {filename} | Relevance: {1 - dist:.2f}\n"
                    f"{snippet}\n"
                )

                result_data.append({
                    "case_ref": case_ref,
                    "year": year,
                    "filename": filename,
                    "distance": dist,
                    "snippet": snippet,
                })

            if not result_data:
                return ToolResult(
                    success=True,
                    data=[],
                    formatted_text=f"No sufficiently relevant judgments found for '{query}'.{sc_note}"
                )

            formatted_results.append(sc_note)

            return ToolResult(
                success=True,
                data=result_data,
                formatted_text="\n".join(formatted_results)
            )

        except Exception as e:
            logger.error("Case search error: {}", str(e))
            return ToolResult(
                success=False,
                data=None,
                formatted_text=f"Case search failed: {str(e)}. Please try a different query.",
                error_message=str(e)
            )

    async def _check_sc_website(self) -> str:
        """
        Health check the Supreme Court website.
        
        Returns a note string about accessing the full judgment text online.
        Non-blocking — returns gracefully on timeout or failure.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get("https://www.supremecourt.gov.pk/")
                if response.status_code == 200:
                    return "\n\n📌 For full judgment texts, visit: https://www.supremecourt.gov.pk/"
        except Exception:
            pass  # Silently proceed — this is a nice-to-have, not critical

        return ""
