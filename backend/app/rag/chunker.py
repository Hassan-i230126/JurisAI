"""
Juris AI — Text Chunker
Text chunking with metadata preservation using langchain-text-splitters.
"""

import hashlib
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.models.schemas import DocumentChunk


# ─── Chunker Configuration ───────────────────────────────────────────────────
# Approximate characters per token for English legal text (~4 chars/token)
CHARS_PER_TOKEN = 4

# Separators ordered from most to least desirable split points
SEPARATORS = ["\n\n", "\n", ". ", " "]

# Minimum chunk length in characters — skip tiny fragments
MIN_CHUNK_LENGTH = 80


def create_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a configured text splitter instance.
    
    Uses token-based sizing approximated via character count.
    chunk_size and chunk_overlap are in tokens (from config).
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * CHARS_PER_TOKEN,
        chunk_overlap=CHUNK_OVERLAP * CHARS_PER_TOKEN,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def hash_text(text: str) -> str:
    """
    Compute SHA-256 hash of text for deduplication.
    
    Args:
        text: The text to hash.
        
    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    text: str,
    metadata: Dict[str, str],
    splitter: RecursiveCharacterTextSplitter = None,
) -> List[DocumentChunk]:
    """
    Split a document into chunks with metadata inheritance.
    
    Each chunk inherits the parent document's metadata and receives
    additional chunk_index and total_chunks fields.
    
    Args:
        text: The full document text.
        metadata: The parent document's metadata dict.
        splitter: Optional pre-created splitter (created if None).
        
    Returns:
        List of DocumentChunk objects ready for indexing.
    """
    if not text or len(text.strip()) < MIN_CHUNK_LENGTH:
        return []

    if splitter is None:
        splitter = create_splitter()

    # Split the text
    chunks_text = splitter.split_text(text)

    # Filter out tiny chunks
    chunks_text = [c for c in chunks_text if len(c.strip()) >= MIN_CHUNK_LENGTH]

    if not chunks_text:
        return []

    total_chunks = len(chunks_text)
    result = []

    for i, chunk_text in enumerate(chunks_text):
        # Inherit parent metadata and add chunk-specific fields
        chunk_meta = dict(metadata)
        chunk_meta["chunk_index"] = str(i)
        chunk_meta["total_chunks"] = str(total_chunks)

        chunk = DocumentChunk(
            text=chunk_text.strip(),
            metadata=chunk_meta,
            chunk_hash=hash_text(chunk_text.strip()),
        )
        result.append(chunk)

    return result


def chunk_documents(
    documents: List[Dict],
    label: str = "documents",
) -> List[DocumentChunk]:
    """
    Chunk a list of documents, each with text and metadata.
    
    Args:
        documents: List of dicts with 'text' and 'metadata' keys.
        label: Label for logging (e.g. 'Dataset 1').
        
    Returns:
        List of all DocumentChunks across all documents.
    """
    splitter = create_splitter()
    all_chunks: List[DocumentChunk] = []

    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        chunks = chunk_document(text, metadata, splitter)
        all_chunks.extend(chunks)

    logger.info(
        "Chunking complete for {} | {} documents → {} chunks",
        label, len(documents), len(all_chunks)
    )

    return all_chunks
