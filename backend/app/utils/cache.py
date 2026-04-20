"""
Juris AI — Simple LRU Cache for Embeddings
Thread-safe LRU cache to avoid redundant embedding API calls.
"""

from collections import OrderedDict
from typing import Optional, List
import threading

from app.config import EMBEDDING_CACHE_SIZE


class LRUCache:
    """
    Thread-safe Least Recently Used (LRU) cache.
    
    Used primarily for caching embedding vectors keyed by query text,
    so repeated or similar queries within a session don't require
    re-computation via the embedding model.
    """

    def __init__(self, max_size: int = EMBEDDING_CACHE_SIZE):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of entries to store before eviction.
        """
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """
        Retrieve a value from the cache.
        
        If the key exists, it is moved to the end (most recently used).
        
        Args:
            key: The cache key (typically the query text).
            
        Returns:
            The cached embedding vector, or None if not found.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: List[float]) -> None:
        """
        Store a value in the cache, evicting the oldest entry if at capacity.
        
        Args:
            key: The cache key (typically the query text).
            value: The embedding vector to cache.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._max_size:
                    # Evict the least recently used entry (first item)
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self._cache)

    @property
    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
        }
