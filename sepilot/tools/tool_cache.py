"""Tool call caching system for performance optimization"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class ToolCallCache:
    """LRU cache with TTL for tool call results

    Features:
    - LRU eviction policy
    - Time-To-Live (TTL) for cache entries
    - Cache statistics (hit rate, size)
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """Initialize cache

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _get_cache_key(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Generate cache key from tool name and arguments using hash.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Cache key string or None if args cannot be serialized
        """
        try:
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
            args_hash = hashlib.md5(args_str.encode(), usedforsecurity=False).hexdigest()
            return f"{tool_name}:{args_hash}"
        except (TypeError, ValueError) as e:
            logger.debug(f"Cannot cache {tool_name}: args not serializable ({e})")
            return None

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired

        Args:
            timestamp: Entry creation timestamp

        Returns:
            True if expired
        """
        return (time.time() - timestamp) > self.ttl_seconds

    def get(self, tool_name: str, args: dict[str, Any]) -> Any | None:
        """Get cached result for tool call (thread-safe)

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Cached result or None if not found/expired
        """
        with self._lock:
            key = self._get_cache_key(tool_name, args)

            # Cannot cache if key generation failed
            if key is None:
                self.misses += 1
                return None

            if key not in self.cache:
                self.misses += 1
                return None

            # Check TTL
            result, timestamp = self.cache[key]
            if self._is_expired(timestamp):
                # Remove expired entry
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return result

    def set(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        """Cache tool call result (thread-safe)

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool execution result
        """
        with self._lock:
            key = self._get_cache_key(tool_name, args)

            # Cannot cache if key generation failed
            if key is None:
                return  # Skip caching

            # Update existing entry
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = (result, time.time())
                return

            # Add new entry
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (LRU)
                self.cache.popitem(last=False)
                self.evictions += 1

            self.cache[key] = (result, time.time())

    def invalidate(self, tool_name: str | None = None) -> None:
        """Invalidate cache entries (thread-safe)

        Args:
            tool_name: If specified, only invalidate entries for this tool.
                      If None, clear entire cache.
        """
        with self._lock:
            if tool_name is None:
                self.cache.clear()
            else:
                # Remove entries matching tool name
                keys_to_remove = [
                    key for key in self.cache
                    if key.startswith(f"{tool_name}:")
                ]
                for key in keys_to_remove:
                    del self.cache[key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics (thread-safe)

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate_pct": round(hit_rate, 2),
                "total_requests": total_requests
            }

    def clear_expired(self) -> int:
        """Remove all expired entries (thread-safe)

        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            keys_to_remove = [
                key for key, (_, timestamp) in self.cache.items()
                if (current_time - timestamp) > self.ttl_seconds
            ]

            for key in keys_to_remove:
                del self.cache[key]

            return len(keys_to_remove)

    def reset_stats(self) -> None:
        """Reset statistics counters (thread-safe)"""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0


# Global cache instance (can be configured per agent)
_global_cache: ToolCallCache | None = None


def get_global_cache() -> ToolCallCache:
    """Get or create global cache instance

    Returns:
        Global ToolCallCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ToolCallCache()
    return _global_cache


def configure_global_cache(max_size: int = 100, ttl_seconds: int = 300) -> None:
    """Configure global cache

    Args:
        max_size: Maximum cache size
        ttl_seconds: TTL for cache entries
    """
    global _global_cache
    _global_cache = ToolCallCache(max_size=max_size, ttl_seconds=ttl_seconds)
