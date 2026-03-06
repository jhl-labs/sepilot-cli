"""Caching system for prompts and tool results"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """Represents a cached entry"""
    key: str
    value: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl <= 0:  # No expiration
            return False
        return time.time() - self.timestamp > self.ttl


class PromptCache:
    """Cache for generated prompts to avoid regeneration"""

    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        """Initialize prompt cache

        Args:
            max_size: Maximum number of cached prompts
            default_ttl: Default time to live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        # OrderedDict provides O(1) move_to_end for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def _generate_key(self, profile: str, context: str, tools: str) -> str:
        """Generate cache key from prompt components"""
        content = json.dumps([profile, context, tools], sort_keys=True)
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def get(self, profile: str, context: str, tools: str) -> str | None:
        """Get cached prompt if available"""
        key = self._generate_key(profile, context, tools)

        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                entry.hits += 1
                # O(1) move to end for LRU
                self._cache.move_to_end(key)
                return entry.value
            else:
                # Remove expired entry
                del self._cache[key]

        return None

    def set(self, profile: str, context: str, tools: str, prompt: str, ttl: int | None = None) -> None:
        """Cache a generated prompt"""
        key = self._generate_key(profile, context, tools)
        ttl = ttl if ttl is not None else self.default_ttl

        # Evict LRU entry if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._cache.popitem(last=False)  # O(1) eviction of oldest

        entry = CacheEntry(
            key=key,
            value=prompt,
            timestamp=time.time(),
            ttl=ttl
        )
        self._cache[key] = entry
        self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cached prompts"""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(entry.hits for entry in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "entries": [
                {
                    "key": key[:8],
                    "hits": entry.hits,
                    "age": int(time.time() - entry.timestamp),
                    "expired": entry.is_expired()
                }
                for key, entry in self._cache.items()
            ]
        }


class ToolResultCache:
    """Cache for tool execution results"""

    def __init__(self, max_size: int = 50, default_ttl: int = 300):
        """Initialize tool result cache

        Args:
            max_size: Maximum number of cached results
            default_ttl: Default time to live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def _generate_key(self, tool_name: str, params: dict[str, Any]) -> str:
        """Generate cache key from tool call"""
        # Sort params for consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        content = f"{tool_name}:{params_str}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def should_cache(self, tool_name: str) -> bool:
        """Determine if tool results should be cached"""
        # Don't cache certain tools that have side effects
        no_cache_tools = ['file_write', 'file_edit', 'bash', 'process', 'git']
        return tool_name not in no_cache_tools

    def get(self, tool_name: str, params: dict[str, Any]) -> Any | None:
        """Get cached tool result if available"""
        if not self.should_cache(tool_name):
            return None

        key = self._generate_key(tool_name, params)

        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                entry.hits += 1
                self._cache.move_to_end(key)  # Mark as recently used
                return entry.value
            else:
                del self._cache[key]

        return None

    def set(self, tool_name: str, params: dict[str, Any], result: Any, ttl: int | None = None) -> None:
        """Cache a tool execution result"""
        if not self.should_cache(tool_name):
            return

        key = self._generate_key(tool_name, params)
        ttl = ttl if ttl is not None else self.default_ttl

        # Evict oldest entry if cache is full (O(1) with OrderedDict)
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._cache.popitem(last=False)  # Remove least recently used

        entry = CacheEntry(
            key=key,
            value=result,
            timestamp=time.time(),
            ttl=ttl
        )
        self._cache[key] = entry
        self._cache.move_to_end(key)  # Mark as most recently used

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern

        Args:
            pattern: Pattern to match (e.g., "file_read:*")

        Returns:
            Number of entries invalidated
        """
        to_remove = []
        for key in self._cache:
            if pattern == "*" or key.startswith(pattern.replace("*", "")):
                to_remove.append(key)

        for key in to_remove:
            del self._cache[key]

        return len(to_remove)

    def clear(self) -> None:
        """Clear all cached results"""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(entry.hits for entry in self._cache.values())
        expired = sum(1 for entry in self._cache.values() if entry.is_expired())

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "expired_entries": expired,
            "cache_efficiency": total_hits / max(len(self._cache), 1)
        }


class CacheManager:
    """Centralized cache management"""

    def __init__(self):
        """Initialize cache manager"""
        self.prompt_cache = PromptCache()
        self.tool_cache = ToolResultCache()
        self._enabled = True

    def enable(self) -> None:
        """Enable caching"""
        self._enabled = True

    def disable(self) -> None:
        """Disable caching"""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self._enabled

    def get_prompt(self, profile: str, context: str, tools: str) -> str | None:
        """Get cached prompt"""
        if not self._enabled:
            return None
        return self.prompt_cache.get(profile, context, tools)

    def cache_prompt(self, profile: str, context: str, tools: str, prompt: str) -> None:
        """Cache a prompt"""
        if self._enabled:
            self.prompt_cache.set(profile, context, tools, prompt)

    def get_tool_result(self, tool_name: str, params: dict[str, Any]) -> Any | None:
        """Get cached tool result"""
        if not self._enabled:
            return None
        return self.tool_cache.get(tool_name, params)

    def cache_tool_result(self, tool_name: str, params: dict[str, Any], result: Any) -> None:
        """Cache a tool result"""
        if self._enabled:
            self.tool_cache.set(tool_name, params, result)

    def clear_all(self) -> None:
        """Clear all caches"""
        self.prompt_cache.clear()
        self.tool_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get overall cache statistics"""
        return {
            "enabled": self._enabled,
            "prompt_cache": self.prompt_cache.get_stats(),
            "tool_cache": self.tool_cache.get_stats()
        }
