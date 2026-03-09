# -*- coding: utf-8 -*-
"""
AgentGuard Memory Storage

In-memory ring buffer storage backend.
Fast, bounded, thread-safe. Ideal for real-time monitoring and short-lived sessions.

Records are stored in a collections.deque with a configurable max size.
When the buffer is full, the oldest records are automatically evicted.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Optional

from agentguard.storage.base import BaseStorage
from agentguard.types import AuditRecord, LogLevel


class MemoryStorage(BaseStorage):
    """
    In-memory ring buffer for audit records.

    Thread-safe. When ``maxlen`` is reached, oldest records are dropped.

    Args:
        maxlen: Maximum number of records to keep. Defaults to 10000.
    """

    def __init__(self, maxlen: int = 10_000) -> None:
        self._buffer: Deque[AuditRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._maxlen = maxlen

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def append(self, record: AuditRecord) -> None:
        """Append a record to the ring buffer. Thread-safe."""
        with self._lock:
            self._buffer.append(record)

    def get_records(
        self,
        limit: int = 100,
        offset: int = 0,
        level: Optional[LogLevel] = None,
        category: Optional[str] = None,
    ) -> List[AuditRecord]:
        """
        Retrieve records from the buffer, most recent first.

        Filtering is applied before limit/offset.
        """
        with self._lock:
            records = list(self._buffer)

        # Most recent first
        records.reverse()

        # Apply filters
        if level is not None:
            records = [r for r in records if r.level >= level]
        if category is not None:
            records = [r for r in records if r.category == category]

        # Apply pagination
        return records[offset : offset + limit]

    def count(self) -> int:
        """Return current number of records in buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Remove all records from buffer."""
        with self._lock:
            self._buffer.clear()

    def get_all(self) -> List[AuditRecord]:
        """Return all records in chronological order."""
        with self._lock:
            return list(self._buffer)
