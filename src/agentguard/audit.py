# -*- coding: utf-8 -*-
"""
AgentGuard Audit Trail

The central nervous system of AgentGuard. Every agent action flows through here,
producing immutable AuditRecords that are stored, broadcast to subscribers,
and available for rule evaluation.

Architecture:
    Agent Action → AuditTrail.record() → Storage (Memory + optional JSONL)
                                       → Subscribers (RuleEngine, UI, custom)
                                       → Metrics update

Key design decisions:
    1. Zero external dependencies — only Python stdlib
    2. Thread-safe — all operations use locks
    3. Subscriber pattern — RuleEngine subscribes to audit stream
    4. Dual storage — in-memory (fast queries) + optional disk (persistence)
    5. Non-blocking — record() never raises, never blocks the agent
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agentguard.storage.base import BaseStorage
from agentguard.storage.memory import MemoryStorage
from agentguard.storage.jsonl import JSONLStorage
from agentguard.types import (
    AuditRecord,
    EventType,
    LogLevel,
    make_record,
)


# Type alias for subscriber callbacks
AuditSubscriber = Callable[[AuditRecord], None]


class AuditTrail:
    """
    Structured audit logging for AI agent behavior.

    Records every LLM call, tool execution, decision, and error as an
    immutable AuditRecord. Supports real-time subscribers (for rule evaluation)
    and dual storage (memory + disk).

    Args:
        session_id: Unique identifier for this audit session. Auto-generated if not provided.
        memory_maxlen: Maximum records in the in-memory ring buffer.
        persist: If True, also write records to a JSONL file on disk.
        persist_path: Path to the JSONL file (only used if persist=True).
        min_level: Minimum log level to record. Records below this level are silently dropped.

    Example::

        trail = AuditTrail(session_id="my-session", persist=True)

        # Record an event
        trail.record(
            level=LogLevel.INFO,
            event_type=EventType.TOOL_CALL_START,
            category="tool",
            action="web_search",
            detail="Searching for 'AI safety'",
            metadata={"query": "AI safety"},
        )

        # Subscribe to real-time events
        trail.subscribe(lambda record: print(f"Event: {record.action}"))

        # Query records
        recent = trail.get_records(limit=10, level=LogLevel.WARN)
    """

    def __init__(
        self,
        session_id: str = "",
        memory_maxlen: int = 10_000,
        persist: bool = False,
        persist_path: str = "./agentguard_audit.jsonl",
        min_level: LogLevel = LogLevel.TRACE,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._min_level = min_level
        self._subscribers: List[AuditSubscriber] = []
        self._sub_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

        # Metrics
        self._total_records = 0
        self._level_counts: Dict[LogLevel, int] = {level: 0 for level in LogLevel}
        self._category_counts: Dict[str, int] = {}
        self._start_time = time.time()

        # Storage backends
        self._memory = MemoryStorage(maxlen=memory_maxlen)
        self._disk: Optional[JSONLStorage] = None
        if persist:
            self._disk = JSONLStorage(path=persist_path)

    @property
    def session_id(self) -> str:
        """The unique session identifier."""
        return self._session_id

    @property
    def total_records(self) -> int:
        """Total number of records created in this session."""
        return self._total_records

    # ================================================================
    # Recording
    # ================================================================

    def record(
        self,
        level: LogLevel,
        event_type: EventType,
        category: str,
        action: str,
        detail: str = "",
        duration_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditRecord]:
        """
        Create and store an audit record.

        This method is designed to never raise exceptions and never block
        the calling agent. If storage or subscriber notification fails,
        the error is silently swallowed.

        Args:
            level: Severity level of this event.
            event_type: What kind of event this is.
            category: High-level category (e.g., "llm", "tool", "agent").
            action: Specific action (e.g., "web_search", "chat_completion").
            detail: Human-readable description.
            duration_ms: How long the operation took (0 if not applicable).
            metadata: Additional structured data.

        Returns:
            The created AuditRecord, or None if the level was below min_level.
        """
        if level < self._min_level:
            return None

        try:
            rec = make_record(
                level=level,
                event_type=event_type,
                category=category,
                action=action,
                detail=detail,
                duration_ms=duration_ms,
                metadata=metadata,
                session_id=self._session_id,
            )

            # Store
            self._memory.append(rec)
            if self._disk is not None:
                self._disk.append(rec)

            # Update metrics
            with self._metrics_lock:
                self._total_records += 1
                self._level_counts[level] = self._level_counts.get(level, 0) + 1
                self._category_counts[category] = self._category_counts.get(category, 0) + 1

            # Notify subscribers (non-blocking)
            self._notify_subscribers(rec)

            return rec

        except Exception:
            # Never crash the agent because of audit logging
            return None

    # ================================================================
    # Querying
    # ================================================================

    def get_records(
        self,
        limit: int = 100,
        offset: int = 0,
        level: Optional[LogLevel] = None,
        category: Optional[str] = None,
    ) -> List[AuditRecord]:
        """
        Retrieve audit records from memory, most recent first.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            level: If set, only return records at or above this level.
            category: If set, only return records matching this category.
        """
        return self._memory.get_records(
            limit=limit, offset=offset, level=level, category=category
        )

    def get_all(self) -> List[AuditRecord]:
        """Return all records in chronological order."""
        return self._memory.get_all()

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregated metrics for this session."""
        with self._metrics_lock:
            return {
                "session_id": self._session_id,
                "total_records": self._total_records,
                "elapsed_seconds": round(time.time() - self._start_time, 2),
                "level_counts": {
                    name: self._level_counts.get(level, 0)
                    for level, name in {
                        LogLevel.TRACE: "trace",
                        LogLevel.DEBUG: "debug",
                        LogLevel.INFO: "info",
                        LogLevel.WARN: "warn",
                        LogLevel.ERROR: "error",
                        LogLevel.CRITICAL: "critical",
                    }.items()
                },
                "category_counts": dict(self._category_counts),
                "subscriber_count": len(self._subscribers),
            }

    # ================================================================
    # Subscribers
    # ================================================================

    def subscribe(self, callback: AuditSubscriber) -> None:
        """
        Register a callback to be notified of every new audit record.

        The callback receives the AuditRecord as its only argument.
        Callbacks should be fast and non-blocking — they run synchronously
        in the recording thread.

        Args:
            callback: A callable that accepts an AuditRecord.
        """
        with self._sub_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: AuditSubscriber) -> None:
        """Remove a previously registered subscriber."""
        with self._sub_lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def _notify_subscribers(self, record: AuditRecord) -> None:
        """Notify all subscribers of a new record. Never raises."""
        with self._sub_lock:
            subscribers = list(self._subscribers)

        for sub in subscribers:
            try:
                sub(record)
            except Exception:
                # Never let a bad subscriber crash the audit system
                pass

    # ================================================================
    # Export
    # ================================================================

    def export_json(self) -> str:
        """Export all records as a JSON array string."""
        records = self._memory.get_all()
        return json.dumps(
            [r.to_dict() for r in records],
            ensure_ascii=False,
            indent=2,
        )

    def export_records(self) -> List[Dict[str, Any]]:
        """Export all records as a list of dictionaries."""
        return [r.to_dict() for r in self._memory.get_all()]

    # ================================================================
    # Lifecycle
    # ================================================================

    def clear(self) -> None:
        """Clear all records from memory (disk records are preserved)."""
        self._memory.clear()
        with self._metrics_lock:
            self._total_records = 0
            self._level_counts = {level: 0 for level in LogLevel}
            self._category_counts = {}

    def close(self) -> None:
        """Flush disk storage and clean up resources."""
        if self._disk is not None:
            self._disk.close()

    def __enter__(self) -> "AuditTrail":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"AuditTrail(session_id={self._session_id!r}, "
            f"records={self._total_records}, "
            f"subscribers={len(self._subscribers)})"
        )
