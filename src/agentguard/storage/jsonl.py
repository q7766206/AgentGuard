# -*- coding: utf-8 -*-
"""
AgentGuard JSONL Storage

Append-only JSONL (JSON Lines) file storage backend.
Each line is a JSON-serialized AuditRecord.

Designed for:
- Persistent audit trails that survive process restarts
- Post-mortem analysis of agent behavior
- Compliance and regulatory requirements
- Integration with log aggregation tools (ELK, Splunk, etc.)

Thread-safe: uses a lock to serialize writes.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Optional

from agentguard.storage.base import BaseStorage
from agentguard.types import AuditRecord, EventType, LogLevel, make_record


class JSONLStorage(BaseStorage):
    """
    Append-only JSONL file storage for audit records.

    Each record is written as a single JSON line, flushed immediately.
    Reads parse the entire file (suitable for post-mortem, not real-time queries).

    Args:
        path: Path to the JSONL file. Created if it doesn't exist.
        flush_every: Flush to disk after this many writes. 1 = every write (safest).
    """

    def __init__(self, path: str = "./agentguard_audit.jsonl", flush_every: int = 1) -> None:
        self._path = Path(path)
        self._flush_every = max(1, flush_every)
        self._write_count = 0
        self._lock = threading.Lock()
        self._file = None
        self._ensure_dir()
        self._open()

    def _ensure_dir(self) -> None:
        """Create parent directories if they don't exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        """Open the file for appending."""
        self._file = open(self._path, "a", encoding="utf-8")

    def append(self, record: AuditRecord) -> None:
        """Append a record as a JSON line. Thread-safe."""
        line = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._file is None or self._file.closed:
                self._open()
            self._file.write(line + "\n")
            self._write_count += 1
            if self._write_count >= self._flush_every:
                self._file.flush()
                self._write_count = 0

    def get_records(
        self,
        limit: int = 100,
        offset: int = 0,
        level: Optional[LogLevel] = None,
        category: Optional[str] = None,
    ) -> List[AuditRecord]:
        """
        Read records from the JSONL file.

        Note: This reads the entire file. For large files, consider using
        external tools (jq, grep) or a database backend.
        """
        records = self._read_all()

        # Most recent first
        records.reverse()

        # Apply filters
        if level is not None:
            records = [r for r in records if r.level >= level]
        if category is not None:
            records = [r for r in records if r.category == category]

        return records[offset : offset + limit]

    def count(self) -> int:
        """Count records by counting lines in the file."""
        if not self._path.exists():
            return 0
        with open(self._path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def clear(self) -> None:
        """Truncate the file, removing all records."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()
            # Truncate
            with open(self._path, "w", encoding="utf-8") as f:
                pass
            self._open()
            self._write_count = 0

    def close(self) -> None:
        """Flush and close the file."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
                self._file = None

    def _read_all(self) -> List[AuditRecord]:
        """Parse all records from the JSONL file."""
        if not self._path.exists():
            return []

        records: List[AuditRecord] = []
        # Flush pending writes before reading
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()

        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = self._dict_to_record(data)
                    records.append(record)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip malformed lines — don't crash on corrupted data
                    continue
        return records

    @staticmethod
    def _dict_to_record(data: dict) -> AuditRecord:
        """Reconstruct an AuditRecord from a dictionary."""
        # Map level name back to enum
        level_name = data.get("level", "INFO")
        level_map = {v: k for k, v in {
            LogLevel.TRACE: "TRACE", LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO", LogLevel.WARN: "WARN",
            LogLevel.ERROR: "ERROR", LogLevel.CRITICAL: "CRITICAL",
        }.items()}
        level = level_map.get(level_name, LogLevel.INFO)

        # Map event_type string back to enum
        event_type_str = data.get("event_type", "custom")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.CUSTOM

        return AuditRecord(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", 0.0),
            level=level,
            event_type=event_type,
            category=data.get("category", ""),
            action=data.get("action", ""),
            detail=data.get("detail", ""),
            duration_ms=data.get("duration_ms", 0),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id", ""),
        )

    def __del__(self) -> None:
        """Ensure file is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
