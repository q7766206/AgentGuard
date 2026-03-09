# -*- coding: utf-8 -*-
"""
AgentGuard Storage Base

Abstract base class for all storage backends.
Storage backends are responsible for persisting and retrieving AuditRecords.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agentguard.types import AuditRecord, LogLevel


class BaseStorage(ABC):
    """
    Abstract base class for audit record storage.

    Implementations must be thread-safe — AgentGuard may write records
    from multiple threads concurrently.
    """

    @abstractmethod
    def append(self, record: AuditRecord) -> None:
        """Append a single audit record to storage."""
        ...

    @abstractmethod
    def get_records(
        self,
        limit: int = 100,
        offset: int = 0,
        level: Optional[LogLevel] = None,
        category: Optional[str] = None,
    ) -> List[AuditRecord]:
        """
        Retrieve audit records with optional filtering.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).
            level: If set, only return records at or above this level.
            category: If set, only return records matching this category.

        Returns:
            List of AuditRecords, most recent first.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of records in storage."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all records from storage."""
        ...

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
