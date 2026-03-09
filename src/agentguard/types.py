# -*- coding: utf-8 -*-
"""
AgentGuard Type Definitions

All shared types, enums, and data structures used across AgentGuard modules.
Zero external dependencies — only Python stdlib.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Sequence


# ============================================================
# Log / Severity Levels
# ============================================================

class LogLevel(IntEnum):
    """Severity levels for audit records."""
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    CRITICAL = 50


LEVEL_NAMES: Dict[LogLevel, str] = {
    LogLevel.TRACE: "TRACE",
    LogLevel.DEBUG: "DEBUG",
    LogLevel.INFO: "INFO",
    LogLevel.WARN: "WARN",
    LogLevel.ERROR: "ERROR",
    LogLevel.CRITICAL: "CRITICAL",
}


# ============================================================
# Event Types (what happened)
# ============================================================

class EventType(str, Enum):
    """Categories of auditable events in an agent's lifecycle."""
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    RULE_VIOLATION = "rule_violation"
    ALERT = "alert"
    CUSTOM = "custom"


# ============================================================
# Rule Actions & Severity
# ============================================================

class RuleAction(str, Enum):
    """What to do when a rule is violated."""
    LOG = "log"            # Just log a warning
    THROTTLE = "throttle"  # Slow down (add delay before next operation)
    BLOCK = "block"        # Block the current operation
    KILL = "kill"          # Kill the agent loop immediately
    NOTIFY = "notify"      # Send alert to configured channels


class RuleSeverity(str, Enum):
    """How serious a rule violation is."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# Alert Levels
# ============================================================

class AlertLevel(IntEnum):
    """Severity levels for alerts."""
    INFO = 0
    WARNING = 1
    DANGER = 2
    CRITICAL = 3


ALERT_LEVEL_NAMES: Dict[AlertLevel, str] = {
    AlertLevel.INFO: "INFO",
    AlertLevel.WARNING: "WARNING",
    AlertLevel.DANGER: "DANGER",
    AlertLevel.CRITICAL: "CRITICAL",
}


# ============================================================
# Core Data Structures
# ============================================================

@dataclass(frozen=True)
class AuditRecord:
    """
    Single immutable audit log entry.

    Every agent action (LLM call, tool execution, decision) produces one AuditRecord.
    Records are append-only and never modified after creation.
    """
    id: str
    timestamp: float
    level: LogLevel
    event_type: EventType
    category: str
    action: str
    detail: str = ""
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "level": LEVEL_NAMES.get(self.level, str(self.level)),
            "event_type": self.event_type.value,
            "category": self.category,
            "action": self.action,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "session_id": self.session_id,
        }


class _RecordIDGenerator:
    """Thread-safe monotonic ID generator for audit records."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def next_id(self, session_id: str = "") -> str:
        with self._lock:
            self._counter += 1
            return f"ag-{session_id[:8]}-{self._counter:06d}" if session_id else f"ag-{self._counter:06d}"


# Module-level singleton
_id_gen = _RecordIDGenerator()


def make_record(
    level: LogLevel,
    event_type: EventType,
    category: str,
    action: str,
    detail: str = "",
    duration_ms: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = "",
) -> AuditRecord:
    """Factory function to create an AuditRecord with auto-generated ID and timestamp."""
    return AuditRecord(
        id=_id_gen.next_id(session_id),
        timestamp=time.time(),
        level=level,
        event_type=event_type,
        category=category,
        action=action,
        detail=detail,
        duration_ms=duration_ms,
        metadata=metadata or {},
        session_id=session_id,
    )


@dataclass
class RuleViolation:
    """Result of a rule evaluation that detected a violation."""
    rule_name: str
    severity: RuleSeverity
    action: RuleAction
    message: str
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """An alert generated by the alert system."""
    id: str
    level: AlertLevel
    title: str
    message: str
    source: str = ""
    rule_name: str = ""
    auto_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "level": ALERT_LEVEL_NAMES.get(self.level, str(self.level)),
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "rule_name": self.rule_name,
            "auto_action": self.auto_action,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "acknowledged": self.acknowledged,
        }


class _AlertIDGenerator:
    """Thread-safe ID generator for alerts."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"alert-{self._counter:06d}"


_alert_id_gen = _AlertIDGenerator()


def make_alert(
    level: AlertLevel,
    title: str,
    message: str,
    source: str = "",
    rule_name: str = "",
    auto_action: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Alert:
    """Factory function to create an Alert with auto-generated ID."""
    return Alert(
        id=_alert_id_gen.next_id(),
        level=level,
        title=title,
        message=message,
        source=source,
        rule_name=rule_name,
        auto_action=auto_action,
        metadata=metadata or {},
    )


@dataclass
class RuleContext:
    """
    Runtime context passed to rules during evaluation.

    Contains aggregated metrics that rules can inspect to make decisions.
    """
    session_id: str = ""
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0
    current_intent: str = ""
    message_count: int = 0
    recent_tool_calls: List[str] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardStats:
    """Aggregated statistics from an AgentGuard session."""
    session_id: str
    total_records: int = 0
    total_alerts: int = 0
    total_violations: int = 0
    total_blocks: int = 0
    total_kills: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_errors: int = 0
    start_time: float = field(default_factory=time.time)
    rules_active: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_records": self.total_records,
            "total_alerts": self.total_alerts,
            "total_violations": self.total_violations,
            "total_blocks": self.total_blocks,
            "total_kills": self.total_kills,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "total_llm_calls": self.total_llm_calls,
            "total_errors": self.total_errors,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "rules_active": self.rules_active,
        }
