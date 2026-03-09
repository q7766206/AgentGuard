# -*- coding: utf-8 -*-
"""
AgentGuard Core

The main AgentGuard class — the single entry point for developers.
Wires together AuditTrail, RuleEngine, and exception handling into
a clean, minimal API.

Usage::

    from agentguard import AgentGuard
    from agentguard.rules import LoopDetection, TokenBudget, SensitiveOp

    guard = AgentGuard(rules=[
        LoopDetection(max_repeats=5),
        TokenBudget(max_tokens=100000),
        SensitiveOp(),
    ])

    # Hook into your agent loop:
    guard.before_tool_call("shell", {"command": "ls -la"})
    # ... tool executes ...
    guard.after_tool_call("shell", result="file list...", duration_ms=150)

    guard.before_llm_call(messages=[...])
    # ... LLM responds ...
    guard.after_llm_call(response="...", tokens=500)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agentguard.audit import AuditTrail
from agentguard.exceptions import (
    AgentGuardBlock,
    AgentGuardKill,
    AgentGuardThrottle,
)
from agentguard.rules.base import BaseRule
from agentguard.rules.engine import RuleEngine
from agentguard.types import (
    AuditRecord,
    EventType,
    GuardStats,
    LogLevel,
    RuleAction,
    RuleContext,
    RuleViolation,
)


class AgentGuard:
    """
    Security middleware for AI agents.

    Provides a simple API to audit agent actions, enforce safety rules,
    and react to violations. Designed to be framework-agnostic — works with
    LangChain, CrewAI, OpenClaw, or any custom agent framework.

    Args:
        rules: List of rules to enforce. See ``agentguard.rules.builtin``.
        session_id: Unique session identifier. Auto-generated if not provided.
        persist: If True, write audit records to disk (JSONL).
        persist_path: Path to the JSONL file.
        memory_maxlen: Max records in the in-memory ring buffer.
        min_level: Minimum log level to record.
        raise_on_block: If True, raise AgentGuardBlock when a rule blocks.
        raise_on_kill: If True, raise AgentGuardKill when a rule kills.
    """

    def __init__(
        self,
        rules: Optional[List[BaseRule]] = None,
        session_id: str = "",
        persist: bool = False,
        persist_path: str = "./agentguard_audit.jsonl",
        memory_maxlen: int = 10_000,
        min_level: LogLevel = LogLevel.INFO,
        raise_on_block: bool = True,
        raise_on_kill: bool = True,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._raise_on_block = raise_on_block
        self._raise_on_kill = raise_on_kill

        # Audit trail
        self._trail = AuditTrail(
            session_id=self._session_id,
            memory_maxlen=memory_maxlen,
            persist=persist,
            persist_path=persist_path,
            min_level=min_level,
        )

        # Rule engine
        self._engine = RuleEngine(rules=rules)
        self._engine.attach(self._trail)

    # ================================================================
    # Properties
    # ================================================================

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def trail(self) -> AuditTrail:
        """Access the underlying AuditTrail for advanced queries."""
        return self._trail

    @property
    def engine(self) -> RuleEngine:
        """Access the underlying RuleEngine for advanced configuration."""
        return self._engine

    # ================================================================
    # Internal: violation snapshot mechanism
    # ================================================================

    def _violation_snapshot(self) -> int:
        """Take a snapshot of current violation count before recording."""
        return len(self._engine.violations)

    def _check_new_violations(
        self, prev_count: int, tool_name: str = ""
    ) -> Optional[RuleViolation]:
        """
        Check if new violations appeared since the snapshot.
        Only reacts to violations that were triggered by the most recent record.
        """
        all_violations = self._engine.violations
        new_violations = all_violations[prev_count:]

        if not new_violations:
            return None

        # Process the most severe new violation
        # Priority: KILL > BLOCK > THROTTLE > LOG
        kill_v = next((v for v in new_violations if v.action == RuleAction.KILL), None)
        block_v = next((v for v in new_violations if v.action == RuleAction.BLOCK), None)
        throttle_v = next((v for v in new_violations if v.action == RuleAction.THROTTLE), None)

        if kill_v and self._raise_on_kill:
            raise AgentGuardKill(
                kill_v.message,
                rule_name=kill_v.rule_name,
                context=kill_v.metadata,
            )

        if block_v and self._raise_on_block:
            raise AgentGuardBlock(
                block_v.message,
                rule_name=block_v.rule_name,
                tool_name=tool_name,
                context=block_v.metadata,
            )

        if throttle_v:
            delay = throttle_v.metadata.get("delay_seconds", 1.0)
            raise AgentGuardThrottle(
                throttle_v.message,
                delay_seconds=delay,
                rule_name=throttle_v.rule_name,
                context=throttle_v.metadata,
            )

        # Return the first new violation for informational purposes
        return new_violations[0]

    # ================================================================
    # Agent Lifecycle Hooks
    # ================================================================

    def before_tool_call(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        **extra_metadata: Any,
    ) -> Optional[RuleViolation]:
        """
        Call BEFORE executing a tool. Records the event and checks rules.

        Raises:
            AgentGuardBlock: If a rule blocks this tool call.
            AgentGuardKill: If a rule kills the agent.
        """
        metadata = {"args": args or {}}
        metadata.update(extra_metadata)
        snap = self._violation_snapshot()

        self._trail.record(
            level=LogLevel.INFO,
            event_type=EventType.TOOL_CALL_START,
            category="tool",
            action=tool_name,
            detail=f"Calling tool: {tool_name}",
            metadata=metadata,
        )

        return self._check_new_violations(snap, tool_name=tool_name)

    def after_tool_call(
        self,
        tool_name: str,
        result: str = "",
        duration_ms: int = 0,
        error: Optional[str] = None,
        **extra_metadata: Any,
    ) -> Optional[RuleViolation]:
        """Call AFTER a tool finishes. Records the result and checks rules."""
        metadata: Dict[str, Any] = {"duration_ms": duration_ms}
        metadata.update(extra_metadata)
        snap = self._violation_snapshot()

        if error:
            self._trail.record(
                level=LogLevel.ERROR,
                event_type=EventType.ERROR,
                category="tool",
                action=tool_name,
                detail=f"Tool error: {error[:500]}",
                duration_ms=duration_ms,
                metadata=metadata,
            )
        else:
            self._trail.record(
                level=LogLevel.INFO,
                event_type=EventType.TOOL_CALL_END,
                category="tool",
                action=tool_name,
                detail=result[:500] if result else "",
                duration_ms=duration_ms,
                metadata=metadata,
            )

        return self._check_new_violations(snap, tool_name=tool_name)

    def before_llm_call(
        self,
        model: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        **extra_metadata: Any,
    ) -> Optional[RuleViolation]:
        """Call BEFORE an LLM invocation."""
        metadata: Dict[str, Any] = {"model": model}
        if messages:
            metadata["message_count"] = len(messages)
        metadata.update(extra_metadata)

        self._trail.record(
            level=LogLevel.INFO,
            event_type=EventType.LLM_CALL_START,
            category="llm",
            action=model or "llm_call",
            detail=f"LLM call: {model}",
            metadata=metadata,
        )
        return None

    def after_llm_call(
        self,
        model: str = "",
        response: str = "",
        tokens: int = 0,
        duration_ms: int = 0,
        error: Optional[str] = None,
        **extra_metadata: Any,
    ) -> Optional[RuleViolation]:
        """Call AFTER an LLM responds."""
        metadata: Dict[str, Any] = {
            "model": model,
            "total_tokens": tokens,
        }
        metadata.update(extra_metadata)
        snap = self._violation_snapshot()

        if error:
            self._trail.record(
                level=LogLevel.ERROR,
                event_type=EventType.ERROR,
                category="llm",
                action=model or "llm_call",
                detail=f"LLM error: {error[:500]}",
                duration_ms=duration_ms,
                metadata=metadata,
            )
        else:
            self._trail.record(
                level=LogLevel.INFO,
                event_type=EventType.LLM_CALL_END,
                category="llm",
                action=model or "llm_call",
                detail=response[:200] if response else "",
                duration_ms=duration_ms,
                metadata=metadata,
            )

        return self._check_new_violations(snap)

    def record_error(
        self,
        error: str,
        category: str = "agent",
        action: str = "error",
        **extra_metadata: Any,
    ) -> Optional[RuleViolation]:
        """Record an error event."""
        snap = self._violation_snapshot()
        self._trail.record(
            level=LogLevel.ERROR,
            event_type=EventType.ERROR,
            category=category,
            action=action,
            detail=error[:500],
            metadata=dict(extra_metadata),
        )
        return self._check_new_violations(snap)

    def record_custom(
        self,
        action: str,
        detail: str = "",
        level: LogLevel = LogLevel.INFO,
        category: str = "custom",
        **extra_metadata: Any,
    ) -> None:
        """Record a custom event."""
        self._trail.record(
            level=level,
            event_type=EventType.CUSTOM,
            category=category,
            action=action,
            detail=detail,
            metadata=dict(extra_metadata),
        )

    # ================================================================
    # Convenience
    # ================================================================

    def on_violation(self, callback: Callable[[RuleViolation, AuditRecord], None]) -> None:
        """Register a callback for rule violations."""
        self._engine.on_violation(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from audit trail and rule engine."""
        trail_metrics = self._trail.get_metrics()
        engine_stats = self._engine.get_stats()
        return {
            "session_id": self._session_id,
            "audit": trail_metrics,
            "rules": engine_stats,
        }

    def get_violations(self) -> List[RuleViolation]:
        """Get all violations detected in this session."""
        return self._engine.violations

    def get_records(self, **kwargs: Any) -> List[AuditRecord]:
        """Query audit records. See AuditTrail.get_records() for args."""
        return self._trail.get_records(**kwargs)

    def export_json(self) -> str:
        """Export all audit records as JSON."""
        return self._trail.export_json()

    # ================================================================
    # Lifecycle
    # ================================================================

    def reset(self) -> None:
        """Reset audit trail and rule engine for a new session."""
        self._trail.clear()
        self._engine.reset()

    def close(self) -> None:
        """Flush and close all resources."""
        self._engine.detach()
        self._trail.close()

    def __enter__(self) -> "AgentGuard":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self._engine.get_stats()
        return (
            f"AgentGuard(session={self._session_id!r}, "
            f"rules={stats['rules_active']}, "
            f"violations={stats['total_violations']})"
        )
