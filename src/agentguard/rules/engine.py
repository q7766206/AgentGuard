# -*- coding: utf-8 -*-
"""
AgentGuard Rule Engine

The brain that connects audit records to rules.
Subscribes to the AuditTrail, evaluates every record against all active rules,
and fires actions (log, block, throttle, kill) when violations are detected.

Architecture:
    AuditTrail → (subscriber) → RuleEngine.on_record()
                                    ↓
                              for rule in rules:
                                  rule.evaluate(record, context)
                                    ↓
                              violation? → fire action → notify callbacks
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from agentguard.audit import AuditTrail
from agentguard.rules.base import BaseRule
from agentguard.types import (
    AuditRecord,
    EventType,
    LogLevel,
    RuleAction,
    RuleContext,
    RuleViolation,
)


# Type alias for violation callbacks
ViolationCallback = Callable[[RuleViolation, AuditRecord], None]


class RuleEngine:
    """
    Evaluates audit records against a set of rules.

    The RuleEngine maintains a RuleContext that aggregates session-level metrics
    (total tokens, tool calls, errors, etc.) and passes it to each rule along
    with the current audit record.

    The engine can be connected to an AuditTrail via ``attach()``, which
    subscribes it to the audit stream. It can also be used standalone by
    calling ``evaluate()`` directly.

    Args:
        rules: List of rules to evaluate. Can be modified later via add_rule/remove_rule.

    Example::

        from agentguard.rules.builtin import LoopDetection, TokenBudget, SensitiveOp

        engine = RuleEngine(rules=[
            LoopDetection(max_repeats=5),
            TokenBudget(max_tokens=100000),
            SensitiveOp(),
        ])

        # Attach to audit trail
        engine.attach(audit_trail)

        # Or evaluate manually
        violations = engine.evaluate(record)
    """

    def __init__(self, rules: Optional[List[BaseRule]] = None) -> None:
        self._rules: List[BaseRule] = list(rules or [])
        self._context = RuleContext()
        self._violations: List[RuleViolation] = []
        self._callbacks: List[ViolationCallback] = []
        self._lock = threading.Lock()
        self._ctx_lock = threading.Lock()
        self._attached_trail: Optional[AuditTrail] = None
        # Stable reference for subscribe/unsubscribe (bound methods create new objects each access)
        self._on_record_ref: Callable[[AuditRecord], None] = self._on_record

        # Stats
        self._total_evaluations = 0
        self._total_violations = 0
        self._total_blocks = 0
        self._total_kills = 0

    # ================================================================
    # Rule Management
    # ================================================================

    def add_rule(self, rule: BaseRule) -> None:
        """Add a rule to the engine."""
        with self._lock:
            self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.name != name]
            return len(self._rules) < before

    def get_rule(self, name: str) -> Optional[BaseRule]:
        """Get a rule by name."""
        with self._lock:
            for r in self._rules:
                if r.name == name:
                    return r
        return None

    @property
    def rules(self) -> List[BaseRule]:
        """List of active rules (copy)."""
        with self._lock:
            return list(self._rules)

    @property
    def context(self) -> RuleContext:
        """Current rule evaluation context."""
        return self._context

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(self, record: AuditRecord) -> List[RuleViolation]:
        """
        Evaluate a single audit record against all enabled rules.

        Updates the internal RuleContext before evaluation.
        Returns a list of violations (empty if no rules fired).

        Args:
            record: The audit record to evaluate.

        Returns:
            List of RuleViolations from rules that fired.
        """
        # Update context from record
        self._update_context(record)

        violations: List[RuleViolation] = []

        with self._lock:
            rules_snapshot = [r for r in self._rules if r.enabled]

        for rule in rules_snapshot:
            try:
                violation = rule.evaluate(record, self._context)
                if violation is not None:
                    violations.append(violation)
            except Exception:
                # A broken rule should never crash the engine
                pass

        # Record violations
        if violations:
            with self._lock:
                self._violations.extend(violations)
                self._total_violations += len(violations)
                for v in violations:
                    if v.action == RuleAction.BLOCK:
                        self._total_blocks += 1
                    elif v.action == RuleAction.KILL:
                        self._total_kills += 1

            # Notify callbacks
            for v in violations:
                self._notify_callbacks(v, record)

        with self._lock:
            self._total_evaluations += 1

        return violations

    def _update_context(self, record: AuditRecord) -> None:
        """Update the RuleContext based on the incoming record."""
        with self._ctx_lock:
            # Track tokens
            if record.event_type == EventType.LLM_CALL_END:
                tokens = record.metadata.get("total_tokens", 0)
                self._context.total_tokens += tokens
                self._context.total_llm_calls += 1

            # Track tool calls
            if record.event_type == EventType.TOOL_CALL_START:
                self._context.total_tool_calls += 1
                self._context.recent_tool_calls.append(record.action)
                # Keep only last 50
                if len(self._context.recent_tool_calls) > 50:
                    self._context.recent_tool_calls = self._context.recent_tool_calls[-50:]

            # Track errors
            if record.event_type == EventType.ERROR or record.level >= LogLevel.ERROR:
                self._context.total_errors += 1
                self._context.recent_errors.append(record.detail[:200])
                if len(self._context.recent_errors) > 20:
                    self._context.recent_errors = self._context.recent_errors[-20:]

            # Update elapsed time
            self._context.elapsed_seconds = time.time() - (
                record.timestamp - self._context.elapsed_seconds
                if self._context.elapsed_seconds == 0
                else self._context.elapsed_seconds
            )

    # ================================================================
    # Attachment to AuditTrail
    # ================================================================

    def attach(self, trail: AuditTrail) -> None:
        """
        Subscribe this engine to an AuditTrail's record stream.

        Every new record will be automatically evaluated against all rules.

        Args:
            trail: The AuditTrail to subscribe to.
        """
        if self._attached_trail is not None:
            self.detach()
        self._attached_trail = trail
        trail.subscribe(self._on_record_ref)

    def detach(self) -> None:
        """Unsubscribe from the currently attached AuditTrail."""
        if self._attached_trail is not None:
            self._attached_trail.unsubscribe(self._on_record_ref)
            self._attached_trail = None

    def _on_record(self, record: AuditRecord) -> None:
        """Subscriber callback — evaluate the record."""
        self.evaluate(record)

    # ================================================================
    # Violation Callbacks
    # ================================================================

    def on_violation(self, callback: ViolationCallback) -> None:
        """
        Register a callback to be notified when any rule fires.

        The callback receives (violation, record) as arguments.
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, violation: RuleViolation, record: AuditRecord) -> None:
        """Notify all registered callbacks. Never raises."""
        for cb in self._callbacks:
            try:
                cb(violation, record)
            except Exception:
                pass

    # ================================================================
    # History & Stats
    # ================================================================

    @property
    def violations(self) -> List[RuleViolation]:
        """All violations detected so far (copy)."""
        with self._lock:
            return list(self._violations)

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        with self._lock:
            return {
                "total_evaluations": self._total_evaluations,
                "total_violations": self._total_violations,
                "total_blocks": self._total_blocks,
                "total_kills": self._total_kills,
                "rules_active": sum(1 for r in self._rules if r.enabled),
                "rules_total": len(self._rules),
                "context": {
                    "total_tokens": self._context.total_tokens,
                    "total_tool_calls": self._context.total_tool_calls,
                    "total_llm_calls": self._context.total_llm_calls,
                    "total_errors": self._context.total_errors,
                },
            }

    # ================================================================
    # Reset
    # ================================================================

    def reset(self) -> None:
        """Reset all rules and context. Used when starting a new session."""
        with self._lock:
            for rule in self._rules:
                rule.reset()
            self._context = RuleContext()
            self._violations.clear()
            self._total_evaluations = 0
            self._total_violations = 0
            self._total_blocks = 0
            self._total_kills = 0

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"RuleEngine(rules={len(self._rules)}, "
                f"violations={self._total_violations}, "
                f"evaluations={self._total_evaluations})"
            )
