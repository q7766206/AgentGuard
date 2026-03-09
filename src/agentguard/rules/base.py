# -*- coding: utf-8 -*-
"""
AgentGuard Rule Base Class

All rules inherit from BaseRule and implement the evaluate() method.
Rules are stateful — they can track history across multiple evaluations
to detect patterns like loops, cascading errors, or budget overruns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agentguard.types import (
    AuditRecord,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
)


class BaseRule(ABC):
    """
    Abstract base class for all AgentGuard rules.

    A rule inspects an AuditRecord (the current event) and a RuleContext
    (aggregated session metrics), then decides whether a violation occurred.

    Subclasses must implement:
        - evaluate(): inspect the record and context, return a RuleViolation or None.

    Subclasses may override:
        - reset(): clear internal state (called when the session resets).

    Attributes:
        name: Human-readable rule name (used in logs and alerts).
        enabled: If False, the rule is skipped during evaluation.
        severity: Default severity for violations from this rule.
        action: Default action to take when this rule fires.
    """

    def __init__(
        self,
        name: str = "",
        enabled: bool = True,
        severity: RuleSeverity = RuleSeverity.MEDIUM,
        action: RuleAction = RuleAction.LOG,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.severity = severity
        self.action = action

    @abstractmethod
    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        """
        Evaluate whether this record + context constitutes a violation.

        Args:
            record: The current audit event being processed.
            context: Aggregated session metrics (tokens, tool calls, errors, etc.).

        Returns:
            A RuleViolation if the rule is triggered, or None if everything is fine.
        """
        ...

    def reset(self) -> None:
        """
        Reset internal state. Called when the session resets.
        Override in subclasses that maintain state (e.g., counters, timestamps).
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"enabled={self.enabled}, severity={self.severity.value})"
        )
