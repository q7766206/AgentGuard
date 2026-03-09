# -*- coding: utf-8 -*-
"""
AgentGuard Exceptions

All custom exceptions raised by AgentGuard.
Each exception carries context about what triggered it and what rule was involved.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AgentGuardError(Exception):
    """Base exception for all AgentGuard errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.context = context or {}
        super().__init__(message)


class AgentGuardBlock(AgentGuardError):
    """
    Raised when a rule blocks the current operation.

    The agent should catch this, skip the blocked operation, and continue
    with the next step (or ask the user for guidance).
    """

    def __init__(
        self,
        message: str,
        rule_name: str = "",
        tool_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rule_name = rule_name
        self.tool_name = tool_name
        super().__init__(message, context)

    def __str__(self) -> str:
        parts = [f"[AgentGuard BLOCK]"]
        if self.rule_name:
            parts.append(f"Rule: {self.rule_name}")
        if self.tool_name:
            parts.append(f"Tool: {self.tool_name}")
        parts.append(str(self.args[0]))
        return " | ".join(parts)


class AgentGuardKill(AgentGuardError):
    """
    Raised when a rule decides the agent must stop immediately.

    This is a hard stop — the agent loop should terminate.
    Used for critical safety violations like runaway token usage or error cascades.
    """

    def __init__(
        self,
        message: str,
        rule_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rule_name = rule_name
        super().__init__(message, context)

    def __str__(self) -> str:
        parts = [f"[AgentGuard KILL]"]
        if self.rule_name:
            parts.append(f"Rule: {self.rule_name}")
        parts.append(str(self.args[0]))
        return " | ".join(parts)


class AgentGuardThrottle(AgentGuardError):
    """
    Raised when a rule throttles the current operation.

    The agent should slow down — typically by adding a delay before the next operation.
    """

    def __init__(
        self,
        message: str,
        delay_seconds: float = 1.0,
        rule_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.rule_name = rule_name
        super().__init__(message, context)

    def __str__(self) -> str:
        return (
            f"[AgentGuard THROTTLE] Rule: {self.rule_name} | "
            f"Delay: {self.delay_seconds}s | {self.args[0]}"
        )


class AgentGuardConfigError(AgentGuardError):
    """Raised when AgentGuard is misconfigured."""
    pass


class KYAError(AgentGuardError):
    """Base exception for KYA (Know Your Agent) related errors."""
    pass


class KYADenied(KYAError):
    """Raised when an agent tries to use a capability it's not authorized for."""

    def __init__(
        self,
        message: str,
        capability: str = "",
        agent_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.capability = capability
        self.agent_name = agent_name
        super().__init__(message, context)

    def __str__(self) -> str:
        return (
            f"[KYA DENIED] Agent: {self.agent_name} | "
            f"Capability: {self.capability} | {self.args[0]}"
        )


class KYABudgetExceeded(KYAError):
    """Raised when an agent exceeds its declared resource budget."""

    def __init__(
        self,
        message: str,
        budget_type: str = "",
        limit: int = 0,
        current: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.budget_type = budget_type
        self.limit = limit
        self.current = current
        super().__init__(message, context)

    def __str__(self) -> str:
        return (
            f"[KYA BUDGET EXCEEDED] {self.budget_type}: "
            f"{self.current}/{self.limit} | {self.args[0]}"
        )


class KYAManifestError(KYAError):
    """Raised when an agent.yaml manifest is invalid or cannot be parsed."""
    pass
