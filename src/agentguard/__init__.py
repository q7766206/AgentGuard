# -*- coding: utf-8 -*-
"""
AgentGuard — Open-source security middleware for AI agents.

Audit every action. Enforce safety rules. Verify agent identity.

Usage::

    from agentguard import AgentGuard, LoopDetection, TokenBudget, SensitiveOp

    guard = AgentGuard(rules=[
        LoopDetection(max_repeats=5),
        TokenBudget(max_tokens=100000),
        SensitiveOp(),
    ])

    # In your agent loop:
    guard.before_tool_call("shell", {"command": "rm -rf /"})
    # → Raises AgentGuardBlock!

For framework integration::

    # LangChain 1.0
    from agentguard.adapters.langchain import AgentGuardMiddleware
    agent = create_agent(model="...", middleware=[AgentGuardMiddleware(guard)])

"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = [
    # Main entry point
    "AgentGuard",
    # Types & data structures
    "AuditRecord",
    "Alert",
    "RuleViolation",
    "RuleContext",
    "GuardStats",
    "LogLevel",
    "EventType",
    "RuleAction",
    "RuleSeverity",
    "AlertLevel",
    "make_record",
    "make_alert",
    # Exceptions
    "AgentGuardError",
    "AgentGuardBlock",
    "AgentGuardKill",
    "AgentGuardThrottle",
    "AgentGuardConfigError",
    "KYAError",
    "KYADenied",
    "KYABudgetExceeded",
    "KYAManifestError",
]

# Types (always available, zero dependencies)
from agentguard.types import (
    AuditRecord,
    Alert,
    AlertLevel,
    EventType,
    GuardStats,
    LogLevel,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
    make_alert,
    make_record,
)

# Exceptions (always available, zero dependencies)
from agentguard.exceptions import (
    AgentGuardBlock,
    AgentGuardConfigError,
    AgentGuardError,
    AgentGuardKill,
    AgentGuardThrottle,
    KYABudgetExceeded,
    KYADenied,
    KYAError,
    KYAManifestError,
)


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy import for AgentGuard main class to keep import fast."""
    if name == "AgentGuard":
        from agentguard.core import AgentGuard
        return AgentGuard
    raise AttributeError(f"module 'agentguard' has no attribute {name!r}")
