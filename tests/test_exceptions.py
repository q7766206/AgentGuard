# -*- coding: utf-8 -*-
"""Tests for agentguard.exceptions module."""

import pytest
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


class TestAgentGuardBlock:
    def test_block_basic(self):
        exc = AgentGuardBlock("Blocked dangerous command")
        assert "Blocked dangerous command" in str(exc)
        assert "[AgentGuard BLOCK]" in str(exc)

    def test_block_with_rule_and_tool(self):
        exc = AgentGuardBlock(
            "rm -rf detected",
            rule_name="SensitiveOp",
            tool_name="shell",
        )
        assert "SensitiveOp" in str(exc)
        assert "shell" in str(exc)
        assert exc.rule_name == "SensitiveOp"
        assert exc.tool_name == "shell"

    def test_block_is_agent_guard_error(self):
        exc = AgentGuardBlock("test")
        assert isinstance(exc, AgentGuardError)
        assert isinstance(exc, Exception)


class TestAgentGuardKill:
    def test_kill_basic(self):
        exc = AgentGuardKill("Token budget exceeded")
        assert "[AgentGuard KILL]" in str(exc)

    def test_kill_with_rule(self):
        exc = AgentGuardKill("Too many errors", rule_name="ErrorCascade")
        assert "ErrorCascade" in str(exc)


class TestAgentGuardThrottle:
    def test_throttle_basic(self):
        exc = AgentGuardThrottle("Slow down", delay_seconds=2.0)
        assert exc.delay_seconds == 2.0
        assert "[AgentGuard THROTTLE]" in str(exc)


class TestKYAExceptions:
    def test_kya_denied(self):
        exc = KYADenied(
            "Not authorized to use shell",
            capability="shell_execute",
            agent_name="research-agent",
        )
        assert "KYA DENIED" in str(exc)
        assert "shell_execute" in str(exc)
        assert "research-agent" in str(exc)
        assert isinstance(exc, KYAError)
        assert isinstance(exc, AgentGuardError)

    def test_kya_budget_exceeded(self):
        exc = KYABudgetExceeded(
            "Token limit reached",
            budget_type="tokens",
            limit=100000,
            current=150000,
        )
        assert "KYA BUDGET EXCEEDED" in str(exc)
        assert "150000" in str(exc)
        assert "100000" in str(exc)

    def test_kya_manifest_error(self):
        exc = KYAManifestError("Invalid YAML")
        assert isinstance(exc, KYAError)


class TestExceptionContext:
    def test_context_dict(self):
        exc = AgentGuardError("test", context={"key": "value"})
        assert exc.context == {"key": "value"}

    def test_context_default_empty(self):
        exc = AgentGuardError("test")
        assert exc.context == {}
