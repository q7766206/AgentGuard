# -*- coding: utf-8 -*-
"""Tests for agentguard.types module."""

import pytest
from agentguard.types import (
    AuditRecord,
    Alert,
    AlertLevel,
    EventType,
    LogLevel,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
    GuardStats,
    make_record,
    make_alert,
)


class TestAuditRecord:
    def test_make_record_creates_valid_record(self):
        record = make_record(
            level=LogLevel.INFO,
            event_type=EventType.TOOL_CALL_START,
            category="test",
            action="test_action",
            detail="test detail",
            duration_ms=100,
            metadata={"key": "value"},
            session_id="test-session",
        )
        assert isinstance(record, AuditRecord)
        assert record.level == LogLevel.INFO
        assert record.event_type == EventType.TOOL_CALL_START
        assert record.category == "test"
        assert record.action == "test_action"
        assert record.detail == "test detail"
        assert record.duration_ms == 100
        assert record.metadata == {"key": "value"}
        assert record.session_id == "test-session"
        assert record.id.startswith("ag-")
        assert record.timestamp > 0

    def test_record_to_dict(self):
        record = make_record(
            level=LogLevel.ERROR,
            event_type=EventType.ERROR,
            category="test",
            action="error",
        )
        d = record.to_dict()
        assert "id" in d
        assert "timestamp" in d
        assert "iso_time" in d
        assert d["level"] == "ERROR"
        assert d["event_type"] == "error"

    def test_record_is_immutable(self):
        record = make_record(
            level=LogLevel.INFO,
            event_type=EventType.CUSTOM,
            category="test",
            action="test",
        )
        with pytest.raises(AttributeError):
            record.level = LogLevel.ERROR


class TestAlert:
    def test_make_alert_creates_valid_alert(self):
        alert = make_alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test",
            source="test_source",
            rule_name="TestRule",
        )
        assert isinstance(alert, Alert)
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.source == "test_source"
        assert alert.rule_name == "TestRule"
        assert alert.id.startswith("alert-")
        assert alert.timestamp > 0
        assert not alert.acknowledged

    def test_alert_to_dict(self):
        alert = make_alert(
            level=AlertLevel.CRITICAL,
            title="Critical",
            message="Critical message",
        )
        d = alert.to_dict()
        assert d["level"] == "CRITICAL"
        assert d["title"] == "Critical"
        assert "timestamp" in d
        assert "iso_time" in d


class TestEnums:
    def test_log_level_order(self):
        assert LogLevel.TRACE < LogLevel.DEBUG < LogLevel.INFO < LogLevel.WARN < LogLevel.ERROR < LogLevel.CRITICAL

    def test_rule_action_values(self):
        assert RuleAction.LOG.value == "log"
        assert RuleAction.BLOCK.value == "block"
        assert RuleAction.KILL.value == "kill"

    def test_event_type_values(self):
        assert EventType.LLM_CALL_START.value == "llm_call_start"
        assert EventType.TOOL_CALL_END.value == "tool_call_end"


class TestGuardStats:
    def test_guard_stats_creation(self):
        stats = GuardStats(session_id="test-session")
        assert stats.session_id == "test-session"
        assert stats.total_records == 0
        assert stats.elapsed_seconds >= 0

    def test_guard_stats_to_dict(self):
        stats = GuardStats(session_id="test-session")
        stats.total_records = 10
        stats.total_alerts = 2
        d = stats.to_dict()
        assert d["session_id"] == "test-session"
        assert d["total_records"] == 10
        assert d["total_alerts"] == 2
        assert "elapsed_seconds" in d


class TestRuleContext:
    def test_rule_context_defaults(self):
        ctx = RuleContext()
        assert ctx.session_id == ""
        assert ctx.total_tokens == 0
        assert ctx.recent_tool_calls == []
