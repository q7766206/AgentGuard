# -*- coding: utf-8 -*-
"""Tests for agentguard.audit module."""

import json
import os
import threading
import time

import pytest

from agentguard.audit import AuditTrail
from agentguard.types import AuditRecord, EventType, LogLevel


class TestAuditTrailBasic:
    def test_creation_with_defaults(self):
        trail = AuditTrail()
        assert trail.session_id  # auto-generated
        assert trail.total_records == 0

    def test_creation_with_custom_session_id(self):
        trail = AuditTrail(session_id="my-session")
        assert trail.session_id == "my-session"

    def test_record_creates_audit_record(self):
        trail = AuditTrail()
        rec = trail.record(
            level=LogLevel.INFO,
            event_type=EventType.TOOL_CALL_START,
            category="tool",
            action="web_search",
            detail="Searching for AI safety",
            metadata={"query": "AI safety"},
        )
        assert rec is not None
        assert isinstance(rec, AuditRecord)
        assert rec.level == LogLevel.INFO
        assert rec.action == "web_search"
        assert rec.metadata == {"query": "AI safety"}
        assert trail.total_records == 1

    def test_record_respects_min_level(self):
        trail = AuditTrail(min_level=LogLevel.WARN)
        rec_info = trail.record(
            level=LogLevel.INFO,
            event_type=EventType.CUSTOM,
            category="test",
            action="should_be_dropped",
        )
        rec_warn = trail.record(
            level=LogLevel.WARN,
            event_type=EventType.CUSTOM,
            category="test",
            action="should_be_kept",
        )
        assert rec_info is None
        assert rec_warn is not None
        assert trail.total_records == 1

    def test_multiple_records(self):
        trail = AuditTrail()
        for i in range(10):
            trail.record(
                level=LogLevel.INFO,
                event_type=EventType.CUSTOM,
                category="test",
                action=f"action_{i}",
            )
        assert trail.total_records == 10


class TestAuditTrailQuerying:
    def test_get_records_most_recent_first(self):
        trail = AuditTrail()
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "first")
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "second")
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "third")
        records = trail.get_records(limit=10)
        assert records[0].action == "third"
        assert records[2].action == "first"

    def test_get_records_with_level_filter(self):
        trail = AuditTrail()
        trail.record(LogLevel.DEBUG, EventType.CUSTOM, "test", "debug")
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "info")
        trail.record(LogLevel.ERROR, EventType.CUSTOM, "test", "error")
        records = trail.get_records(level=LogLevel.ERROR)
        assert len(records) == 1
        assert records[0].action == "error"

    def test_get_records_with_category_filter(self):
        trail = AuditTrail()
        trail.record(LogLevel.INFO, EventType.CUSTOM, "llm", "chat")
        trail.record(LogLevel.INFO, EventType.CUSTOM, "tool", "search")
        trail.record(LogLevel.INFO, EventType.CUSTOM, "llm", "embed")
        records = trail.get_records(category="llm")
        assert len(records) == 2

    def test_get_all_chronological(self):
        trail = AuditTrail()
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "first")
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "second")
        records = trail.get_all()
        assert records[0].action == "first"
        assert records[1].action == "second"


class TestAuditTrailSubscribers:
    def test_subscriber_receives_records(self):
        trail = AuditTrail()
        received = []
        trail.subscribe(lambda rec: received.append(rec))

        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "action1")
        trail.record(LogLevel.WARN, EventType.CUSTOM, "test", "action2")

        assert len(received) == 2
        assert received[0].action == "action1"
        assert received[1].action == "action2"

    def test_multiple_subscribers(self):
        trail = AuditTrail()
        received_a = []
        received_b = []
        trail.subscribe(lambda rec: received_a.append(rec))
        trail.subscribe(lambda rec: received_b.append(rec))

        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "action1")

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_bad_subscriber_doesnt_crash(self):
        trail = AuditTrail()

        def bad_subscriber(rec):
            raise RuntimeError("I'm broken!")

        received = []
        trail.subscribe(bad_subscriber)
        trail.subscribe(lambda rec: received.append(rec))

        # Should not raise
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "action1")

        # Good subscriber still receives the record
        assert len(received) == 1

    def test_unsubscribe(self):
        trail = AuditTrail()
        received = []
        callback = lambda rec: received.append(rec)
        trail.subscribe(callback)

        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "before")
        trail.unsubscribe(callback)
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "after")

        assert len(received) == 1
        assert received[0].action == "before"


class TestAuditTrailPersistence:
    def test_persist_to_jsonl(self, tmp_path):
        path = os.path.join(str(tmp_path), "audit.jsonl")
        trail = AuditTrail(persist=True, persist_path=path)
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "persisted_action")
        trail.close()

        # Verify file exists and contains the record
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["action"] == "persisted_action"

    def test_context_manager(self, tmp_path):
        path = os.path.join(str(tmp_path), "audit.jsonl")
        with AuditTrail(persist=True, persist_path=path) as trail:
            trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "ctx_action")

        # File should be flushed and closed
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1


class TestAuditTrailMetrics:
    def test_metrics(self):
        trail = AuditTrail()
        trail.record(LogLevel.INFO, EventType.CUSTOM, "llm", "chat")
        trail.record(LogLevel.ERROR, EventType.ERROR, "tool", "crash")
        trail.record(LogLevel.INFO, EventType.CUSTOM, "llm", "embed")

        metrics = trail.get_metrics()
        assert metrics["total_records"] == 3
        assert metrics["level_counts"]["info"] == 2
        assert metrics["level_counts"]["error"] == 1
        assert metrics["category_counts"]["llm"] == 2
        assert metrics["category_counts"]["tool"] == 1
        assert metrics["elapsed_seconds"] >= 0


class TestAuditTrailExport:
    def test_export_json(self):
        trail = AuditTrail()
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "action1")
        trail.record(LogLevel.WARN, EventType.CUSTOM, "test", "action2")

        json_str = trail.export_json()
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["action"] == "action1"

    def test_export_records(self):
        trail = AuditTrail()
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "action1")
        records = trail.export_records()
        assert isinstance(records, list)
        assert isinstance(records[0], dict)
        assert records[0]["action"] == "action1"


class TestAuditTrailClear:
    def test_clear_resets_everything(self):
        trail = AuditTrail()
        trail.record(LogLevel.INFO, EventType.CUSTOM, "test", "action1")
        trail.record(LogLevel.ERROR, EventType.ERROR, "test", "action2")
        assert trail.total_records == 2

        trail.clear()
        assert trail.total_records == 0
        assert len(trail.get_records()) == 0


class TestAuditTrailThreadSafety:
    def test_concurrent_recording(self):
        trail = AuditTrail()
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(100):
                    trail.record(
                        LogLevel.INFO,
                        EventType.CUSTOM,
                        "test",
                        f"thread_{thread_id}_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert trail.total_records == 1000

    def test_concurrent_recording_with_subscriber(self):
        trail = AuditTrail()
        received = []
        lock = threading.Lock()

        def safe_subscriber(rec):
            with lock:
                received.append(rec)

        trail.subscribe(safe_subscriber)
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    trail.record(
                        LogLevel.INFO,
                        EventType.CUSTOM,
                        "test",
                        f"thread_{thread_id}_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(received) == 500


class TestAuditTrailRepr:
    def test_repr(self):
        trail = AuditTrail(session_id="test-123")
        r = repr(trail)
        assert "test-123" in r
        assert "records=0" in r
