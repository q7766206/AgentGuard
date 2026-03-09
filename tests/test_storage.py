# -*- coding: utf-8 -*-
"""Tests for agentguard.storage module."""

import os
import tempfile
import threading

import pytest

from agentguard.storage.memory import MemoryStorage
from agentguard.storage.jsonl import JSONLStorage
from agentguard.types import AuditRecord, EventType, LogLevel, make_record


def _make_test_record(
    level: LogLevel = LogLevel.INFO,
    category: str = "test",
    action: str = "test_action",
) -> AuditRecord:
    return make_record(
        level=level,
        event_type=EventType.CUSTOM,
        category=category,
        action=action,
        detail="test detail",
    )


# ============================================================
# MemoryStorage Tests
# ============================================================

class TestMemoryStorage:
    def test_append_and_count(self):
        store = MemoryStorage(maxlen=100)
        assert store.count() == 0
        store.append(_make_test_record())
        assert store.count() == 1
        store.append(_make_test_record())
        assert store.count() == 2

    def test_ring_buffer_eviction(self):
        store = MemoryStorage(maxlen=3)
        for i in range(5):
            store.append(_make_test_record(action=f"action_{i}"))
        assert store.count() == 3
        records = store.get_all()
        # Should keep the 3 most recent
        assert records[0].action == "action_2"
        assert records[1].action == "action_3"
        assert records[2].action == "action_4"

    def test_get_records_most_recent_first(self):
        store = MemoryStorage()
        store.append(_make_test_record(action="first"))
        store.append(_make_test_record(action="second"))
        store.append(_make_test_record(action="third"))
        records = store.get_records(limit=10)
        assert records[0].action == "third"
        assert records[2].action == "first"

    def test_get_records_with_limit(self):
        store = MemoryStorage()
        for i in range(10):
            store.append(_make_test_record(action=f"action_{i}"))
        records = store.get_records(limit=3)
        assert len(records) == 3

    def test_get_records_with_offset(self):
        store = MemoryStorage()
        for i in range(10):
            store.append(_make_test_record(action=f"action_{i}"))
        records = store.get_records(limit=3, offset=2)
        assert len(records) == 3
        # Most recent first, skip 2 → action_7
        assert records[0].action == "action_7"

    def test_get_records_filter_by_level(self):
        store = MemoryStorage()
        store.append(_make_test_record(level=LogLevel.INFO))
        store.append(_make_test_record(level=LogLevel.WARN))
        store.append(_make_test_record(level=LogLevel.ERROR))
        store.append(_make_test_record(level=LogLevel.DEBUG))
        records = store.get_records(level=LogLevel.WARN)
        assert len(records) == 2  # WARN + ERROR
        assert all(r.level >= LogLevel.WARN for r in records)

    def test_get_records_filter_by_category(self):
        store = MemoryStorage()
        store.append(_make_test_record(category="llm"))
        store.append(_make_test_record(category="tool"))
        store.append(_make_test_record(category="llm"))
        records = store.get_records(category="llm")
        assert len(records) == 2

    def test_clear(self):
        store = MemoryStorage()
        store.append(_make_test_record())
        store.append(_make_test_record())
        store.clear()
        assert store.count() == 0

    def test_get_all_chronological(self):
        store = MemoryStorage()
        store.append(_make_test_record(action="first"))
        store.append(_make_test_record(action="second"))
        records = store.get_all()
        assert records[0].action == "first"
        assert records[1].action == "second"

    def test_thread_safety(self):
        store = MemoryStorage(maxlen=10000)
        errors = []

        def writer(start: int):
            try:
                for i in range(100):
                    store.append(_make_test_record(action=f"thread_{start}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 1000


# ============================================================
# JSONLStorage Tests
# ============================================================

class TestJSONLStorage:
    def _make_storage(self, tmp_path: str) -> JSONLStorage:
        path = os.path.join(tmp_path, "test_audit.jsonl")
        return JSONLStorage(path=path)

    def test_append_and_count(self, tmp_path):
        store = self._make_storage(str(tmp_path))
        assert store.count() == 0
        store.append(_make_test_record())
        assert store.count() == 1
        store.append(_make_test_record())
        assert store.count() == 2
        store.close()

    def test_persistence_across_instances(self, tmp_path):
        path = os.path.join(str(tmp_path), "persist.jsonl")
        store1 = JSONLStorage(path=path)
        store1.append(_make_test_record(action="persisted"))
        store1.close()

        store2 = JSONLStorage(path=path)
        assert store2.count() == 1
        records = store2.get_records()
        assert records[0].action == "persisted"
        store2.close()

    def test_get_records_most_recent_first(self, tmp_path):
        store = self._make_storage(str(tmp_path))
        store.append(_make_test_record(action="first"))
        store.append(_make_test_record(action="second"))
        records = store.get_records()
        assert records[0].action == "second"
        assert records[1].action == "first"
        store.close()

    def test_get_records_filter_by_level(self, tmp_path):
        store = self._make_storage(str(tmp_path))
        store.append(_make_test_record(level=LogLevel.INFO))
        store.append(_make_test_record(level=LogLevel.ERROR))
        records = store.get_records(level=LogLevel.ERROR)
        assert len(records) == 1
        assert records[0].level == LogLevel.ERROR
        store.close()

    def test_clear(self, tmp_path):
        store = self._make_storage(str(tmp_path))
        store.append(_make_test_record())
        store.append(_make_test_record())
        store.clear()
        assert store.count() == 0
        store.close()

    def test_malformed_lines_skipped(self, tmp_path):
        path = os.path.join(str(tmp_path), "malformed.jsonl")
        # Write some valid and invalid lines
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"id":"1","timestamp":1,"level":"INFO","event_type":"custom","category":"t","action":"a"}\n')
            f.write('this is not json\n')
            f.write('{"id":"2","timestamp":2,"level":"ERROR","event_type":"error","category":"t","action":"b"}\n')

        store = JSONLStorage(path=path)
        records = store.get_records(limit=100)
        assert len(records) == 2
        store.close()

    def test_thread_safety(self, tmp_path):
        store = self._make_storage(str(tmp_path))
        errors = []

        def writer(start: int):
            try:
                for i in range(50):
                    store.append(_make_test_record(action=f"thread_{start}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 500
        store.close()
