# -*- coding: utf-8 -*-
"""Performance benchmark for AgentGuard audit module."""

import time
import threading
import sys
sys.path.insert(0, r"D:\AgentGuard\src")

from agentguard.audit import AuditTrail
from agentguard.types import EventType, LogLevel


def bench_memory_only():
    """Benchmark: 10000 records to memory-only storage."""
    trail = AuditTrail(memory_maxlen=20000)
    start = time.perf_counter()
    for i in range(10000):
        trail.record(
            level=LogLevel.INFO,
            event_type=EventType.TOOL_CALL_START,
            category="tool",
            action=f"action_{i}",
            detail="benchmark detail",
            metadata={"index": i},
        )
    elapsed = time.perf_counter() - start
    per_record_us = (elapsed / 10000) * 1_000_000
    print(f"Memory-only: 10000 records in {elapsed:.3f}s ({per_record_us:.1f}us/record)")
    assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s (target < 1.0s)"


def bench_with_disk(tmp_dir: str):
    """Benchmark: 10000 records with JSONL persistence."""
    import os
    path = os.path.join(tmp_dir, "bench_audit.jsonl")
    trail = AuditTrail(persist=True, persist_path=path)
    start = time.perf_counter()
    for i in range(10000):
        trail.record(
            level=LogLevel.INFO,
            event_type=EventType.TOOL_CALL_START,
            category="tool",
            action=f"action_{i}",
            detail="benchmark detail",
            metadata={"index": i},
        )
    elapsed = time.perf_counter() - start
    per_record_us = (elapsed / 10000) * 1_000_000
    trail.close()
    file_size = os.path.getsize(path) / 1024
    print(f"With disk:   10000 records in {elapsed:.3f}s ({per_record_us:.1f}us/record, file={file_size:.0f}KB)")
    assert elapsed < 5.0, f"Too slow: {elapsed:.3f}s (target < 5.0s)"


def bench_with_subscriber():
    """Benchmark: 10000 records with 1 subscriber."""
    trail = AuditTrail()
    count = [0]
    trail.subscribe(lambda rec: count.__setitem__(0, count[0] + 1))
    start = time.perf_counter()
    for i in range(10000):
        trail.record(
            level=LogLevel.INFO,
            event_type=EventType.TOOL_CALL_START,
            category="tool",
            action=f"action_{i}",
        )
    elapsed = time.perf_counter() - start
    per_record_us = (elapsed / 10000) * 1_000_000
    print(f"With sub:    10000 records in {elapsed:.3f}s ({per_record_us:.1f}us/record, notified={count[0]})")
    assert count[0] == 10000


def bench_concurrent():
    """Benchmark: 10 threads x 1000 records each."""
    trail = AuditTrail(memory_maxlen=20000)
    start = time.perf_counter()
    errors = []

    def writer(tid):
        try:
            for i in range(1000):
                trail.record(
                    level=LogLevel.INFO,
                    event_type=EventType.CUSTOM,
                    category="test",
                    action=f"t{tid}_{i}",
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start
    per_record_us = (elapsed / 10000) * 1_000_000
    print(f"Concurrent:  10x1000 records in {elapsed:.3f}s ({per_record_us:.1f}us/record, errors={len(errors)})")
    assert len(errors) == 0
    assert trail.total_records == 10000


if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("AgentGuard Audit Performance Benchmark")
    print("=" * 60)
    bench_memory_only()
    with tempfile.TemporaryDirectory() as tmp:
        bench_with_disk(tmp)
    bench_with_subscriber()
    bench_concurrent()
    print("=" * 60)
    print("ALL BENCHMARKS PASSED")
    print("=" * 60)
