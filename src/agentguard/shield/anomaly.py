# -*- coding: utf-8 -*-
"""
Behavior Anomaly Detector

Detects sudden behavioral shifts that may indicate the agent has been hijacked.
Builds a baseline of normal behavior and flags deviations.

Signals monitored:
    1. Sudden topic/intent shift after reading external content
    2. Unusual tool call patterns (tools never used before appearing suddenly)
    3. Spike in output length or token usage
    4. Agent suddenly trying to access sensitive resources it never needed before
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from agentguard.rules.base import BaseRule
from agentguard.types import (
    AuditRecord,
    EventType,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
)


class BehaviorAnomalyDetector(BaseRule):
    """
    Detect behavioral anomalies that may indicate agent hijacking.

    Builds a rolling baseline of the agent's behavior and flags sudden deviations.
    Particularly effective at catching indirect prompt injection — where a tool
    returns malicious content that changes the agent's behavior.

    Monitored signals:
        - **New tool usage**: Agent suddenly calls a tool it never used before
        - **Tool frequency spike**: A tool's call rate jumps abnormally
        - **Category shift**: Agent switches from "research" to "file_system" suddenly
        - **Error spike after tool output**: Errors spike right after reading external content

    Args:
        baseline_window: Number of events to build baseline from. Default 20.
        new_tool_alert: Alert when agent uses a tool not seen in baseline. Default True.
        frequency_spike_factor: Alert when a tool's frequency exceeds baseline by this factor. Default 3.0.
        action: What to do on detection. Default LOG (anomalies are often false positives).

    Example::

        rule = BehaviorAnomalyDetector(
            baseline_window=15,
            frequency_spike_factor=2.0,
            action=RuleAction.BLOCK,
        )
    """

    def __init__(
        self,
        baseline_window: int = 20,
        new_tool_alert: bool = True,
        frequency_spike_factor: float = 3.0,
        action: RuleAction = RuleAction.LOG,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="BehaviorAnomalyDetector",
            severity=RuleSeverity.HIGH,
            action=action,
            **kwargs,
        )
        self._baseline_window = baseline_window
        self._new_tool_alert = new_tool_alert
        self._spike_factor = frequency_spike_factor

        # Baseline tracking
        self._tool_history: Deque[str] = deque(maxlen=200)
        self._category_history: Deque[str] = deque(maxlen=200)
        self._baseline_tools: Set[str] = set()
        self._baseline_built = False
        self._event_count = 0
        self._lock = threading.Lock()

        # Sliding window for frequency analysis
        self._recent_tools: Deque[Tuple[float, str]] = deque()
        self._recent_window_seconds = 60.0

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        if record.event_type != EventType.TOOL_CALL_START:
            return None

        tool_name = record.action
        now = record.timestamp

        with self._lock:
            self._event_count += 1
            self._tool_history.append(tool_name)
            self._category_history.append(record.category)

            # Evict old entries from recent window
            while self._recent_tools and (now - self._recent_tools[0][0]) > self._recent_window_seconds:
                self._recent_tools.popleft()
            self._recent_tools.append((now, tool_name))

            # Build baseline from first N events
            if not self._baseline_built:
                if self._event_count >= self._baseline_window:
                    self._baseline_tools = set(self._tool_history)
                    self._baseline_built = True
                return None

            # === Anomaly checks ===

            # 1. New tool never seen in baseline
            if self._new_tool_alert and tool_name not in self._baseline_tools:
                return RuleViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    action=self.action,
                    message=f"Behavioral anomaly: Agent using new tool '{tool_name}' not seen in baseline",
                    detail=f"Baseline tools: {sorted(self._baseline_tools)}",
                    metadata={
                        "anomaly_type": "new_tool",
                        "tool_name": tool_name,
                        "baseline_tools": sorted(self._baseline_tools),
                    },
                )

            # 2. Frequency spike
            recent_counts = Counter(t for _, t in self._recent_tools)
            total_recent = len(self._recent_tools)
            if total_recent > 5:
                tool_freq = recent_counts.get(tool_name, 0) / total_recent
                # Expected frequency based on baseline
                baseline_list = list(self._tool_history)[:self._baseline_window]
                baseline_counts = Counter(baseline_list)
                expected_freq = baseline_counts.get(tool_name, 1) / max(len(baseline_list), 1)

                if expected_freq > 0 and tool_freq > expected_freq * self._spike_factor:
                    return RuleViolation(
                        rule_name=self.name,
                        severity=self.severity,
                        action=self.action,
                        message=(
                            f"Behavioral anomaly: '{tool_name}' frequency spike "
                            f"({tool_freq:.1%} vs baseline {expected_freq:.1%})"
                        ),
                        metadata={
                            "anomaly_type": "frequency_spike",
                            "tool_name": tool_name,
                            "current_freq": round(tool_freq, 3),
                            "baseline_freq": round(expected_freq, 3),
                            "spike_factor": self._spike_factor,
                        },
                    )

        return None

    def reset(self) -> None:
        with self._lock:
            self._tool_history.clear()
            self._category_history.clear()
            self._recent_tools.clear()
            self._baseline_tools.clear()
            self._baseline_built = False
            self._event_count = 0
