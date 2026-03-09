# -*- coding: utf-8 -*-
"""
AgentGuard Built-in Rules

Production-tested rules extracted from real agent monitoring.
Each rule is self-contained, stateful, and thread-safe.

Rules:
    1. LoopDetection    — Same tool called N times in M seconds
    2. TokenBudget      — Total tokens exceed threshold
    3. ErrorCascade     — N errors in M seconds → circuit breaker
    4. SensitiveOp      — Dangerous tool/command patterns blocked
    5. TimeoutGuard     — Single operation exceeds time limit
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Pattern, Sequence, Set, Tuple

from agentguard.rules.base import BaseRule
from agentguard.types import (
    AuditRecord,
    EventType,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
)


# ============================================================
# 1. Loop Detection
# ============================================================

class LoopDetection(BaseRule):
    """
    Detect when an agent is stuck in a loop calling the same tool repeatedly.

    Tracks tool call history within a sliding time window. If the same tool
    (or the same tool+args combination) appears more than ``max_repeats`` times
    within ``window_seconds``, a violation is fired.

    Args:
        max_repeats: Maximum allowed repetitions before triggering. Default 5.
        window_seconds: Time window to track repetitions. Default 60.
        track_args: If True, also compare tool arguments (stricter). Default False.
        action: What to do on violation. Default BLOCK.

    Example::

        rule = LoopDetection(max_repeats=3, window_seconds=30)
        # Fires if the same tool is called 3+ times in 30 seconds
    """

    def __init__(
        self,
        max_repeats: int = 5,
        window_seconds: float = 60.0,
        track_args: bool = False,
        action: RuleAction = RuleAction.BLOCK,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="LoopDetection",
            severity=RuleSeverity.HIGH,
            action=action,
            **kwargs,
        )
        self._max_repeats = max_repeats
        self._window_seconds = window_seconds
        self._track_args = track_args
        # History: deque of (timestamp, tool_key)
        self._history: Deque[Tuple[float, str]] = deque()
        self._lock = threading.Lock()

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        # Only check tool call events
        if record.event_type != EventType.TOOL_CALL_START:
            return None

        tool_name = record.action
        if self._track_args:
            # Include args hash in the key for stricter matching
            args_str = str(record.metadata.get("args", ""))
            tool_key = f"{tool_name}:{args_str[:200]}"
        else:
            tool_key = tool_name

        now = record.timestamp
        with self._lock:
            # Evict expired entries
            while self._history and (now - self._history[0][0]) > self._window_seconds:
                self._history.popleft()

            # Add current call
            self._history.append((now, tool_key))

            # Count occurrences of this tool_key
            count = sum(1 for _, k in self._history if k == tool_key)

        if count >= self._max_repeats:
            return RuleViolation(
                rule_name=self.name,
                severity=self.severity,
                action=self.action,
                message=(
                    f"Loop detected: '{tool_name}' called {count} times "
                    f"in {self._window_seconds}s (limit: {self._max_repeats})"
                ),
                detail=f"tool_key={tool_key}",
                metadata={
                    "tool_name": tool_name,
                    "count": count,
                    "max_repeats": self._max_repeats,
                    "window_seconds": self._window_seconds,
                },
            )
        return None

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


# ============================================================
# 2. Token Budget
# ============================================================

class TokenBudget(BaseRule):
    """
    Enforce a maximum token budget for the session.

    Checks the ``total_tokens`` field in RuleContext against the configured limit.
    Can also enforce per-call limits by inspecting record metadata.

    Args:
        max_tokens: Maximum total tokens allowed per session. Default 100000.
        max_tokens_per_call: Maximum tokens per single LLM call. Default 0 (no limit).
        action: What to do on violation. Default KILL (stop the agent).

    Example::

        rule = TokenBudget(max_tokens=50000)
        # Kills the agent if total tokens exceed 50k
    """

    def __init__(
        self,
        max_tokens: int = 100_000,
        max_tokens_per_call: int = 0,
        action: RuleAction = RuleAction.KILL,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="TokenBudget",
            severity=RuleSeverity.CRITICAL,
            action=action,
            **kwargs,
        )
        self._max_tokens = max_tokens
        self._max_per_call = max_tokens_per_call

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        # Check per-call limit
        if self._max_per_call > 0 and record.event_type == EventType.LLM_CALL_END:
            call_tokens = record.metadata.get("total_tokens", 0)
            if call_tokens > self._max_per_call:
                return RuleViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    action=RuleAction.BLOCK,
                    message=(
                        f"Single LLM call used {call_tokens} tokens "
                        f"(limit: {self._max_per_call})"
                    ),
                    metadata={
                        "call_tokens": call_tokens,
                        "max_per_call": self._max_per_call,
                    },
                )

        # Check session total
        if context.total_tokens >= self._max_tokens:
            pct = round(context.total_tokens / self._max_tokens * 100, 1)
            return RuleViolation(
                rule_name=self.name,
                severity=self.severity,
                action=self.action,
                message=(
                    f"Token budget exceeded: {context.total_tokens:,} / "
                    f"{self._max_tokens:,} ({pct}%)"
                ),
                metadata={
                    "total_tokens": context.total_tokens,
                    "max_tokens": self._max_tokens,
                    "percentage": pct,
                },
            )
        return None


# ============================================================
# 3. Error Cascade
# ============================================================

class ErrorCascade(BaseRule):
    """
    Detect cascading errors and trigger a circuit breaker.

    If ``max_errors`` errors occur within ``window_seconds``, the rule fires.
    This prevents the agent from burning tokens retrying a fundamentally broken operation.

    Args:
        max_errors: Maximum errors before triggering. Default 5.
        window_seconds: Time window to count errors. Default 60.
        action: What to do on violation. Default KILL.

    Example::

        rule = ErrorCascade(max_errors=3, window_seconds=30)
        # Kills the agent after 3 errors in 30 seconds
    """

    def __init__(
        self,
        max_errors: int = 5,
        window_seconds: float = 60.0,
        action: RuleAction = RuleAction.KILL,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="ErrorCascade",
            severity=RuleSeverity.CRITICAL,
            action=action,
            **kwargs,
        )
        self._max_errors = max_errors
        self._window_seconds = window_seconds
        self._error_times: Deque[float] = deque()
        self._lock = threading.Lock()

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        # Only track error events
        if record.event_type != EventType.ERROR and record.level < 40:  # ERROR = 40
            return None

        now = record.timestamp
        with self._lock:
            # Evict expired
            while self._error_times and (now - self._error_times[0]) > self._window_seconds:
                self._error_times.popleft()

            self._error_times.append(now)
            error_count = len(self._error_times)

        if error_count >= self._max_errors:
            return RuleViolation(
                rule_name=self.name,
                severity=self.severity,
                action=self.action,
                message=(
                    f"Error cascade: {error_count} errors in "
                    f"{self._window_seconds}s (limit: {self._max_errors})"
                ),
                detail=f"Recent errors: {context.recent_errors[:5]}",
                metadata={
                    "error_count": error_count,
                    "max_errors": self._max_errors,
                    "window_seconds": self._window_seconds,
                },
            )
        return None

    def reset(self) -> None:
        with self._lock:
            self._error_times.clear()


# ============================================================
# 4. Sensitive Operation
# ============================================================

class SensitiveOp(BaseRule):
    """
    Block dangerous tool calls and command patterns.

    Maintains a list of regex patterns that match dangerous operations.
    When a tool call matches any pattern, it is blocked immediately.

    Default blocked patterns include:
        - rm -rf, del /f, format (destructive file operations)
        - DROP TABLE, DELETE FROM (destructive database operations)
        - sudo, runas (privilege escalation)
        - curl | bash, wget | sh (remote code execution)
        - /etc/passwd, /etc/shadow (sensitive file access)

    Args:
        extra_patterns: Additional regex patterns to block.
        blocked_tools: Tool names to block entirely (e.g., ["shell", "exec"]).
        action: What to do on violation. Default BLOCK.

    Example::

        rule = SensitiveOp(
            extra_patterns=[r"api[_-]?key", r"password"],
            blocked_tools=["dangerous_tool"],
        )
    """

    # Default dangerous patterns
    DEFAULT_PATTERNS: List[str] = [
        # Destructive file operations
        r"rm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r){1,2}",
        r"del\s+/[fFsS]",
        r"format\s+[a-zA-Z]:",
        r"rmdir\s+/[sS]",
        r"shutil\.rmtree",
        # Database destruction
        r"DROP\s+(TABLE|DATABASE|SCHEMA)",
        r"DELETE\s+FROM\s+\w+\s*(;|$|WHERE\s+1\s*=\s*1)",
        r"TRUNCATE\s+TABLE",
        # Privilege escalation
        r"\bsudo\b",
        r"\brunas\b",
        r"chmod\s+[0-7]*777",
        # Remote code execution
        r"curl\s+.*\|\s*(ba)?sh",
        r"wget\s+.*\|\s*(ba)?sh",
        r"eval\s*\(",
        r"exec\s*\(",
        # Sensitive file access
        r"/etc/(passwd|shadow|sudoers)",
        r"\.ssh/(id_rsa|authorized_keys)",
        r"\.(env|pem|key|p12|pfx)\b",
        # Exfiltration patterns
        r"base64\s+.*\|\s*curl",
        r"nc\s+-[a-zA-Z]*l",  # netcat listener
    ]

    def __init__(
        self,
        extra_patterns: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        action: RuleAction = RuleAction.BLOCK,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="SensitiveOp",
            severity=RuleSeverity.CRITICAL,
            action=action,
            **kwargs,
        )
        all_patterns = self.DEFAULT_PATTERNS + (extra_patterns or [])
        self._compiled: List[Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in all_patterns
        ]
        self._pattern_sources = all_patterns
        self._blocked_tools: Set[str] = set(blocked_tools or [])

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        if record.event_type != EventType.TOOL_CALL_START:
            return None

        tool_name = record.action

        # Check blocked tools
        if tool_name in self._blocked_tools:
            return RuleViolation(
                rule_name=self.name,
                severity=self.severity,
                action=self.action,
                message=f"Blocked tool: '{tool_name}' is not allowed",
                metadata={"tool_name": tool_name, "reason": "blocked_tool"},
            )

        # Check patterns against tool args and detail
        text_to_check = " ".join([
            record.detail,
            str(record.metadata.get("args", "")),
            str(record.metadata.get("command", "")),
            str(record.metadata.get("input", "")),
        ])

        for i, pattern in enumerate(self._compiled):
            match = pattern.search(text_to_check)
            if match:
                return RuleViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    action=self.action,
                    message=(
                        f"Sensitive operation detected in '{tool_name}': "
                        f"matched pattern '{self._pattern_sources[i]}'"
                    ),
                    detail=f"Matched: '{match.group()}'",
                    metadata={
                        "tool_name": tool_name,
                        "pattern": self._pattern_sources[i],
                        "matched_text": match.group(),
                        "reason": "pattern_match",
                    },
                )
        return None


# ============================================================
# 5. Timeout Guard
# ============================================================

class TimeoutGuard(BaseRule):
    """
    Flag operations that exceed a time limit.

    Checks the ``duration_ms`` field of completed events (tool_call_end, llm_call_end).
    If an operation took longer than the threshold, a violation is fired.

    Args:
        max_duration_ms: Maximum allowed duration in milliseconds. Default 30000 (30s).
        action: What to do on violation. Default LOG (just warn, don't block).

    Example::

        rule = TimeoutGuard(max_duration_ms=10000)
        # Warns if any operation takes longer than 10 seconds
    """

    def __init__(
        self,
        max_duration_ms: int = 30_000,
        action: RuleAction = RuleAction.LOG,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="TimeoutGuard",
            severity=RuleSeverity.MEDIUM,
            action=action,
            **kwargs,
        )
        self._max_duration_ms = max_duration_ms

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        # Only check completed events that have duration
        if record.event_type not in (EventType.TOOL_CALL_END, EventType.LLM_CALL_END):
            return None

        if record.duration_ms <= 0:
            return None

        if record.duration_ms > self._max_duration_ms:
            return RuleViolation(
                rule_name=self.name,
                severity=self.severity,
                action=self.action,
                message=(
                    f"Operation '{record.action}' took {record.duration_ms}ms "
                    f"(limit: {self._max_duration_ms}ms)"
                ),
                metadata={
                    "action": record.action,
                    "duration_ms": record.duration_ms,
                    "max_duration_ms": self._max_duration_ms,
                },
            )
        return None
