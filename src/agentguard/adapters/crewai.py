# -*- coding: utf-8 -*-
"""
AgentGuard CrewAI Adapter

Integrates AgentGuard with CrewAI's callback system.
Provides step_callback and task_callback wrappers.

Usage::

    from agentguard import AgentGuard
    from agentguard.rules import LoopDetection, SensitiveOp
    from agentguard.adapters.crewai import make_crewai_callbacks

    guard = AgentGuard(rules=[LoopDetection(), SensitiveOp()])
    step_cb, task_cb = make_crewai_callbacks(guard)

    agent = Agent(
        role="researcher",
        step_callback=step_cb,
        ...
    )
    task = Task(
        description="...",
        callback=task_cb,
        ...
    )

Requires: crewai >= 0.80
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

from agentguard.core import AgentGuard
from agentguard.types import EventType, LogLevel


def make_crewai_callbacks(
    guard: AgentGuard,
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Create CrewAI-compatible callbacks from an AgentGuard instance.

    Returns:
        Tuple of (step_callback, task_callback).

    - ``step_callback`` is called after each agent step (tool call or thought).
    - ``task_callback`` is called when a task completes.
    """

    def step_callback(step_output: Any) -> None:
        """
        Called after each CrewAI agent step.

        Inspects the step output to determine if it was a tool call or thought,
        then records it through AgentGuard.
        """
        try:
            # CrewAI step output varies by version
            # Common patterns: step_output.tool, step_output.tool_input, step_output.log
            tool_name = ""
            tool_input = {}
            log_text = ""

            if hasattr(step_output, "tool"):
                tool_name = str(step_output.tool)
            if hasattr(step_output, "tool_input"):
                raw_input = step_output.tool_input
                if isinstance(raw_input, dict):
                    tool_input = raw_input
                elif isinstance(raw_input, str):
                    tool_input = {"input": raw_input}
            if hasattr(step_output, "log"):
                log_text = str(step_output.log)[:500]
            if hasattr(step_output, "result"):
                log_text = log_text or str(step_output.result)[:500]

            if tool_name:
                # This is a tool call step
                guard.before_tool_call(tool_name, tool_input)
                guard.after_tool_call(tool_name, result=log_text)
            else:
                # This is a thought/reasoning step
                guard.trail.record(
                    level=LogLevel.DEBUG,
                    event_type=EventType.CUSTOM,
                    category="crewai",
                    action="agent_thought",
                    detail=log_text,
                )

        except Exception:
            # Never crash CrewAI because of our callback
            pass

    def task_callback(task_output: Any) -> None:
        """
        Called when a CrewAI task completes.

        Records the task completion and summary stats.
        """
        try:
            description = ""
            result = ""

            if hasattr(task_output, "description"):
                description = str(task_output.description)[:200]
            if hasattr(task_output, "raw"):
                result = str(task_output.raw)[:500]
            elif hasattr(task_output, "result"):
                result = str(task_output.result)[:500]

            guard.trail.record(
                level=LogLevel.INFO,
                event_type=EventType.AGENT_END,
                category="crewai",
                action="task_complete",
                detail=f"Task completed: {description}",
                metadata={
                    "result_length": len(result),
                    "stats": guard.get_stats(),
                },
            )
        except Exception:
            pass

    return step_callback, task_callback
