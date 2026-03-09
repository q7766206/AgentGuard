# -*- coding: utf-8 -*-
"""
AgentGuard LangChain 1.0 Adapter

Integrates AgentGuard with LangChain 1.0's create_agent middleware system.
Provides both decorator-style hooks and a class-based middleware.

Usage::

    from agentguard import AgentGuard
    from agentguard.rules import LoopDetection, TokenBudget, SensitiveOp
    from agentguard.adapters.langchain import make_guard_middleware

    guard = AgentGuard(rules=[
        LoopDetection(max_repeats=5),
        TokenBudget(max_tokens=100000),
        SensitiveOp(),
    ])

    middleware = make_guard_middleware(guard)

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=[...],
        middleware=[*middleware],
    )

Requires: langchain >= 1.0
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from agentguard.core import AgentGuard
from agentguard.exceptions import AgentGuardBlock, AgentGuardKill
from agentguard.types import EventType, LogLevel

# Lazy imports — only fail when actually used, not at import time
_LANGCHAIN_AVAILABLE = False
try:
    from langchain.agents.middleware import (
        before_agent,
        before_model,
        after_model,
        after_agent,
        wrap_tool_call,
    )
    from langchain.agents.middleware import AgentState
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


def _check_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain >= 1.0 is required for this adapter. "
            "Install it with: pip install 'agentguard[langchain]'"
        )


def make_guard_middleware(guard: AgentGuard) -> List[Any]:
    """
    Create a list of LangChain 1.0 middleware hooks from an AgentGuard instance.

    Returns a list of decorated functions that can be passed directly to
    ``create_agent(middleware=[...])``.

    Args:
        guard: An AgentGuard instance with rules configured.

    Returns:
        List of middleware hooks: [before_agent_hook, before_model_hook,
        after_model_hook, wrap_tool_hook, after_agent_hook]

    Example::

        guard = AgentGuard(rules=[SensitiveOp(), TokenBudget()])
        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[*make_guard_middleware(guard)],
        )
    """
    _check_langchain()

    # Track timing
    _state: Dict[str, Any] = {"agent_start": 0.0, "model_start": 0.0}

    @before_agent
    def guard_before_agent(state: AgentState, runtime: Any = None) -> Optional[dict]:
        """Record agent start and initialize session."""
        _state["agent_start"] = time.time()
        guard.trail.record(
            level=LogLevel.INFO,
            event_type=EventType.AGENT_START,
            category="agent",
            action="agent_start",
            detail="Agent invocation started",
            metadata={
                "message_count": len(state.get("messages", [])),
            },
        )
        return None

    @before_model
    def guard_before_model(state: AgentState, runtime: Any = None) -> Optional[dict]:
        """Record LLM call start."""
        _state["model_start"] = time.time()
        messages = state.get("messages", [])
        guard.before_llm_call(
            model="langchain_agent",
            messages=[{"role": "system", "content": f"{len(messages)} messages"}],
        )
        return None

    @after_model
    def guard_after_model(state: AgentState, runtime: Any = None) -> Optional[dict]:
        """Record LLM call end with token usage."""
        duration_ms = int((time.time() - _state.get("model_start", time.time())) * 1000)
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        tokens = 0
        response_text = ""
        if last_msg:
            # Try to extract token usage from the message
            if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
                tokens = getattr(last_msg.usage_metadata, "total_tokens", 0)
            if hasattr(last_msg, "content"):
                response_text = str(last_msg.content)[:200]

        try:
            guard.after_llm_call(
                model="langchain_agent",
                response=response_text,
                tokens=tokens,
                duration_ms=duration_ms,
            )
        except (AgentGuardKill, AgentGuardBlock):
            # Re-raise — these should propagate to stop the agent
            raise
        except Exception:
            pass

        return None

    @wrap_tool_call
    def guard_wrap_tool(request: Any, handler: Callable) -> Any:
        """
        Wrap every tool call with AgentGuard checks.

        Checks rules BEFORE the tool executes (can block dangerous calls).
        Records the result AFTER the tool executes.
        """
        tool_name = ""
        tool_args = {}

        # Extract tool info from the request
        if hasattr(request, "tool_call"):
            tc = request.tool_call
            tool_name = getattr(tc, "name", "") or str(getattr(tc, "type", ""))
            tool_args = getattr(tc, "args", {}) or {}
        elif hasattr(request, "name"):
            tool_name = request.name
            tool_args = getattr(request, "args", {}) or {}

        # Check rules BEFORE execution
        start = time.time()
        try:
            guard.before_tool_call(tool_name, tool_args)
        except AgentGuardBlock:
            # Return a ToolMessage indicating the block
            # The agent will see this and can decide what to do
            raise
        except AgentGuardKill:
            raise

        # Execute the tool
        try:
            result = handler(request)
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            guard.after_tool_call(
                tool_name,
                error=str(e)[:500],
                duration_ms=duration_ms,
            )
            raise

        # Record success
        duration_ms = int((time.time() - start) * 1000)
        result_text = ""
        if hasattr(result, "content"):
            result_text = str(result.content)[:500]
        elif isinstance(result, str):
            result_text = result[:500]

        guard.after_tool_call(
            tool_name,
            result=result_text,
            duration_ms=duration_ms,
        )

        return result

    @after_agent
    def guard_after_agent(state: AgentState, runtime: Any = None) -> Optional[dict]:
        """Record agent completion with summary stats."""
        duration_ms = int((time.time() - _state.get("agent_start", time.time())) * 1000)
        stats = guard.get_stats()
        guard.trail.record(
            level=LogLevel.INFO,
            event_type=EventType.AGENT_END,
            category="agent",
            action="agent_end",
            detail="Agent invocation completed",
            duration_ms=duration_ms,
            metadata={
                "total_violations": stats["rules"]["total_violations"],
                "total_tokens": stats["rules"]["context"]["total_tokens"],
                "total_tool_calls": stats["rules"]["context"]["total_tool_calls"],
            },
        )
        return None

    return [
        guard_before_agent,
        guard_before_model,
        guard_after_model,
        guard_wrap_tool,
        guard_after_agent,
    ]
