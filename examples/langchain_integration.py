# -*- coding: utf-8 -*-
"""
AgentGuard + LangChain 1.0 Integration Example

Shows how to add security middleware to a LangChain agent.
Requires: pip install agentguard[langchain] langchain-openai
"""

from agentguard import AgentGuard
from agentguard.rules import LoopDetection, TokenBudget, SensitiveOp
from agentguard.adapters.langchain import make_guard_middleware

# 1. Create AgentGuard with rules
guard = AgentGuard(
    rules=[
        LoopDetection(max_repeats=5),
        TokenBudget(max_tokens=100_000),
        SensitiveOp(),
    ],
    persist=True,
)

# 2. Generate middleware hooks
middleware = make_guard_middleware(guard)

# 3. Create LangChain agent with security middleware
# from langchain_openai import ChatOpenAI
# from langchain.agents import create_agent
# from langchain.tools import tool
#
# @tool
# def web_search(query: str) -> str:
#     """Search the web."""
#     return f"Results for: {query}"
#
# agent = create_agent(
#     model=ChatOpenAI(model="gpt-4o"),
#     tools=[web_search],
#     middleware=[*middleware],  # ← AgentGuard hooks injected here
# )
#
# # 4. Run the agent — AgentGuard monitors every step
# result = agent.invoke({"messages": [{"role": "user", "content": "Research AI safety"}]})
#
# # 5. Check what happened
# print(guard.get_stats())
# print(guard.export_json())

print("LangChain integration example (uncomment code above to run with real LLM)")
print(f"Guard created with {len(guard.engine.rules)} rules")
print(f"Middleware hooks: {len(middleware)}")
