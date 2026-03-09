# -*- coding: utf-8 -*-
"""
AgentGuard Quickstart Example

Demonstrates the core API in 30 lines. No external dependencies needed.
"""

from agentguard import AgentGuard, AgentGuardBlock, AgentGuardKill
from agentguard.rules import LoopDetection, TokenBudget, SensitiveOp, ErrorCascade

# 1. Create a guard with rules
guard = AgentGuard(
    rules=[
        LoopDetection(max_repeats=3),       # Block if same tool called 3+ times
        TokenBudget(max_tokens=50_000),      # Kill if tokens exceed 50k
        SensitiveOp(),                       # Block dangerous commands
        ErrorCascade(max_errors=5),          # Kill after 5 errors in 60s
    ],
    persist=True,                            # Save audit trail to disk
    persist_path="./audit.jsonl",
)

# 2. Simulate an agent loop
print("=== Safe tool call ===")
guard.before_tool_call("web_search", {"query": "AI safety research"})
guard.after_tool_call("web_search", result="Found 10 papers", duration_ms=350)
print("  OK: web_search completed")

print("\n=== LLM call ===")
guard.before_llm_call(model="gpt-4o")
guard.after_llm_call(model="gpt-4o", response="Here are the results...", tokens=1200)
print("  OK: LLM call completed")

print("\n=== Dangerous command (should be blocked) ===")
try:
    guard.before_tool_call("shell", {"command": "rm -rf /"})
    print("  ERROR: Should have been blocked!")
except AgentGuardBlock as e:
    print(f"  BLOCKED: {e}")

print("\n=== Loop detection ===")
try:
    for i in range(5):
        guard.before_tool_call("web_search", {"query": f"query_{i}"})
except AgentGuardBlock as e:
    print(f"  BLOCKED: {e}")

# 3. Check stats
print("\n=== Session Stats ===")
stats = guard.get_stats()
print(f"  Total audit records: {stats['audit']['total_records']}")
print(f"  Total violations: {stats['rules']['total_violations']}")
print(f"  Total tokens used: {stats['rules']['context']['total_tokens']}")

# 4. Export audit trail
print(f"\n=== Audit trail saved to ./audit.jsonl ===")
guard.close()
print("Done!")
