# -*- coding: utf-8 -*-
"""
Custom Rule Example

Shows how to create your own rule by extending BaseRule.
"""

from agentguard import AgentGuard
from agentguard.rules import BaseRule, RuleEngine
from agentguard.types import (
    AuditRecord,
    EventType,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
)


class CostBudget(BaseRule):
    """Block when estimated API cost exceeds a dollar threshold."""

    # Rough pricing: $0.01 per 1000 tokens (adjust for your model)
    COST_PER_1K_TOKENS = 0.01

    def __init__(self, max_cost_usd: float = 5.0, **kwargs):
        super().__init__(
            name="CostBudget",
            severity=RuleSeverity.HIGH,
            action=RuleAction.KILL,
            **kwargs,
        )
        self._max_cost = max_cost_usd

    def evaluate(self, record: AuditRecord, context: RuleContext):
        estimated_cost = (context.total_tokens / 1000) * self.COST_PER_1K_TOKENS
        if estimated_cost >= self._max_cost:
            return RuleViolation(
                rule_name=self.name,
                severity=self.severity,
                action=self.action,
                message=f"Estimated cost ${estimated_cost:.2f} exceeds ${self._max_cost:.2f}",
                metadata={"estimated_cost": estimated_cost, "max_cost": self._max_cost},
            )
        return None


class ProfanityFilter(BaseRule):
    """Block tool calls that contain profanity in their output."""

    BAD_WORDS = {"badword1", "badword2"}  # Replace with actual list

    def __init__(self, **kwargs):
        super().__init__(
            name="ProfanityFilter",
            severity=RuleSeverity.MEDIUM,
            action=RuleAction.BLOCK,
            **kwargs,
        )

    def evaluate(self, record: AuditRecord, context: RuleContext):
        if record.event_type != EventType.TOOL_CALL_END:
            return None
        text = record.detail.lower()
        for word in self.BAD_WORDS:
            if word in text:
                return RuleViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    action=self.action,
                    message=f"Profanity detected in tool output: '{word}'",
                )
        return None


# Use custom rules
guard = AgentGuard(rules=[
    CostBudget(max_cost_usd=1.0),
    ProfanityFilter(),
])

# Simulate some usage
guard.after_llm_call(model="gpt-4o", tokens=50000, response="...")
print(f"Stats: {guard.get_stats()}")
print(f"Violations: {guard.get_violations()}")
guard.close()
