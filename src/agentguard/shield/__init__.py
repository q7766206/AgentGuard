# -*- coding: utf-8 -*-
"""
AgentGuard Shield — Active Defense Modules

Shield modules protect agents from external attacks, not just self-inflicted errors.
Each module is a rule that plugs into the existing RuleEngine.

Modules:
    1. PromptInjectionDetector — Detect prompt injection in inputs/outputs
    2. DataLeakageDetector    — Catch API keys, passwords, PII in agent output
    3. BehaviorAnomalyDetector — Detect sudden behavioral shifts (possible hijack)
    4. ExfilDetector           — Catch data exfiltration attempts
"""

from agentguard.shield.injection import PromptInjectionDetector
from agentguard.shield.leakage import DataLeakageDetector
from agentguard.shield.anomaly import BehaviorAnomalyDetector
from agentguard.shield.exfil import ExfilDetector

__all__ = [
    "PromptInjectionDetector",
    "DataLeakageDetector",
    "BehaviorAnomalyDetector",
    "ExfilDetector",
]
