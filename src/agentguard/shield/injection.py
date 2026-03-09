# -*- coding: utf-8 -*-
"""
Prompt Injection Detector

Multi-layer detection for prompt injection attacks:
    Layer 1: Regex pattern matching (fast, catches known patterns)
    Layer 2: Heuristic scoring (medium, catches structural anomalies)
    Layer 3: Canary token verification (optional, catches indirect injection)

Scans both INPUTS (user messages, tool outputs fed back to LLM) and
OUTPUTS (LLM responses that may have been hijacked).

Zero external dependencies — pure Python stdlib.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

from agentguard.rules.base import BaseRule
from agentguard.types import (
    AuditRecord,
    EventType,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
)


class PromptInjectionDetector(BaseRule):
    """
    Detect prompt injection attempts in agent inputs and outputs.

    Uses a multi-layer approach:
        1. **Regex patterns** — Known injection phrases in 4 languages (EN/ZH/ES/FR)
        2. **Heuristic scoring** — Structural signals like role impersonation,
           instruction override attempts, delimiter abuse
        3. **Canary tokens** — Optional unique strings injected into system prompts;
           if they appear in output, the agent has been hijacked

    Args:
        sensitivity: Detection sensitivity. "low" = fewer false positives,
                     "high" = catches more but may flag legitimate content.
                     Default "medium".
        extra_patterns: Additional regex patterns to detect.
        canary_token: Optional canary string. If set, any output containing
                      this string triggers a CRITICAL alert.
        scan_tool_output: If True, also scan tool outputs that will be fed
                          back to the LLM. Default True.
        action: What to do on detection. Default BLOCK.

    Example::

        rule = PromptInjectionDetector(
            sensitivity="high",
            canary_token="AG_CANARY_x7k9m2",
        )
    """

    # ================================================================
    # Known injection patterns (multi-language)
    # ================================================================

    PATTERNS_EN: List[str] = [
        # Direct instruction override
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|directives?)",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
        r"forget\s+(everything|all|your)\s+(previous|prior|above)",
        r"override\s+(your|the|all)\s+(instructions?|rules?|system)",
        # Role hijacking
        r"you\s+are\s+now\s+(a|an|the)\s+",
        r"act\s+as\s+(a|an|if)\s+",
        r"pretend\s+(you\s+are|to\s+be)\s+",
        r"from\s+now\s+on\s+you\s+(are|will|should|must)",
        r"switch\s+to\s+(\w+\s+)?mode",
        r"enter\s+(developer|admin|debug|god|sudo|root)\s+mode",
        # System prompt extraction
        r"(show|reveal|display|print|output|repeat|tell)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?|directives?)",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
        r"(copy|paste|echo)\s+(your|the)\s+(system|initial)\s+(prompt|message)",
        # Delimiter/boundary attacks
        r"<\|?(system|endoftext|im_start|im_end)\|?>",
        r"\[INST\]|\[/INST\]",
        r"###\s*(system|instruction|new\s+task)",
        # Jailbreak patterns
        r"(DAN|DUDE|STAN|KEVIN)\s+(mode|prompt|jailbreak)",
        r"do\s+anything\s+now",
        r"(no|without)\s+(restrictions?|limitations?|filters?|censorship)",
        r"bypass\s+(safety|content|ethical)\s+(filters?|restrictions?|guidelines?)",
        # Indirect injection markers
        r"(IMPORTANT|URGENT|CRITICAL)\s*[:\-!]\s*(ignore|override|forget|disregard)",
        r"(new|updated|revised)\s+(instructions?|task|objective)\s*[:\-]",
        r"the\s+user\s+(wants|asked|said)\s+you\s+to\s+(ignore|forget|override)",
    ]

    PATTERNS_ZH: List[str] = [
        r"忽略(之前|上面|以前|先前|所有)(的)?(指令|提示|规则|指示|要求)",
        r"无视(之前|上面|以前|先前|所有)(的)?(指令|提示|规则)",
        r"你现在是(一个|一名)?",
        r"假装你是",
        r"从现在开始你(是|要|将|必须)",
        r"(显示|展示|输出|告诉我|重复)(你的)?(系统)?(提示|指令|规则)",
        r"进入(开发者|管理员|调试|上帝|超级)?模式",
        r"(取消|解除|绕过)(安全|内容|道德)(限制|过滤|审查)",
    ]

    PATTERNS_ES: List[str] = [
        r"ignora\s+(todas?\s+)?(las?\s+)?(instrucciones?|reglas?|indicaciones?)\s+(anteriores?|previas?)",
        r"ahora\s+eres\s+(un|una)\s+",
        r"(muestra|revela|repite)\s+(tu|el)\s+(prompt|instrucciones?)\s+(del\s+)?sistema",
    ]

    PATTERNS_FR: List[str] = [
        r"ignore[rz]?\s+(toutes?\s+)?(les?\s+)?(instructions?|r[eè]gles?)\s+(pr[eé]c[eé]dentes?|ant[eé]rieures?)",
        r"tu\s+es\s+maintenant\s+(un|une)\s+",
        r"(montre|r[eé]v[eè]le|affiche)\s+(ton|le)\s+(prompt|instructions?)\s+(syst[eè]me)?",
    ]

    # ================================================================
    # Heuristic signals
    # ================================================================

    HEURISTIC_SIGNALS: List[Tuple[str, float, str]] = [
        # (pattern, weight, description)
        (r"```\s*(system|instruction)", 0.6, "Code block with system/instruction label"),
        (r"={3,}\s*(new|system|instruction)", 0.5, "Separator with instruction keyword"),
        (r"-{3,}\s*(new|system|instruction)", 0.5, "Separator with instruction keyword"),
        (r"(BEGIN|START|END)\s+(SYSTEM|INSTRUCTION|PROMPT)", 0.7, "Explicit boundary marker"),
        (r"<(system|instruction|prompt)>", 0.8, "XML-style injection tag"),
        (r"\bAI\s*:\s*", 0.3, "AI role prefix in user input"),
        (r"\bassistant\s*:\s*", 0.4, "Assistant role prefix in user input"),
        (r"\bhuman\s*:\s*.*\bassistant\s*:\s*", 0.7, "Multi-turn injection"),
        (r"(tool_call|function_call)\s*[:\(]", 0.6, "Fake tool call in text"),
        (r"\\n\\n(system|instruction)", 0.4, "Escaped newline injection"),
    ]

    def __init__(
        self,
        sensitivity: str = "medium",
        extra_patterns: Optional[List[str]] = None,
        canary_token: Optional[str] = None,
        scan_tool_output: bool = True,
        action: RuleAction = RuleAction.BLOCK,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="PromptInjectionDetector",
            severity=RuleSeverity.CRITICAL,
            action=action,
            **kwargs,
        )
        self._sensitivity = sensitivity
        self._canary_token = canary_token
        self._scan_tool_output = scan_tool_output

        # Compile all patterns
        all_patterns = (
            self.PATTERNS_EN + self.PATTERNS_ZH +
            self.PATTERNS_ES + self.PATTERNS_FR +
            (extra_patterns or [])
        )
        self._compiled: List[Tuple[Pattern[str], str]] = [
            (re.compile(p, re.IGNORECASE | re.DOTALL), p)
            for p in all_patterns
        ]

        # Compile heuristic patterns
        self._heuristics: List[Tuple[Pattern[str], float, str]] = [
            (re.compile(p, re.IGNORECASE), w, d)
            for p, w, d in self.HEURISTIC_SIGNALS
        ]

        # Sensitivity thresholds
        self._thresholds = {
            "low": 0.8,
            "medium": 0.5,
            "high": 0.3,
        }
        self._threshold = self._thresholds.get(sensitivity, 0.5)

        # Stats
        self._detections = 0
        self._lock = threading.Lock()

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        # Determine what text to scan based on event type
        text = self._extract_scannable_text(record)
        if not text:
            return None

        # Layer 1: Canary token check (highest priority)
        if self._canary_token and self._canary_token in text:
            with self._lock:
                self._detections += 1
            return RuleViolation(
                rule_name=self.name,
                severity=RuleSeverity.CRITICAL,
                action=RuleAction.KILL,
                message="CANARY TOKEN DETECTED — Agent has been hijacked!",
                detail=f"Canary token '{self._canary_token[:8]}...' found in output",
                metadata={
                    "detection_type": "canary_token",
                    "event_type": record.event_type.value,
                },
            )

        # Layer 2: Regex pattern matching
        for compiled, source in self._compiled:
            match = compiled.search(text)
            if match:
                with self._lock:
                    self._detections += 1
                return RuleViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    action=self.action,
                    message=f"Prompt injection detected: matched '{source[:60]}'",
                    detail=f"Matched text: '{match.group()[:100]}'",
                    metadata={
                        "detection_type": "regex",
                        "pattern": source,
                        "matched_text": match.group()[:200],
                        "event_type": record.event_type.value,
                    },
                )

        # Layer 3: Heuristic scoring
        score = 0.0
        triggered_heuristics: List[str] = []
        for compiled_h, weight, desc in self._heuristics:
            if compiled_h.search(text):
                score += weight
                triggered_heuristics.append(desc)

        if score >= self._threshold:
            with self._lock:
                self._detections += 1
            return RuleViolation(
                rule_name=self.name,
                severity=RuleSeverity.HIGH,
                action=self.action,
                message=f"Prompt injection suspected (heuristic score: {score:.2f}/{self._threshold:.2f})",
                detail=f"Triggered: {', '.join(triggered_heuristics)}",
                metadata={
                    "detection_type": "heuristic",
                    "score": score,
                    "threshold": self._threshold,
                    "triggered": triggered_heuristics,
                    "event_type": record.event_type.value,
                },
            )

        return None

    def _extract_scannable_text(self, record: AuditRecord) -> str:
        """Extract text to scan from the audit record."""
        parts: List[str] = []

        # Scan LLM inputs (messages going to the model)
        if record.event_type == EventType.LLM_CALL_START:
            parts.append(record.detail)
            msgs = record.metadata.get("messages", [])
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        parts.append(str(m.get("content", "")))

        # Scan LLM outputs (responses that may have been hijacked)
        if record.event_type == EventType.LLM_CALL_END:
            parts.append(record.detail)

        # Scan tool outputs (indirect injection via tool results)
        if self._scan_tool_output and record.event_type == EventType.TOOL_CALL_END:
            parts.append(record.detail)
            parts.append(str(record.metadata.get("result", "")))

        # Scan tool inputs (user might inject via tool args)
        if record.event_type == EventType.TOOL_CALL_START:
            parts.append(str(record.metadata.get("args", "")))
            parts.append(str(record.metadata.get("input", "")))
            parts.append(record.detail)

        return " ".join(p for p in parts if p)

    @property
    def detections(self) -> int:
        """Total number of injections detected."""
        with self._lock:
            return self._detections

    def reset(self) -> None:
        with self._lock:
            self._detections = 0
