# -*- coding: utf-8 -*-
"""
Data Leakage Detector

Catches sensitive data in agent outputs before it leaves the system:
    - API keys (OpenAI, AWS, GCP, Azure, Stripe, GitHub, etc.)
    - Passwords and secrets in common formats
    - Private keys (RSA, SSH, PGP)
    - PII (email, phone, SSN, credit card)
    - Internal URLs and IP addresses
    - Environment variable dumps

Scans LLM outputs and tool call results.
Zero external dependencies.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Pattern, Tuple

from agentguard.rules.base import BaseRule
from agentguard.types import (
    AuditRecord,
    EventType,
    RuleAction,
    RuleContext,
    RuleSeverity,
    RuleViolation,
)


class DataLeakageDetector(BaseRule):
    """
    Detect sensitive data leakage in agent outputs.

    Scans LLM responses and tool outputs for API keys, passwords, PII,
    and other sensitive patterns. Designed to catch accidental exposure
    before data leaves the system.

    Args:
        scan_categories: Which categories to scan. Default: all.
            Options: "api_keys", "passwords", "private_keys", "pii", "internal"
        extra_patterns: Additional (name, pattern) tuples to detect.
        allowlist: Set of strings to ignore (e.g., known safe tokens).
        action: What to do on detection. Default BLOCK.

    Example::

        rule = DataLeakageDetector(
            scan_categories=["api_keys", "passwords"],
            allowlist={"sk-test-not-real-key"},
        )
    """

    # ================================================================
    # Detection patterns by category
    # ================================================================

    API_KEY_PATTERNS: List[Tuple[str, str]] = [
        ("OpenAI API Key", r"sk-[a-zA-Z0-9]{20,}"),
        ("OpenAI Project Key", r"sk-proj-[a-zA-Z0-9\-_]{20,}"),
        ("Anthropic API Key", r"sk-ant-[a-zA-Z0-9\-_]{20,}"),
        ("AWS Access Key", r"AKIA[0-9A-Z]{16}"),
        ("AWS Secret Key", r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}"),
        ("GCP API Key", r"AIza[0-9A-Za-z\-_]{35}"),
        ("GCP Service Account", r'"type"\s*:\s*"service_account"'),
        ("Azure Key", r"(?i)(azure|subscription)[_\-]?key\s*[=:]\s*['\"]?[a-f0-9]{32}"),
        ("Stripe Secret Key", r"sk_(live|test)_[a-zA-Z0-9]{24,}"),
        ("Stripe Publishable Key", r"pk_(live|test)_[a-zA-Z0-9]{24,}"),
        ("GitHub Token", r"gh[pousr]_[A-Za-z0-9_]{36,}"),
        ("GitHub Classic Token", r"ghp_[A-Za-z0-9]{36}"),
        ("GitLab Token", r"glpat-[A-Za-z0-9\-_]{20,}"),
        ("Slack Token", r"xox[bpras]-[0-9]{10,}-[a-zA-Z0-9\-]+"),
        ("Slack Webhook", r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[a-zA-Z0-9]+"),
        ("Discord Token", r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27,}"),
        ("Telegram Bot Token", r"\d{8,10}:[A-Za-z0-9_-]{35}"),
        ("SendGrid Key", r"SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}"),
        ("Twilio Key", r"SK[a-f0-9]{32}"),
        ("Heroku API Key", r"(?i)heroku[_\-]?api[_\-]?key\s*[=:]\s*['\"]?[a-f0-9\-]{36}"),
        ("Generic API Key", r"(?i)(api[_\-]?key|apikey|api[_\-]?token|access[_\-]?token)\s*[=:]\s*['\"]?[a-zA-Z0-9\-_]{20,}['\"]?"),
        ("Generic Secret", r"(?i)(secret|password|passwd|pwd|token)\s*[=:]\s*['\"]?[^\s'\"]{8,}['\"]?"),
        ("Bearer Token", r"(?i)bearer\s+[a-zA-Z0-9\-_.~+/]+=*"),
        ("Basic Auth", r"(?i)basic\s+[A-Za-z0-9+/]+=*"),
    ]

    PRIVATE_KEY_PATTERNS: List[Tuple[str, str]] = [
        ("RSA Private Key", r"-----BEGIN\s*(RSA\s+)?PRIVATE\s+KEY-----"),
        ("EC Private Key", r"-----BEGIN\s*EC\s+PRIVATE\s+KEY-----"),
        ("SSH Private Key", r"-----BEGIN\s*OPENSSH\s+PRIVATE\s+KEY-----"),
        ("PGP Private Key", r"-----BEGIN\s*PGP\s+PRIVATE\s+KEY\s+BLOCK-----"),
        ("PKCS8 Key", r"-----BEGIN\s*ENCRYPTED\s+PRIVATE\s+KEY-----"),
        ("Certificate", r"-----BEGIN\s*CERTIFICATE-----"),
    ]

    PII_PATTERNS: List[Tuple[str, str]] = [
        ("Email Address", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        ("US Phone", r"(?<!\d)(\+?1[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}(?!\d)"),
        ("US SSN", r"(?<!\d)\d{3}[\-\s]?\d{2}[\-\s]?\d{4}(?!\d)"),
        ("Credit Card (Visa)", r"(?<!\d)4\d{3}[\-\s]?\d{4}[\-\s]?\d{4}[\-\s]?\d{4}(?!\d)"),
        ("Credit Card (MC)", r"(?<!\d)5[1-5]\d{2}[\-\s]?\d{4}[\-\s]?\d{4}[\-\s]?\d{4}(?!\d)"),
        ("Credit Card (Amex)", r"(?<!\d)3[47]\d{2}[\-\s]?\d{6}[\-\s]?\d{5}(?!\d)"),
        ("IPv4 Private", r"(?<!\d)(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})(?!\d)"),
    ]

    INTERNAL_PATTERNS: List[Tuple[str, str]] = [
        ("Internal URL", r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|internal|staging|dev\.)[\w./\-:]*"),
        ("Env Var Dump", r"(?i)(DATABASE_URL|REDIS_URL|MONGO_URI|DB_PASSWORD)\s*[=:]\s*[^\s]+"),
        ("Connection String", r"(?i)(mongodb|postgres|mysql|redis)://[^\s]+@[^\s]+"),
        (".env Content", r"(?i)^[A-Z_]{3,}=[^\s]{5,}$"),
    ]

    def __init__(
        self,
        scan_categories: Optional[List[str]] = None,
        extra_patterns: Optional[List[Tuple[str, str]]] = None,
        allowlist: Optional[set] = None,
        action: RuleAction = RuleAction.BLOCK,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="DataLeakageDetector",
            severity=RuleSeverity.CRITICAL,
            action=action,
            **kwargs,
        )
        self._allowlist = allowlist or set()

        # Build pattern list based on categories
        categories = set(scan_categories or ["api_keys", "passwords", "private_keys", "pii", "internal"])
        all_patterns: List[Tuple[str, str]] = []

        if "api_keys" in categories or "passwords" in categories:
            all_patterns.extend(self.API_KEY_PATTERNS)
        if "private_keys" in categories:
            all_patterns.extend(self.PRIVATE_KEY_PATTERNS)
        if "pii" in categories:
            all_patterns.extend(self.PII_PATTERNS)
        if "internal" in categories:
            all_patterns.extend(self.INTERNAL_PATTERNS)
        if extra_patterns:
            all_patterns.extend(extra_patterns)

        self._compiled: List[Tuple[str, Pattern[str]]] = []
        for name, pattern in all_patterns:
            try:
                self._compiled.append((name, re.compile(pattern, re.MULTILINE)))
            except re.error:
                pass  # Skip invalid patterns

        self._detections = 0
        self._lock = threading.Lock()

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        # Only scan outputs (LLM responses and tool results)
        if record.event_type not in (
            EventType.LLM_CALL_END,
            EventType.TOOL_CALL_END,
        ):
            return None

        text = record.detail + " " + str(record.metadata.get("result", ""))

        for name, pattern in self._compiled:
            match = pattern.search(text)
            if match:
                matched_text = match.group()
                # Check allowlist
                if matched_text in self._allowlist:
                    continue
                # Skip very short matches (likely false positives)
                if len(matched_text) < 6:
                    continue

                with self._lock:
                    self._detections += 1

                # Redact the matched text for the violation message
                redacted = matched_text[:4] + "***" + matched_text[-2:] if len(matched_text) > 8 else "***"

                return RuleViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    action=self.action,
                    message=f"Data leakage detected: {name} found in agent output",
                    detail=f"Redacted match: '{redacted}'",
                    metadata={
                        "detection_type": name,
                        "redacted_match": redacted,
                        "event_type": record.event_type.value,
                        "action_name": record.action,
                    },
                )

        return None

    @property
    def detections(self) -> int:
        with self._lock:
            return self._detections

    def reset(self) -> None:
        with self._lock:
            self._detections = 0
