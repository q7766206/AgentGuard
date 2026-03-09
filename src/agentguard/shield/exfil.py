# -*- coding: utf-8 -*-
"""
Exfiltration Detector

Catches attempts to exfiltrate data from the agent's environment:
    - Encoding data (base64, hex) and sending via HTTP
    - Writing sensitive data to world-readable locations
    - DNS-based exfiltration (encoding data in DNS queries)
    - Steganographic patterns (hiding data in normal-looking output)
    - Unusual outbound network calls to unknown domains
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


class ExfilDetector(BaseRule):
    """
    Detect data exfiltration attempts by the agent.

    Catches patterns where the agent tries to send internal data to external
    services, encode sensitive information, or use covert channels.

    Args:
        trusted_domains: Set of domains the agent is allowed to contact.
                         If set, any HTTP call to an unlisted domain triggers an alert.
        action: What to do on detection. Default BLOCK.

    Example::

        rule = ExfilDetector(
            trusted_domains={"api.openai.com", "google.com", "github.com"},
        )
    """

    EXFIL_PATTERNS: List[Tuple[str, str]] = [
        # Encoding + sending
        ("Base64 encode + HTTP", r"base64[.\s]*(encode|b64encode).*?(requests|urllib|curl|wget|fetch|http)"),
        ("Base64 + curl pipe", r"base64\s+.*\|\s*(curl|wget|nc)"),
        ("Hex encode + send", r"(hex|encode)\s*\(.*\).*?(curl|wget|requests|http)"),

        # Reverse shell / remote access
        ("Reverse shell", r"(bash|sh|zsh)\s+-i\s+[>|&]+\s+/dev/tcp/"),
        ("Netcat reverse", r"nc\s+-[a-zA-Z]*e\s+(bash|sh|cmd)"),
        ("Python reverse shell", r"socket\.connect\s*\(\s*\(\s*['\"][\d.]+['\"]"),

        # DNS exfiltration
        ("DNS exfil", r"(nslookup|dig|host)\s+[a-zA-Z0-9]{20,}\."),

        # Clipboard / screenshot exfil
        ("Clipboard steal", r"(pbcopy|xclip|xsel|clip\.exe|Get-Clipboard).*?(curl|wget|requests|http)"),
        ("Screenshot + send", r"(screenshot|screencapture|scrot).*?(curl|upload|send|http)"),

        # File exfiltration
        ("Tar + send", r"tar\s+[a-zA-Z]*[cz].*\|\s*(curl|wget|nc|ssh)"),
        ("Zip + upload", r"(zip|7z|rar)\s+.*?(curl|wget|upload|http|ftp)"),
        ("Cat + curl", r"cat\s+.*\|\s*(curl|wget|nc)"),
        ("Scp outbound", r"scp\s+.*\s+\w+@[\d.]+:"),

        # Webhook / external POST
        ("Webhook POST", r"(requests\.post|curl\s+-X\s+POST|fetch.*POST).*?(webhook|hook|callback|exfil)"),

        # Encoded data in URL
        ("Data in URL", r"https?://[^\s]*[?&](data|payload|d|p|q)=[A-Za-z0-9+/=%]{50,}"),
    ]

    URL_PATTERN = re.compile(
        r"https?://([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}",
        re.IGNORECASE,
    )

    def __init__(
        self,
        trusted_domains: Optional[Set[str]] = None,
        action: RuleAction = RuleAction.BLOCK,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="ExfilDetector",
            severity=RuleSeverity.CRITICAL,
            action=action,
            **kwargs,
        )
        self._trusted_domains = trusted_domains
        self._compiled: List[Tuple[str, Pattern[str]]] = [
            (name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
            for name, pattern in self.EXFIL_PATTERNS
        ]
        self._detections = 0
        self._lock = threading.Lock()

    def evaluate(
        self,
        record: AuditRecord,
        context: RuleContext,
    ) -> Optional[RuleViolation]:
        if record.event_type != EventType.TOOL_CALL_START:
            return None

        # Build text to scan from tool call details
        text = " ".join([
            record.detail,
            str(record.metadata.get("args", "")),
            str(record.metadata.get("command", "")),
            str(record.metadata.get("input", "")),
            str(record.metadata.get("code", "")),
            str(record.metadata.get("url", "")),
        ])

        # Pattern-based detection
        for name, pattern in self._compiled:
            match = pattern.search(text)
            if match:
                with self._lock:
                    self._detections += 1
                return RuleViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    action=self.action,
                    message=f"Data exfiltration attempt detected: {name}",
                    detail=f"Matched: '{match.group()[:150]}'",
                    metadata={
                        "exfil_type": name,
                        "matched_text": match.group()[:200],
                        "tool_name": record.action,
                    },
                )

        # Domain allowlist check
        if self._trusted_domains is not None:
            urls = self.URL_PATTERN.findall(text)
            # URL_PATTERN findall returns the last group, so we need to re-search
            for url_match in self.URL_PATTERN.finditer(text):
                full_url = url_match.group()
                # Extract domain
                domain = full_url.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
                # Check if domain or any parent domain is trusted
                if not self._is_trusted(domain):
                    with self._lock:
                        self._detections += 1
                    return RuleViolation(
                        rule_name=self.name,
                        severity=RuleSeverity.HIGH,
                        action=self.action,
                        message=f"Untrusted domain access: '{domain}'",
                        detail=f"URL: {full_url[:200]}",
                        metadata={
                            "exfil_type": "untrusted_domain",
                            "domain": domain,
                            "url": full_url[:200],
                            "trusted_domains": sorted(self._trusted_domains),
                            "tool_name": record.action,
                        },
                    )

        return None

    def _is_trusted(self, domain: str) -> bool:
        """Check if a domain is in the trusted list (including parent domains)."""
        if self._trusted_domains is None:
            return True
        domain = domain.lower()
        # Check exact match and parent domains
        parts = domain.split(".")
        for i in range(len(parts)):
            candidate = ".".join(parts[i:])
            if candidate in self._trusted_domains:
                return True
        return False

    @property
    def detections(self) -> int:
        with self._lock:
            return self._detections

    def reset(self) -> None:
        with self._lock:
            self._detections = 0
