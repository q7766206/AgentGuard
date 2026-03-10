<p align="center">
  <h1 align="center">🛡️ AgentGuard</h1>
  <p align="center"><strong>Open-source security middleware for AI agents</strong></p>
  <p align="center">Audit every action. Enforce safety rules. Stop agents before they go wrong.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/agentguard/"><img src="https://img.shields.io/pypi/v/agentguard?color=blue" alt="PyPI"></a>
  <a href="https://github.com/q7766206/agentguard/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
</p>

---

**AgentGuard** is a lightweight, framework-agnostic security layer that sits between your AI agent and the real world. It records every LLM call, tool execution, and decision as an immutable audit trail — and enforces configurable safety rules that can **block**, **throttle**, or **kill** an agent before it causes damage.

**Zero external dependencies.** Works with LangChain, CrewAI, or any custom framework.

## Why AgentGuard?

AI agents are increasingly autonomous — they browse the web, execute code, manage files, and call APIs. But **nobody is watching what they do**. A single hallucination can lead to `rm -rf /`, a runaway loop can burn $500 in API costs, and cascading errors can corrupt your data.

AgentGuard solves this by providing:

- **Audit Trail** — Every action recorded as immutable, queryable, exportable logs
- **Rule Engine** — 5 built-in rules that catch the most common agent failures
- **Shield** — 4 active defense modules against external attacks (prompt injection, data leakage, exfiltration, behavioral hijacking)
- **Real-time Enforcement** — Block dangerous operations *before* they execute
- **Framework Adapters** — Drop-in integration with LangChain 1.0, CrewAI, and more

## Quickstart

```bash
pip install agentguard
```

```python
from agentguard import AgentGuard, AgentGuardBlock
from agentguard.rules import LoopDetection, TokenBudget, SensitiveOp

guard = AgentGuard(rules=[
    LoopDetection(max_repeats=5),     # Stop infinite loops
    TokenBudget(max_tokens=100_000),  # Prevent cost overruns
    SensitiveOp(),                    # Block dangerous commands
])

# In your agent loop:
guard.before_tool_call("web_search", {"query": "AI safety"})
guard.after_tool_call("web_search", result="Found 10 papers", duration_ms=350)

# Dangerous operations are blocked automatically:
try:
    guard.before_tool_call("shell", {"command": "rm -rf /"})
except AgentGuardBlock as e:
    print(f"Blocked: {e}")
    # → [AgentGuard BLOCK] | Rule: SensitiveOp | Tool: shell | ...
```

## Built-in Rules

| Rule | What it catches | Default action |
|------|----------------|----------------|
| `LoopDetection` | Same tool called N times in M seconds | **BLOCK** |
| `TokenBudget` | Total tokens exceed threshold | **KILL** |
| `ErrorCascade` | N errors in M seconds (circuit breaker) | **KILL** |
| `SensitiveOp` | `rm -rf`, `DROP TABLE`, `sudo`, `.env` access, etc. | **BLOCK** |
| `TimeoutGuard` | Single operation exceeds time limit | **LOG** |

All rules are configurable:

```python
LoopDetection(max_repeats=3, window_seconds=30)
TokenBudget(max_tokens=50_000, max_tokens_per_call=5000)
ErrorCascade(max_errors=3, window_seconds=30)
SensitiveOp(extra_patterns=[r"my_secret"], blocked_tools=["dangerous_tool"])
TimeoutGuard(max_duration_ms=10_000)
```

## Shield — Active Defense

AgentGuard Shield protects agents from **external attacks**, not just self-inflicted errors.

```python
from agentguard import AgentGuard
from agentguard.rules import SensitiveOp
from agentguard.shield import (
    PromptInjectionDetector,
    DataLeakageDetector,
    BehaviorAnomalyDetector,
    ExfilDetector,
)

guard = AgentGuard(rules=[
    # Self-protection (built-in rules)
    SensitiveOp(),

    # External attack protection (Shield)
    PromptInjectionDetector(sensitivity="medium"),
    DataLeakageDetector(),
    BehaviorAnomalyDetector(baseline_window=20),
    ExfilDetector(trusted_domains={"api.openai.com", "google.com"}),
])
```

### Shield Modules

| Module | What it catches | Detection method |
|--------|----------------|-----------------|
| `PromptInjectionDetector` | "Ignore previous instructions", role hijacking, jailbreaks | Multi-layer: regex (4 languages) + heuristic scoring + canary tokens |
| `DataLeakageDetector` | API keys, passwords, PII, private keys, connection strings | 25+ regex patterns for OpenAI/AWS/GCP/Stripe/GitHub/etc. |
| `BehaviorAnomalyDetector` | Agent behavioral shifts after reading external content | Rolling baseline comparison, new tool detection, frequency spike |
| `ExfilDetector` | `base64 | curl`, reverse shells, DNS exfil, untrusted domains | Pattern matching + domain allowlist |

### Prompt Injection Detection (Multi-language)

```python
# Detects injection in English, Chinese, Spanish, French
detector = PromptInjectionDetector(
    sensitivity="high",           # low/medium/high
    canary_token="MY_SECRET_123", # Optional: detect if agent leaks this
    scan_tool_output=True,        # Scan tool results for indirect injection
)
```

### Data Leakage Prevention

```python
detector = DataLeakageDetector(
    scan_categories=["api_keys", "passwords", "pii"],  # Choose what to scan
    allowlist={"sk-test-not-real"},                     # Known safe strings
)
```

## Custom Rules

Create your own rules in ~20 lines:

```python
from agentguard.rules import BaseRule
from agentguard.types import AuditRecord, RuleContext, RuleViolation, RuleAction, RuleSeverity

class CostBudget(BaseRule):
    """Kill agent when estimated API cost exceeds threshold."""

    def __init__(self, max_cost_usd: float = 5.0):
        super().__init__(name="CostBudget", severity=RuleSeverity.HIGH, action=RuleAction.KILL)
        self._max_cost = max_cost_usd

    def evaluate(self, record: AuditRecord, context: RuleContext):
        cost = (context.total_tokens / 1000) * 0.01  # $0.01/1k tokens
        if cost >= self._max_cost:
            return RuleViolation(
                rule_name=self.name, severity=self.severity, action=self.action,
                message=f"Cost ${cost:.2f} exceeds ${self._max_cost:.2f}",
            )
        return None

guard = AgentGuard(rules=[CostBudget(max_cost_usd=1.0)])
```

## LangChain 1.0 Integration

```python
from agentguard import AgentGuard
from agentguard.rules import LoopDetection, TokenBudget, SensitiveOp
from agentguard.adapters.langchain import make_guard_middleware
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

guard = AgentGuard(rules=[LoopDetection(), TokenBudget(), SensitiveOp()])

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[...],
    middleware=[*make_guard_middleware(guard)],  # ← One line
)
```

## CrewAI Integration

```python
from agentguard import AgentGuard
from agentguard.rules import SensitiveOp
from agentguard.adapters.crewai import make_crewai_callbacks

guard = AgentGuard(rules=[SensitiveOp()])
step_cb, task_cb = make_crewai_callbacks(guard)

agent = Agent(role="researcher", step_callback=step_cb, ...)
task = Task(description="...", callback=task_cb, ...)
```

## Audit Trail

Every action is recorded and queryable:

```python
# Query recent errors
errors = guard.get_records(level=LogLevel.ERROR, limit=10)

# Export full audit trail as JSON
print(guard.export_json())

# Persist to disk (JSONL format, compatible with jq/ELK/Splunk)
guard = AgentGuard(persist=True, persist_path="./audit.jsonl")

# Get session stats
stats = guard.get_stats()
# → {"session_id": "abc123", "audit": {"total_records": 42}, "rules": {"total_violations": 2}}
```

## Performance

AgentGuard is designed to add negligible overhead to your agent:

| Scenario | Latency per record |
|----------|-------------------|
| Memory-only | **~13μs** |
| With disk persistence | **~100μs** |
| With 1 subscriber | **~15μs** |
| 10 threads concurrent | **~16μs** |

For comparison, a single LLM API call takes 500ms–5s. AgentGuard's overhead is <0.01%.

## Architecture

```
Your Agent
    │
    ├── before_tool_call() ──→ AuditTrail ──→ RuleEngine ──→ Block/Kill/Log
    │                              │               │
    │                         MemoryStorage    ┌── Built-in Rules ──┐
    │                         JSONLStorage     │  LoopDetection     │
    │                                          │  TokenBudget       │
    ├── after_tool_call()                      │  ErrorCascade      │
    ├── before_llm_call()                      │  SensitiveOp       │
    ├── after_llm_call()                       │  TimeoutGuard      │
    │                                          └───────────────────┘
    │                                          ┌── Shield Modules ──┐
    │                                          │  PromptInjection   │
    │                                          │  DataLeakage       │
    │                                          │  BehaviorAnomaly   │
    │                                          │  ExfilDetector     │
    │                                          └───────────────────┘
    └── get_stats() / export_json()
```

## Project Structure

```
agentguard/
├── __init__.py          # Public API exports
├── core.py              # AgentGuard main class
├── audit.py             # AuditTrail (recording, querying, export)
├── types.py             # All types, enums, data structures
├── exceptions.py        # AgentGuardBlock, AgentGuardKill, etc.
├── rules/
│   ├── base.py          # BaseRule abstract class
│   ├── builtin.py       # 5 built-in rules
│   └── engine.py        # RuleEngine (evaluation, context tracking)
├── shield/
│   ├── injection.py     # Prompt injection detector (4 languages)
│   ├── leakage.py       # Data leakage detector (25+ patterns)
│   ├── anomaly.py       # Behavior anomaly detector
│   └── exfil.py         # Exfiltration detector
├── storage/
│   ├── base.py          # BaseStorage abstract class
│   ├── memory.py        # In-memory ring buffer
│   └── jsonl.py         # JSONL file persistence
└── adapters/
    ├── langchain.py     # LangChain 1.0 middleware adapter
    └── crewai.py        # CrewAI callback adapter
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/agentguard/agentguard.git
cd agentguard
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0 — use it in production, fork it, build on it.
