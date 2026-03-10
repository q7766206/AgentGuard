# Changelog

## 0.2.0 (2026-03-10)

### Added — Shield: Active Defense Modules
- `PromptInjectionDetector` — Multi-layer prompt injection detection (regex + heuristic scoring + canary tokens), supports EN/ZH/ES/FR
- `DataLeakageDetector` — Catches API keys (OpenAI/AWS/GCP/Stripe/GitHub/etc.), passwords, PII, private keys, connection strings in agent output (25+ patterns)
- `BehaviorAnomalyDetector` — Detects behavioral shifts after reading external content (new tool usage, frequency spikes)
- `ExfilDetector` — Catches data exfiltration attempts (base64+curl, reverse shells, DNS exfil, untrusted domain access)

### Fixed
- CI workflow: Fixed `PYTHONPATH=src` syntax incompatible with Windows PowerShell
- CI workflow: Dropped Python 3.9 (setup-python download timeout on GitHub Actions)
- `AgentGuard.before_tool_call()`: Fixed violation snapshot mechanism — now only reacts to violations triggered by the current record, not stale violations from previous calls

### Changed
- Bumped minimum Python version to 3.10
- Updated pyproject.toml keywords and classifiers for better discoverability

## 0.1.0 (2026-03-10)

### Added
- Core `AgentGuard` class with full agent lifecycle hooks
- `AuditTrail` with in-memory ring buffer and JSONL persistence
- `RuleEngine` with subscriber pattern and real-time evaluation
- 5 built-in rules: `LoopDetection`, `TokenBudget`, `ErrorCascade`, `SensitiveOp`, `TimeoutGuard`
- `BaseRule` abstract class for custom rule creation
- `BaseStorage` with `MemoryStorage` and `JSONLStorage` backends
- LangChain 1.0 middleware adapter (`make_guard_middleware`)
- CrewAI callback adapter (`make_crewai_callbacks`)
- Structured exception hierarchy: `AgentGuardBlock`, `AgentGuardKill`, `AgentGuardThrottle`
- Full type annotations and thread-safety
- 168 tests, all passing
- Performance: ~14μs/record (memory), ~138μs/record (disk)
