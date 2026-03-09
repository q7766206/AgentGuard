# Changelog

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
- 127 tests, all passing
- Performance: ~13μs/record (memory), ~100μs/record (disk)
