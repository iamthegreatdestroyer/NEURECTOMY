# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for NEURECTOMY.

## Index

| ADR                                      | Title                             | Status   | Date       |
| ---------------------------------------- | --------------------------------- | -------- | ---------- |
| [ADR-001](001-monorepo-structure.md)     | Monorepo Structure with Turborepo | Accepted | 2024-01-15 |
| [ADR-002](002-rust-backend.md)           | Rust for Core Backend Services    | Accepted | 2024-01-15 |
| [ADR-003](003-graphql-api.md)            | GraphQL as Primary API            | Accepted | 2024-01-16 |
| [ADR-004](004-database-strategy.md)      | Multi-Database Strategy           | Accepted | 2024-01-17 |
| [ADR-005](005-authentication.md)         | JWT + API Key Authentication      | Accepted | 2024-01-18 |
| [ADR-006](006-websocket-architecture.md) | WebSocket Real-Time Architecture  | Accepted | 2024-01-19 |
| [ADR-007](007-testing-strategy.md)       | Comprehensive Testing Strategy    | Accepted | 2024-01-20 |
| [ADR-008](008-schema-validation.md)      | Schema Validation Strategy        | Accepted | 2025-01-27 |
| [ADR-009](009-error-handling.md)         | Error Handling Patterns           | Accepted | 2025-01-27 |

## ADR Template

When creating new ADRs, use the following template:

```markdown
# ADR-XXX: Title

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

What is the issue that we're seeing that is motivating this decision?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?

## Alternatives Considered

What other options were considered?
```
