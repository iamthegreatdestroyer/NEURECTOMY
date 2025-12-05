# ADR-009: Error Handling Patterns

## Status

Accepted

## Date

2025-01-27

## Context

NEURECTOMY has multiple error sources:

- Network failures (API calls, WebSocket)
- Validation errors (user input, config)
- Business logic errors (workflow failures, agent errors)
- System errors (database, file system)
- External service errors (LLM APIs, container runtimes)

Challenges:

1. Inconsistent error handling across packages
2. No standardized error codes
3. Unclear error recovery strategies
4. Poor error observability

## Decision

We will implement a **hierarchical error system** with typed errors and recovery strategies.

### Error Class Hierarchy

```typescript
// Base error class
class NeurectomyError extends Error {
  readonly code: string;
  readonly statusCode?: number;
  readonly isRetryable: boolean;
  readonly context?: Record<string, unknown>;
  readonly cause?: Error;
}

// Specialized errors
class ValidationError extends NeurectomyError {}
class NetworkError extends NeurectomyError {}
class TimeoutError extends NetworkError {}
class AuthenticationError extends NeurectomyError {}
class AuthorizationError extends NeurectomyError {}
class AgentError extends NeurectomyError {}
class WorkflowError extends NeurectomyError {}
class ConfigurationError extends NeurectomyError {}
class ResourceNotFoundError extends NeurectomyError {}
class ConflictError extends NeurectomyError {}
class RateLimitError extends NetworkError {}
```

### Error Code Convention

Format: `{DOMAIN}_{CATEGORY}_{SPECIFIC}`

| Domain     | Categories                  | Examples                     |
| ---------- | --------------------------- | ---------------------------- |
| `AGENT`    | `EXEC`, `CONFIG`, `STATE`   | `AGENT_EXEC_TIMEOUT`         |
| `WORKFLOW` | `PARSE`, `RUN`, `STEP`      | `WORKFLOW_RUN_FAILED`        |
| `API`      | `REQ`, `AUTH`, `RATE`       | `API_AUTH_TOKEN_EXPIRED`     |
| `DATA`     | `VALID`, `PERSIST`, `QUERY` | `DATA_VALID_SCHEMA_MISMATCH` |
| `SYS`      | `IO`, `MEM`, `NET`          | `SYS_NET_UNREACHABLE`        |

### Error Response Format

All API errors follow this structure:

```typescript
interface ErrorResponse {
  error: {
    code: string; // "AGENT_EXEC_TIMEOUT"
    message: string; // Human-readable message
    details?: unknown; // Additional context
    requestId?: string; // For tracing
    timestamp: string; // ISO 8601
    path?: string; // Request path
    retryable: boolean; // Can client retry?
    retryAfter?: number; // Seconds to wait
  };
}
```

### Recovery Strategies

| Error Type            | Strategy           | Implementation                     |
| --------------------- | ------------------ | ---------------------------------- |
| `NetworkError`        | Retry with backoff | exponential backoff, max 3 retries |
| `TimeoutError`        | Retry once         | immediate retry, then fail         |
| `RateLimitError`      | Wait and retry     | honor `retryAfter` header          |
| `ValidationError`     | Fix input          | show user errors, no retry         |
| `AuthenticationError` | Re-authenticate    | redirect to login/refresh token    |
| `AuthorizationError`  | Fail               | show permission denied             |
| `ConfigurationError`  | Fail               | require manual fix                 |
| `ConflictError`       | Resolve            | show conflict UI or auto-merge     |

### Error Boundaries (React)

```tsx
// Component-level boundary
<ErrorBoundary fallback={<ErrorFallback />}>
  <Component />
</ErrorBoundary>

// Route-level boundary
<RouteErrorBoundary>
  <RouterProvider router={router} />
</RouteErrorBoundary>

// Global boundary
<GlobalErrorBoundary onError={reportToSentry}>
  <App />
</GlobalErrorBoundary>
```

### Error Logging

All errors are logged with:

- Error code and message
- Stack trace (development)
- Request context (user, route, params)
- System context (version, environment)
- Correlation ID for tracing

```typescript
logger.error("Operation failed", {
  error: {
    code: err.code,
    message: err.message,
    stack: err.stack,
  },
  context: {
    userId: req.userId,
    requestId: req.id,
    path: req.path,
  },
});
```

## Consequences

### Positive

- **Consistent handling**: Same patterns everywhere
- **Better DX**: Clear error types and recovery
- **Observability**: Structured logging enables monitoring
- **User experience**: Appropriate error messages
- **Debugging**: Error context aids troubleshooting

### Negative

- **Boilerplate**: More code for error handling
- **Maintenance**: Must keep error codes documented
- **Learning curve**: Team must learn patterns

### Mitigations

- Utility functions reduce boilerplate
- CI checks for undocumented error codes
- Error handling guide in docs

## Alternatives Considered

### 1. Result Type Pattern (Rust-style)

```typescript
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };
```

**Partially adopted**: Used in some pure functions, but exceptions remain primary for async operations.

### 2. Functional Error Handling (fp-ts)

**Rejected**: Too steep a learning curve, overkill for our needs.

### 3. Simple String Errors

**Rejected**: No type safety, poor DX, no structured logging.

## Implementation

### Phase 1: Core Errors (Complete)

- [x] Define error class hierarchy in @neurectomy/core
- [x] Create error factory functions
- [x] Add error types

### Phase 2: API Integration

- [ ] Standardize API error responses
- [ ] Add error middleware
- [ ] Create error documentation

### Phase 3: Client Integration

- [ ] Create error handling hooks
- [ ] Add error boundaries
- [ ] Implement retry utilities

### Phase 4: Observability

- [ ] Structured error logging
- [ ] Error rate dashboards
- [ ] Alert rules for critical errors

## Error Catalog

### Agent Errors

| Code                   | Message                        | HTTP | Retryable |
| ---------------------- | ------------------------------ | ---- | --------- |
| `AGENT_NOT_FOUND`      | Agent not found                | 404  | No        |
| `AGENT_EXEC_TIMEOUT`   | Agent execution timed out      | 504  | Yes       |
| `AGENT_EXEC_FAILED`    | Agent execution failed         | 500  | Maybe     |
| `AGENT_CONFIG_INVALID` | Invalid agent configuration    | 400  | No        |
| `AGENT_STATE_INVALID`  | Invalid agent state transition | 409  | No        |

### Workflow Errors

| Code                      | Message                  | HTTP | Retryable |
| ------------------------- | ------------------------ | ---- | --------- |
| `WORKFLOW_NOT_FOUND`      | Workflow not found       | 404  | No        |
| `WORKFLOW_PARSE_ERROR`    | Failed to parse workflow | 400  | No        |
| `WORKFLOW_STEP_FAILED`    | Workflow step failed     | 500  | Maybe     |
| `WORKFLOW_CYCLE_DETECTED` | Workflow contains cycle  | 400  | No        |

### API Errors

| Code                | Message                 | HTTP | Retryable |
| ------------------- | ----------------------- | ---- | --------- |
| `API_AUTH_REQUIRED` | Authentication required | 401  | No        |
| `API_AUTH_INVALID`  | Invalid credentials     | 401  | No        |
| `API_AUTH_EXPIRED`  | Token expired           | 401  | Yes\*     |
| `API_FORBIDDEN`     | Access denied           | 403  | No        |
| `API_RATE_LIMITED`  | Rate limit exceeded     | 429  | Yes       |

\*Retry after refreshing token

## References

- [Microsoft REST API Guidelines - Errors](https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses)
- [Google Cloud Error Model](https://cloud.google.com/apis/design/errors)
- [Problem Details RFC 7807](https://www.rfc-editor.org/rfc/rfc7807)
