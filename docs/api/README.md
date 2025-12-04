# NEURECTOMY API Documentation

## Overview

NEURECTOMY provides a GraphQL API for interacting with the agent orchestration platform. The API supports:
- **Queries**: Read operations
- **Mutations**: Write operations
- **Subscriptions**: Real-time updates via WebSocket

## Base URLs

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:8080/graphql` |
| Production | `https://api.neurectomy.io/graphql` |

## Authentication

All API requests require authentication via one of two methods:

### JWT Bearer Token
```http
Authorization: Bearer <jwt_token>
```

### API Key
```http
X-API-Key: nrct_live_<key>
```

## GraphQL Endpoint

### Query Endpoint
```
POST /graphql
Content-Type: application/json

{
  "query": "query { ... }",
  "variables": { ... }
}
```

### Subscription Endpoint (WebSocket)
```
WS /graphql
```

## Schema Reference

- [Queries](./graphql/queries.md)
- [Mutations](./graphql/mutations.md)
- [Subscriptions](./graphql/subscriptions.md)
- [Types](./graphql/types.md)

## REST Endpoints

For non-GraphQL operations:

- [Health & Metrics](./rest/health.md)
- [Authentication](./rest/auth.md)
- [File Upload](./rest/upload.md)

## Error Handling

All errors follow a consistent format:

```json
{
  "errors": [
    {
      "message": "Human-readable error message",
      "extensions": {
        "code": "ERROR_CODE",
        "field": "fieldName",
        "timestamp": "2024-01-15T10:30:00Z"
      }
    }
  ]
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHENTICATED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Invalid input data |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## Rate Limiting

| Endpoint | Limit | Window |
|----------|-------|--------|
| GraphQL queries | 100/min | Per user |
| GraphQL mutations | 30/min | Per user |
| Authentication | 5/min | Per IP |
| File upload | 10/min | Per user |

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

## Pagination

All list queries use cursor-based pagination (Relay specification):

```graphql
query {
  agents(first: 10, after: "cursor") {
    edges {
      cursor
      node {
        id
        name
      }
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
    totalCount
  }
}
```

## Filtering & Sorting

```graphql
query {
  agents(
    first: 10
    filter: { status: ACTIVE, createdAfter: "2024-01-01" }
    orderBy: { field: CREATED_AT, direction: DESC }
  ) {
    edges {
      node {
        id
        name
        status
      }
    }
  }
}
```

## SDKs

Official SDKs:
- [JavaScript/TypeScript](https://github.com/neurectomy/sdk-js)
- [Python](https://github.com/neurectomy/sdk-python)
- [Rust](https://github.com/neurectomy/sdk-rust)

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for API version history.
