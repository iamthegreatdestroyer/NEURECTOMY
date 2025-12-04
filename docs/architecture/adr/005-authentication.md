# ADR-005: JWT + API Key Authentication

## Status
Accepted

## Date
2024-01-18

## Context

NEURECTOMY needs authentication that supports:
1. **Interactive users**: Web UI sessions
2. **Programmatic access**: API integrations, CLI tools
3. **Service-to-service**: Internal microservice communication
4. **Multiple sessions**: Users can be logged in on multiple devices

Requirements:
- Stateless where possible (scalability)
- Revocable access
- Fine-grained permissions
- Audit trail

## Decision

We will implement a **dual authentication system**:

### 1. JWT Tokens (Interactive Users)
- **Access Token**: Short-lived (1 hour), used for API requests
- **Refresh Token**: Long-lived (7 days), used to obtain new access tokens
- **Algorithm**: HS256 (development), RS256 (production)

### 2. API Keys (Programmatic Access)
- **Format**: `nrct_<environment>_<random_32_chars>`
- **Storage**: Hashed in database (Argon2id)
- **Scopes**: Fine-grained permission scopes

### Token Structure
```json
{
  "sub": "user-uuid",
  "email": "user@example.com",
  "role": "admin",
  "token_type": "access",
  "session_id": "session-uuid",
  "iat": 1704067200,
  "exp": 1704070800,
  "nbf": 1704067200,
  "jti": "unique-token-id",
  "iss": "neurectomy",
  "aud": "neurectomy-api"
}
```

### API Key Format
```
nrct_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
│    │    └── Random identifier (32 chars)
│    └── Environment (live/test)
└── Prefix (neurectomy)
```

## Authentication Flow

### JWT Flow
```
┌──────┐      ┌──────────┐      ┌──────────┐
│Client│      │   API    │      │   Auth   │
└──┬───┘      └────┬─────┘      └────┬─────┘
   │  POST /login  │                 │
   │──────────────>│                 │
   │               │ Validate creds  │
   │               │────────────────>│
   │               │  User + Tokens  │
   │               │<────────────────│
   │ Access+Refresh│                 │
   │<──────────────│                 │
   │               │                 │
   │ GET /api (JWT)│                 │
   │──────────────>│                 │
   │               │ Validate JWT    │
   │               │ (stateless)     │
   │   Response    │                 │
   │<──────────────│                 │
```

### API Key Flow
```
┌──────┐      ┌──────────┐      ┌──────────┐
│Client│      │   API    │      │   DB     │
└──┬───┘      └────┬─────┘      └────┬─────┘
   │ Request +     │                 │
   │ X-API-Key     │                 │
   │──────────────>│                 │
   │               │ Hash key,       │
   │               │ lookup          │
   │               │────────────────>│
   │               │ Key record      │
   │               │<────────────────│
   │               │ Check scopes    │
   │   Response    │                 │
   │<──────────────│                 │
```

## Consequences

### Positive
- **Scalability**: JWT validation is stateless
- **Flexibility**: Two auth methods for different use cases
- **Security**: Short-lived tokens limit exposure
- **Auditability**: Session and key IDs enable tracking
- **Revocation**: Refresh tokens and API keys can be revoked

### Negative
- **Complexity**: Two systems to maintain
- **Token Size**: JWTs can be large in headers
- **Key Management**: API keys require secure storage

### Security Measures
1. **Password Hashing**: Argon2id with OWASP recommended params
2. **Rate Limiting**: Login attempts limited to 5/minute
3. **Token Rotation**: Refresh tokens rotated on use
4. **Secure Transport**: HTTPS required
5. **CORS**: Strict origin validation

## Alternatives Considered

### 1. Session-based Authentication (Cookies)
- ✅ Simpler, browser-native
- ❌ Requires session storage (stateful)
- ❌ CSRF vulnerabilities
- ❌ Not suitable for API clients

### 2. OAuth2 Only
- ✅ Industry standard
- ✅ Supports third-party auth
- ❌ Complex for simple API access
- ❌ Overkill for internal services

### 3. API Keys Only
- ✅ Simple for programmatic access
- ❌ Not suitable for web UI (long-lived)
- ❌ Harder to implement fine-grained sessions

## Password Policy

| Requirement | Value |
|-------------|-------|
| Minimum length | 12 characters |
| Maximum length | 128 characters |
| Require uppercase | Yes |
| Require lowercase | Yes |
| Require digit | Yes |
| Require special char | Yes |
| Minimum unique chars | 6 |

## References
- [JWT Best Practices (RFC 8725)](https://datatracker.ietf.org/doc/html/rfc8725)
- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [Argon2 Specification](https://github.com/P-H-C/phc-winner-argon2)
