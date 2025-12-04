# ADR-007: Comprehensive Testing Strategy

## Status
Accepted

## Date
2024-01-20

## Context

NEURECTOMY is a complex platform with:
- Multiple services (Rust, Python, TypeScript)
- Real-time components (WebSockets)
- Database integrations (PostgreSQL, Neo4j, Redis)
- AI/ML components (non-deterministic)
- Security-critical authentication

We need a testing strategy that:
1. Catches bugs before production
2. Enables confident refactoring
3. Documents expected behavior
4. Runs efficiently in CI/CD

## Decision

We will implement a **multi-layered testing strategy**:

### Testing Pyramid
```
                    ╱╲
                   ╱  ╲
                  ╱ E2E╲           Few, slow, high confidence
                 ╱──────╲
                ╱        ╲
               ╱Integration╲       Moderate count
              ╱────────────╲
             ╱              ╲
            ╱   Unit Tests   ╲     Many, fast, isolated
           ╱──────────────────╲
```

### Test Categories

| Type | Purpose | Tools | Coverage Target |
|------|---------|-------|-----------------|
| **Unit** | Individual functions/modules | pytest, cargo test | 90%+ |
| **Integration** | Component interactions | testcontainers | Key paths |
| **E2E** | Full user workflows | Playwright, Cypress | Critical flows |
| **Property** | Invariant verification | proptest, Hypothesis | Edge cases |
| **Performance** | Benchmarks, load tests | criterion, k6 | Baselines |
| **Security** | Vulnerability scanning | cargo-audit, Trivy | All deps |

### Rust Testing Stack
```toml
[dev-dependencies]
# Async testing
tokio-test = "0.4"

# Property-based testing
proptest = "1.4"

# Test data generation
fake = "2.9"

# Mocking
mockall = "0.12"

# Assertions
pretty_assertions = "1.4"

# Integration testing
testcontainers = "0.15"

# HTTP testing
axum-test = "14"

# Snapshot testing
insta = "1.34"
```

### Test Organization
```
tests/
├── unit/           # Fast, isolated tests
│   ├── auth/
│   ├── db/
│   └── graphql/
├── integration/    # Component tests
│   ├── api.rs
│   ├── auth.rs
│   ├── database.rs
│   └── websocket.rs
├── property/       # Property-based tests
│   ├── auth_properties.rs
│   ├── data_properties.rs
│   └── api_properties.rs
├── e2e/           # End-to-end tests
│   └── user_flows.rs
└── common/        # Shared utilities
    ├── factories.rs
    ├── fixtures.rs
    ├── helpers.rs
    └── mocks.rs
```

## Consequences

### Positive
- **Confidence**: High coverage enables fearless refactoring
- **Documentation**: Tests serve as executable specifications
- **Quality**: Catches regressions automatically
- **Speed**: Pyramid structure optimizes CI time
- **Coverage**: Property tests find edge cases humans miss

### Negative
- **Maintenance**: Tests require ongoing maintenance
- **Initial Investment**: Significant upfront effort
- **Flakiness Risk**: Integration tests can be flaky
- **AI Testing**: Non-deterministic outputs harder to test

### Mitigations
- Use deterministic seeds for randomized tests
- Implement retry logic for flaky tests (with limits)
- Mock AI responses in unit tests
- Use snapshot testing for AI output structure

## Testing Patterns

### Factory Pattern
```rust
pub struct UserFactory;

impl UserFactory {
    pub fn create() -> TestUser {
        TestUser {
            id: Uuid::new_v4(),
            email: fake::faker::internet::en::SafeEmail().fake(),
            username: fake::faker::internet::en::Username().fake(),
            ..Default::default()
        }
    }
    
    pub fn create_admin() -> TestUser {
        Self::create().with_role("admin")
    }
}
```

### Fixture Pattern
```rust
pub mod fixtures {
    pub const VALID_JWT: &str = "eyJhbGciOiJIUzI1NiJ9...";
    pub const VALID_API_KEY: &str = "nrct_test_...";
}
```

### Testcontainers Pattern
```rust
#[tokio::test]
async fn test_with_real_database() {
    let postgres = PostgresContainer::new();
    let pool = create_pool(&postgres.connection_string()).await;
    
    // Test with real database
    let result = create_user(&pool, user_data).await;
    assert!(result.is_ok());
}
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
test:
  runs-on: ubuntu-latest
  steps:
    - name: Run unit tests
      run: cargo test --lib
    
    - name: Run integration tests
      run: cargo test --test integration
    
    - name: Run property tests
      run: cargo test --test property
    
    - name: Check coverage
      run: cargo tarpaulin --out Xml
      
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Coverage Requirements
| Component | Minimum | Target |
|-----------|---------|--------|
| Auth module | 95% | 98% |
| Database layer | 90% | 95% |
| API handlers | 85% | 90% |
| Utilities | 80% | 90% |
| Overall | 85% | 90% |

## Alternatives Considered

### 1. E2E-Heavy Strategy
- ❌ Slow feedback loops
- ❌ Expensive to maintain
- ❌ Harder to debug failures

### 2. Unit-Only Strategy
- ❌ Misses integration bugs
- ❌ False confidence
- ❌ No real-world validation

### 3. Manual Testing Only
- ❌ Not scalable
- ❌ Inconsistent coverage
- ❌ Slow release cycles

## References
- [Test Pyramid (Martin Fowler)](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Property-Based Testing](https://hypothesis.works/articles/what-is-property-based-testing/)
- [Rust Testing Guide](https://doc.rust-lang.org/book/ch11-00-testing.html)
