//! Authentication Middleware
//!
//! Axum middleware for JWT and API key authentication:
//! - Token extraction from Authorization header
//! - Role-based access control
//! - Permission checking
//!
//! @FORTRESS - Defense in Depth

use axum::{
    extract::State,
    http::{Request, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
    body::Body,
};
use std::sync::Arc;

use super::{AuthError, AuthResult, JwtService, Claims, Role, Permission};

/// Extract bearer token from request
pub fn extract_bearer_token<B>(req: &Request<B>) -> Option<String> {
    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(|s| s.to_string())
}

/// Extract API key from request
pub fn extract_api_key<B>(req: &Request<B>) -> Option<String> {
    // Try X-API-Key header first
    req.headers()
        .get("X-API-Key")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            // Fall back to query parameter
            req.uri()
                .query()
                .and_then(|q| {
                    q.split('&')
                        .find_map(|pair| {
                            let mut parts = pair.split('=');
                            if parts.next()? == "api_key" {
                                parts.next().map(|s| s.to_string())
                            } else {
                                None
                            }
                        })
                })
        })
}

/// Authentication middleware state
#[derive(Clone)]
pub struct AuthMiddleware {
    pub jwt_service: Arc<JwtService>,
}

impl AuthMiddleware {
    pub fn new(jwt_service: Arc<JwtService>) -> Self {
        Self { jwt_service }
    }

    /// Authenticate request and extract claims
    pub fn authenticate(&self, token: &str) -> AuthResult<Claims> {
        self.jwt_service.validate_access_token(token)
    }
}

/// Require authentication middleware
pub async fn require_auth(
    State(auth): State<Arc<AuthMiddleware>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    let token = match extract_bearer_token(&req) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                "Missing authorization token"
            ).into_response()
        }
    };

    let claims = match auth.authenticate(&token) {
        Ok(c) => c,
        Err(AuthError::TokenExpired) => {
            return (
                StatusCode::UNAUTHORIZED,
                "Token has expired"
            ).into_response()
        }
        Err(_) => {
            return (
                StatusCode::UNAUTHORIZED,
                "Invalid token"
            ).into_response()
        }
    };

    // Insert claims into request extensions
    req.extensions_mut().insert(claims);

    next.run(req).await
}

/// Require specific role middleware
#[derive(Clone)]
pub struct RequireRole {
    pub role: Role,
}

impl RequireRole {
    pub fn new(role: Role) -> Self {
        Self { role }
    }

    pub fn admin() -> Self {
        Self::new(Role::Admin)
    }

    pub fn developer() -> Self {
        Self::new(Role::Developer)
    }
}

/// Check role middleware
pub async fn require_role(
    State((auth, required_role)): State<(Arc<AuthMiddleware>, Role)>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let token = match extract_bearer_token(&req) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                "Missing authorization token"
            ).into_response()
        }
    };

    let claims = match auth.authenticate(&token) {
        Ok(c) => c,
        Err(_) => {
            return (
                StatusCode::UNAUTHORIZED,
                "Invalid token"
            ).into_response()
        }
    };

    // Check role
    let user_role: Role = claims.role.parse().unwrap_or(Role::Guest);
    
    // Admin has access to everything
    if user_role != Role::Admin && user_role != required_role {
        return (
            StatusCode::FORBIDDEN,
            "Insufficient permissions"
        ).into_response();
    }

    next.run(req).await
}

/// Require specific permission middleware
#[derive(Clone)]
pub struct RequireAuth {
    pub permissions: Vec<Permission>,
    pub require_all: bool,
}

impl RequireAuth {
    pub fn new(permissions: Vec<Permission>) -> Self {
        Self {
            permissions,
            require_all: false,
        }
    }

    pub fn require_all(mut self) -> Self {
        self.require_all = true;
        self
    }

    pub fn any_of(permissions: Vec<Permission>) -> Self {
        Self {
            permissions,
            require_all: false,
        }
    }

    pub fn all_of(permissions: Vec<Permission>) -> Self {
        Self {
            permissions,
            require_all: true,
        }
    }
}

/// Permission check middleware
pub async fn require_permission(
    State((auth, required_permissions, require_all)): State<(Arc<AuthMiddleware>, Vec<Permission>, bool)>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let token = match extract_bearer_token(&req) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                "Missing authorization token"
            ).into_response()
        }
    };

    let claims = match auth.authenticate(&token) {
        Ok(c) => c,
        Err(_) => {
            return (
                StatusCode::UNAUTHORIZED,
                "Invalid token"
            ).into_response()
        }
    };

    // Admin has all permissions
    let user_role: Role = claims.role.parse().unwrap_or(Role::Guest);
    if user_role == Role::Admin {
        return next.run(req).await;
    }

    // Check user permissions
    let user_permissions: Vec<Permission> = user_role.permissions();
    
    let has_permission = if require_all {
        required_permissions.iter().all(|p| user_permissions.contains(p))
    } else {
        required_permissions.iter().any(|p| user_permissions.contains(p))
    };

    if !has_permission {
        return (
            StatusCode::FORBIDDEN,
            "Insufficient permissions"
        ).into_response();
    }

    next.run(req).await
}

/// Rate limiting state
#[derive(Clone)]
pub struct RateLimiter {
    /// Requests per window
    pub requests_per_window: u32,
    /// Window duration in seconds
    pub window_seconds: u64,
    /// Store for request counts (in production, use Redis)
    store: Arc<tokio::sync::RwLock<std::collections::HashMap<String, (u32, std::time::Instant)>>>,
}

impl RateLimiter {
    pub fn new(requests_per_window: u32, window_seconds: u64) -> Self {
        Self {
            requests_per_window,
            window_seconds,
            store: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub async fn check(&self, key: &str) -> bool {
        let mut store = self.store.write().await;
        let now = std::time::Instant::now();

        if let Some((count, window_start)) = store.get_mut(key) {
            if now.duration_since(*window_start).as_secs() > self.window_seconds {
                // New window
                *count = 1;
                *window_start = now;
                true
            } else if *count >= self.requests_per_window {
                // Rate limited
                false
            } else {
                // Increment count
                *count += 1;
                true
            }
        } else {
            // New key
            store.insert(key.to_string(), (1, now));
            true
        }
    }
}

/// Rate limiting middleware
pub async fn rate_limit(
    State(limiter): State<Arc<RateLimiter>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // Use IP address as key (in production, also consider user ID)
    let key = req
        .headers()
        .get("X-Forwarded-For")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    if !limiter.check(&key).await {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded"
        ).into_response();
    }

    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;

    #[test]
    fn test_extract_bearer_token() {
        let req = Request::builder()
            .header("Authorization", "Bearer test-token-123")
            .body(Body::empty())
            .unwrap();

        let token = extract_bearer_token(&req);
        assert_eq!(token, Some("test-token-123".to_string()));
    }

    #[test]
    fn test_extract_api_key() {
        let req = Request::builder()
            .header("X-API-Key", "api-key-123")
            .body(Body::empty())
            .unwrap();

        let key = extract_api_key(&req);
        assert_eq!(key, Some("api-key-123".to_string()));
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(3, 60);

        // First 3 requests should pass
        assert!(limiter.check("test-client").await);
        assert!(limiter.check("test-client").await);
        assert!(limiter.check("test-client").await);

        // Fourth should be rate limited
        assert!(!limiter.check("test-client").await);

        // Different client should work
        assert!(limiter.check("other-client").await);
    }
}
