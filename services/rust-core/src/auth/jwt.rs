//! JWT Token Service
//!
//! Implements secure JWT token generation and validation using:
//! - EdDSA (Ed25519) or RS256 algorithms
//! - Access + Refresh token pattern
//! - Token rotation and revocation
//!
//! @CIPHER - Cryptographic best practices

use chrono::{Duration, Utc};
use jsonwebtoken::{
    decode, encode, Algorithm, DecodingKey, EncodingKey, Header, TokenData, Validation,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{rbac::Role, AuthError, AuthResult};

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Email address
    pub email: String,
    /// User role
    pub role: String,
    /// Token type (access, refresh)
    pub token_type: TokenType,
    /// Session ID for tracking
    pub session_id: String,
    /// Issued at (Unix timestamp)
    pub iat: i64,
    /// Expiration (Unix timestamp)
    pub exp: i64,
    /// Not before (Unix timestamp)
    pub nbf: i64,
    /// JWT ID (unique identifier)
    pub jti: String,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
}

/// Token type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    Access,
    Refresh,
}

/// Token pair (access + refresh)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub refresh_expires_in: i64,
}

/// JWT service configuration
#[derive(Debug, Clone)]
pub struct JwtConfig {
    /// Secret key for HS256 (development) or private key path for RS256
    pub secret: String,
    /// Public key path for RS256 (optional)
    pub public_key: Option<String>,
    /// Access token expiration in seconds
    pub access_token_expiry: i64,
    /// Refresh token expiration in seconds
    pub refresh_token_expiry: i64,
    /// Token issuer
    pub issuer: String,
    /// Token audience
    pub audience: String,
    /// Algorithm to use
    pub algorithm: Algorithm,
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret: "neurectomy-development-secret-key-change-in-production".to_string(),
            public_key: None,
            access_token_expiry: 3600,    // 1 hour
            refresh_token_expiry: 604800, // 7 days
            issuer: "neurectomy".to_string(),
            audience: "neurectomy-api".to_string(),
            algorithm: Algorithm::HS256,
        }
    }
}

/// JWT service for token operations
#[derive(Clone)]
pub struct JwtService {
    config: JwtConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl JwtService {
    /// Create new JWT service with configuration
    pub fn new(config: JwtConfig) -> AuthResult<Self> {
        let encoding_key = match config.algorithm {
            Algorithm::HS256 | Algorithm::HS384 | Algorithm::HS512 => {
                EncodingKey::from_secret(config.secret.as_bytes())
            }
            Algorithm::RS256 | Algorithm::RS384 | Algorithm::RS512 => {
                EncodingKey::from_rsa_pem(config.secret.as_bytes())
                    .map_err(|e| AuthError::InternalError(format!("Invalid RSA key: {}", e)))?
            }
            Algorithm::ES256 | Algorithm::ES384 => {
                EncodingKey::from_ec_pem(config.secret.as_bytes())
                    .map_err(|e| AuthError::InternalError(format!("Invalid EC key: {}", e)))?
            }
            Algorithm::EdDSA => EncodingKey::from_ed_pem(config.secret.as_bytes())
                .map_err(|e| AuthError::InternalError(format!("Invalid EdDSA key: {}", e)))?,
            _ => {
                return Err(AuthError::InternalError(
                    "Unsupported algorithm".to_string(),
                ))
            }
        };

        let decoding_key = match config.algorithm {
            Algorithm::HS256 | Algorithm::HS384 | Algorithm::HS512 => {
                DecodingKey::from_secret(config.secret.as_bytes())
            }
            Algorithm::RS256 | Algorithm::RS384 | Algorithm::RS512 => {
                let public_key = config.public_key.as_ref().ok_or_else(|| {
                    AuthError::InternalError("Public key required for RSA".to_string())
                })?;
                DecodingKey::from_rsa_pem(public_key.as_bytes()).map_err(|e| {
                    AuthError::InternalError(format!("Invalid RSA public key: {}", e))
                })?
            }
            Algorithm::ES256 | Algorithm::ES384 => {
                let public_key = config.public_key.as_ref().ok_or_else(|| {
                    AuthError::InternalError("Public key required for EC".to_string())
                })?;
                DecodingKey::from_ec_pem(public_key.as_bytes()).map_err(|e| {
                    AuthError::InternalError(format!("Invalid EC public key: {}", e))
                })?
            }
            Algorithm::EdDSA => {
                let public_key = config.public_key.as_ref().ok_or_else(|| {
                    AuthError::InternalError("Public key required for EdDSA".to_string())
                })?;
                DecodingKey::from_ed_pem(public_key.as_bytes()).map_err(|e| {
                    AuthError::InternalError(format!("Invalid EdDSA public key: {}", e))
                })?
            }
            _ => {
                return Err(AuthError::InternalError(
                    "Unsupported algorithm".to_string(),
                ))
            }
        };

        Ok(Self {
            config,
            encoding_key,
            decoding_key,
        })
    }

    /// Create with default HS256 configuration
    pub fn with_secret(secret: &str) -> Self {
        let config = JwtConfig {
            secret: secret.to_string(),
            ..Default::default()
        };

        Self {
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            config,
        }
    }

    /// Generate a token pair for a user
    pub fn generate_token_pair(
        &self,
        user_id: Uuid,
        email: &str,
        role: &Role,
        session_id: &str,
    ) -> AuthResult<TokenPair> {
        let access_token = self.generate_token(
            user_id,
            email,
            role,
            session_id,
            TokenType::Access,
            self.config.access_token_expiry,
        )?;

        let refresh_token = self.generate_token(
            user_id,
            email,
            role,
            session_id,
            TokenType::Refresh,
            self.config.refresh_token_expiry,
        )?;

        Ok(TokenPair {
            access_token,
            refresh_token,
            token_type: "Bearer".to_string(),
            expires_in: self.config.access_token_expiry,
            refresh_expires_in: self.config.refresh_token_expiry,
        })
    }

    /// Generate a single token
    fn generate_token(
        &self,
        user_id: Uuid,
        email: &str,
        role: &Role,
        session_id: &str,
        token_type: TokenType,
        expiry_seconds: i64,
    ) -> AuthResult<String> {
        let now = Utc::now();
        let exp = now + Duration::seconds(expiry_seconds);

        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            role: role.to_string(),
            token_type,
            session_id: session_id.to_string(),
            iat: now.timestamp(),
            exp: exp.timestamp(),
            nbf: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
            iss: self.config.issuer.clone(),
            aud: self.config.audience.clone(),
        };

        let header = Header::new(self.config.algorithm);

        encode(&header, &claims, &self.encoding_key)
            .map_err(|e| AuthError::InternalError(format!("Token generation failed: {}", e)))
    }

    /// Validate and decode a token
    pub fn validate_token(&self, token: &str) -> AuthResult<Claims> {
        let mut validation = Validation::new(self.config.algorithm);
        validation.set_issuer(&[&self.config.issuer]);
        validation.set_audience(&[&self.config.audience]);
        validation.validate_exp = true;
        validation.validate_nbf = true;

        let token_data: TokenData<Claims> = decode(token, &self.decoding_key, &validation)
            .map_err(|e| match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                jsonwebtoken::errors::ErrorKind::InvalidToken => AuthError::InvalidToken,
                jsonwebtoken::errors::ErrorKind::InvalidSignature => AuthError::InvalidToken,
                _ => AuthError::InvalidToken,
            })?;

        Ok(token_data.claims)
    }

    /// Validate access token specifically
    pub fn validate_access_token(&self, token: &str) -> AuthResult<Claims> {
        let claims = self.validate_token(token)?;

        if claims.token_type != TokenType::Access {
            return Err(AuthError::InvalidToken);
        }

        Ok(claims)
    }

    /// Validate refresh token specifically
    pub fn validate_refresh_token(&self, token: &str) -> AuthResult<Claims> {
        let claims = self.validate_token(token)?;

        if claims.token_type != TokenType::Refresh {
            return Err(AuthError::InvalidToken);
        }

        Ok(claims)
    }

    /// Refresh token pair using refresh token
    pub fn refresh_tokens(&self, refresh_token: &str) -> AuthResult<TokenPair> {
        let claims = self.validate_refresh_token(refresh_token)?;

        let user_id: Uuid = claims.sub.parse().map_err(|_| AuthError::InvalidToken)?;

        let role: Role = claims.role.parse().map_err(|_| AuthError::InvalidToken)?;

        // Generate new session ID on refresh (token rotation)
        let new_session_id = Uuid::new_v4().to_string();

        self.generate_token_pair(user_id, &claims.email, &role, &new_session_id)
    }

    /// Extract claims without validation (for debugging)
    pub fn decode_unverified(&self, token: &str) -> AuthResult<Claims> {
        let mut validation = Validation::new(self.config.algorithm);
        validation.insecure_disable_signature_validation();
        validation.validate_exp = false;
        validation.validate_nbf = false;
        validation.set_required_spec_claims::<&str>(&[]);

        let token_data: TokenData<Claims> =
            decode(token, &self.decoding_key, &validation).map_err(|_| AuthError::InvalidToken)?;

        Ok(token_data.claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_validate_token() {
        let service = JwtService::with_secret("test-secret-key-at-least-32-chars");
        let user_id = Uuid::new_v4();
        let session_id = Uuid::new_v4().to_string();

        let token_pair = service
            .generate_token_pair(user_id, "test@example.com", &Role::Developer, &session_id)
            .expect("Token generation should succeed");

        let claims = service
            .validate_access_token(&token_pair.access_token)
            .expect("Token validation should succeed");

        assert_eq!(claims.sub, user_id.to_string());
        assert_eq!(claims.email, "test@example.com");
        assert_eq!(claims.role, "developer");
    }

    #[test]
    fn test_refresh_token_rotation() {
        let service = JwtService::with_secret("test-secret-key-at-least-32-chars");
        let user_id = Uuid::new_v4();
        let session_id = Uuid::new_v4().to_string();

        let original = service
            .generate_token_pair(user_id, "test@example.com", &Role::Developer, &session_id)
            .expect("Token generation should succeed");

        let refreshed = service
            .refresh_tokens(&original.refresh_token)
            .expect("Token refresh should succeed");

        // New tokens should be valid
        let claims = service
            .validate_access_token(&refreshed.access_token)
            .expect("Refreshed token should be valid");

        assert_eq!(claims.sub, user_id.to_string());
        // Session ID should be different (rotation)
        assert_ne!(claims.session_id, session_id);
    }

    #[test]
    fn test_expired_token() {
        let mut config = JwtConfig::default();
        config.access_token_expiry = -1; // Already expired
        config.secret = "test-secret-key-at-least-32-chars".to_string();

        let service = JwtService::new(config).expect("Service creation should succeed");
        let user_id = Uuid::new_v4();
        let session_id = Uuid::new_v4().to_string();

        let token_pair = service
            .generate_token_pair(user_id, "test@example.com", &Role::Developer, &session_id)
            .expect("Token generation should succeed");

        let result = service.validate_access_token(&token_pair.access_token);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }
}
