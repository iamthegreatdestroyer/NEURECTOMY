//! Password Hashing Service
//!
//! Implements secure password hashing using Argon2id:
//! - Winner of the Password Hashing Competition
//! - Resistant to GPU/ASIC attacks
//! - Memory-hard function
//!
//! @CIPHER - Cryptographic best practices

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Algorithm, Argon2, Params, Version,
};

use super::{AuthError, AuthResult};

/// Password strength requirements
#[derive(Debug, Clone)]
pub struct PasswordPolicy {
    /// Minimum length
    pub min_length: usize,
    /// Maximum length
    pub max_length: usize,
    /// Require uppercase letter
    pub require_uppercase: bool,
    /// Require lowercase letter
    pub require_lowercase: bool,
    /// Require digit
    pub require_digit: bool,
    /// Require special character
    pub require_special: bool,
    /// Minimum unique characters
    pub min_unique_chars: usize,
    /// Banned passwords list (common passwords)
    pub banned_passwords: Vec<String>,
}

impl Default for PasswordPolicy {
    fn default() -> Self {
        Self {
            min_length: 12,
            max_length: 128,
            require_uppercase: true,
            require_lowercase: true,
            require_digit: true,
            require_special: true,
            min_unique_chars: 6,
            banned_passwords: vec![
                "password123!".to_string(),
                "Password123!".to_string(),
                "Admin123!".to_string(),
                "Welcome123!".to_string(),
            ],
        }
    }
}

/// Argon2 configuration for password hashing
#[derive(Debug, Clone)]
pub struct Argon2Config {
    /// Memory cost in KiB (default: 64MB for OWASP recommendation)
    pub memory_cost: u32,
    /// Time cost (iterations)
    pub time_cost: u32,
    /// Parallelism factor
    pub parallelism: u32,
    /// Output length in bytes
    pub output_length: usize,
}

impl Default for Argon2Config {
    fn default() -> Self {
        // OWASP recommended settings for 2024
        Self {
            memory_cost: 65536, // 64 MiB
            time_cost: 3,       // 3 iterations
            parallelism: 4,     // 4 threads
            output_length: 32,  // 256-bit hash
        }
    }
}

/// Password service for hashing and verification
#[derive(Clone)]
pub struct PasswordService {
    argon2: Argon2<'static>,
    policy: PasswordPolicy,
}

impl PasswordService {
    /// Create new password service with default configuration
    pub fn new() -> Self {
        Self::with_config(Argon2Config::default(), PasswordPolicy::default())
    }

    /// Create password service with custom configuration
    pub fn with_config(argon2_config: Argon2Config, policy: PasswordPolicy) -> Self {
        let params = Params::new(
            argon2_config.memory_cost,
            argon2_config.time_cost,
            argon2_config.parallelism,
            Some(argon2_config.output_length),
        )
        .expect("Invalid Argon2 parameters");

        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        Self { argon2, policy }
    }

    /// Validate password against policy
    pub fn validate_password(&self, password: &str) -> AuthResult<()> {
        // Length checks
        if password.len() < self.policy.min_length {
            return Err(AuthError::PasswordTooWeak);
        }
        if password.len() > self.policy.max_length {
            return Err(AuthError::PasswordTooWeak);
        }

        // Character class checks
        if self.policy.require_uppercase && !password.chars().any(|c| c.is_uppercase()) {
            return Err(AuthError::PasswordTooWeak);
        }
        if self.policy.require_lowercase && !password.chars().any(|c| c.is_lowercase()) {
            return Err(AuthError::PasswordTooWeak);
        }
        if self.policy.require_digit && !password.chars().any(|c| c.is_ascii_digit()) {
            return Err(AuthError::PasswordTooWeak);
        }
        if self.policy.require_special && !password.chars().any(|c| !c.is_alphanumeric()) {
            return Err(AuthError::PasswordTooWeak);
        }

        // Unique characters check
        let unique_chars: std::collections::HashSet<char> = password.chars().collect();
        if unique_chars.len() < self.policy.min_unique_chars {
            return Err(AuthError::PasswordTooWeak);
        }

        // Banned passwords check
        let password_lower = password.to_lowercase();
        for banned in &self.policy.banned_passwords {
            if password_lower == banned.to_lowercase() {
                return Err(AuthError::PasswordTooWeak);
            }
        }

        Ok(())
    }

    /// Hash a password using Argon2id
    pub fn hash_password(&self, password: &str) -> AuthResult<String> {
        // Validate password first
        self.validate_password(password)?;

        // Generate random salt
        let salt = SaltString::generate(&mut OsRng);

        // Hash password
        let hash = self
            .argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| AuthError::PasswordHashError(e.to_string()))?;

        Ok(hash.to_string())
    }

    /// Hash password without validation (for migrations, etc.)
    pub fn hash_password_unchecked(&self, password: &str) -> AuthResult<String> {
        let salt = SaltString::generate(&mut OsRng);

        let hash = self
            .argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| AuthError::PasswordHashError(e.to_string()))?;

        Ok(hash.to_string())
    }

    /// Verify a password against a hash
    pub fn verify_password(&self, password: &str, hash: &str) -> AuthResult<bool> {
        let parsed_hash =
            PasswordHash::new(hash).map_err(|e| AuthError::PasswordHashError(e.to_string()))?;

        match self
            .argon2
            .verify_password(password.as_bytes(), &parsed_hash)
        {
            Ok(()) => Ok(true),
            Err(argon2::password_hash::Error::Password) => Ok(false),
            Err(e) => Err(AuthError::PasswordHashError(e.to_string())),
        }
    }

    /// Check if hash needs rehashing (params changed)
    pub fn needs_rehash(&self, hash: &str) -> bool {
        if let Ok(parsed_hash) = PasswordHash::new(hash) {
            // Check if the hash uses different parameters
            // In a real implementation, you'd compare the params
            // For now, we assume newer hashes are fine
            parsed_hash.algorithm != argon2::ARGON2ID_IDENT
        } else {
            true // Invalid hash, needs rehash
        }
    }

    /// Calculate password strength score (0-100)
    pub fn calculate_strength(&self, password: &str) -> u8 {
        let mut score = 0u8;

        // Length contribution (up to 30 points)
        let len = password.len();
        score += (len.min(30) as u8).saturating_mul(1);

        // Character diversity (up to 40 points)
        let has_lower = password.chars().any(|c| c.is_lowercase());
        let has_upper = password.chars().any(|c| c.is_uppercase());
        let has_digit = password.chars().any(|c| c.is_ascii_digit());
        let has_special = password.chars().any(|c| !c.is_alphanumeric());

        if has_lower {
            score = score.saturating_add(10);
        }
        if has_upper {
            score = score.saturating_add(10);
        }
        if has_digit {
            score = score.saturating_add(10);
        }
        if has_special {
            score = score.saturating_add(10);
        }

        // Unique character ratio (up to 30 points)
        let unique: std::collections::HashSet<char> = password.chars().collect();
        let uniqueness = (unique.len() as f32 / password.len().max(1) as f32) * 30.0;
        score = score.saturating_add(uniqueness as u8);

        score.min(100)
    }

    /// Generate a secure random password
    pub fn generate_password(&self, length: usize) -> String {
        use rand::Rng;

        const LOWERCASE: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
        const UPPERCASE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        const DIGITS: &[u8] = b"0123456789";
        const SPECIAL: &[u8] = b"!@#$%^&*()_+-=[]{}|;:,.<>?";

        let mut rng = rand::thread_rng();
        let mut password = Vec::with_capacity(length);

        // Ensure at least one of each required type
        password.push(LOWERCASE[rng.gen_range(0..LOWERCASE.len())]);
        password.push(UPPERCASE[rng.gen_range(0..UPPERCASE.len())]);
        password.push(DIGITS[rng.gen_range(0..DIGITS.len())]);
        password.push(SPECIAL[rng.gen_range(0..SPECIAL.len())]);

        // Fill the rest
        let all_chars: Vec<u8> = [LOWERCASE, UPPERCASE, DIGITS, SPECIAL].concat();
        for _ in 4..length {
            password.push(all_chars[rng.gen_range(0..all_chars.len())]);
        }

        // Shuffle
        use rand::seq::SliceRandom;
        password.shuffle(&mut rng);

        String::from_utf8(password).expect("Generated password should be valid UTF-8")
    }
}

impl Default for PasswordService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_and_verify() {
        let service = PasswordService::new();
        let password = "SecureP@ssw0rd123!";

        let hash = service
            .hash_password(password)
            .expect("Hashing should succeed");
        assert!(hash.starts_with("$argon2id$"));

        let verified = service
            .verify_password(password, &hash)
            .expect("Verification should succeed");
        assert!(verified);

        let wrong_verified = service
            .verify_password("wrong_password", &hash)
            .expect("Verification should succeed");
        assert!(!wrong_verified);
    }

    #[test]
    fn test_password_validation() {
        let service = PasswordService::new();

        // Too short
        assert!(service.validate_password("Short1!").is_err());

        // No uppercase
        assert!(service.validate_password("lowercase123!@#").is_err());

        // No lowercase
        assert!(service.validate_password("UPPERCASE123!@#").is_err());

        // No digit
        assert!(service.validate_password("NoDigitPass!@#").is_err());

        // No special
        assert!(service.validate_password("NoSpecialPass123").is_err());

        // Valid password
        assert!(service.validate_password("ValidP@ssw0rd123").is_ok());
    }

    #[test]
    fn test_password_strength() {
        let service = PasswordService::new();

        // Weak password
        let weak_score = service.calculate_strength("abc");
        assert!(weak_score < 30);

        // Strong password
        let strong_score = service.calculate_strength("MyStr0ng!P@ssw0rd#2024");
        assert!(strong_score > 70);
    }

    #[test]
    fn test_generate_password() {
        let service = PasswordService::new();
        let password = service.generate_password(20);

        assert_eq!(password.len(), 20);
        assert!(service.validate_password(&password).is_ok());
    }
}
