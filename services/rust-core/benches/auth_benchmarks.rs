//! Authentication Performance Benchmarks
//!
//! This module contains benchmarks for measuring authentication operations.
//! Run with: cargo bench --bench auth_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Authentication performance targets
mod targets {
    /// Password hashing should complete within this time (ms)
    /// Note: intentionally slow for security, but bounded
    pub const PASSWORD_HASH_MAX_MS: u64 = 500;

    /// JWT token generation target (ms)
    pub const JWT_GENERATE_TARGET_MS: u64 = 5;

    /// JWT token validation target (ms)
    pub const JWT_VALIDATE_TARGET_MS: u64 = 2;

    /// API key validation target (microseconds)
    pub const API_KEY_VALIDATE_TARGET_US: u64 = 100;
}

/// Benchmark password hashing with Argon2id
fn bench_password_hashing(c: &mut Criterion) {
    use argon2::{
        password_hash::{PasswordHasher, SaltString},
        Argon2, Params,
    };
    use rand::rngs::OsRng;

    let mut group = c.benchmark_group("password_hashing");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10); // Fewer samples since hashing is slow

    // Passwords of varying complexity
    let passwords = vec![
        ("short", "pass123"),
        ("medium", "MySecurePass123!"),
        (
            "long",
            "ThisIsAVeryLongPasswordWithManyCharacters!@#$%^&*()",
        ),
    ];

    // Standard Argon2id parameters (OWASP recommendations)
    let standard_params = Params::new(19456, 2, 1, None).unwrap();
    let argon2_standard = Argon2::new(
        argon2::Algorithm::Argon2id,
        argon2::Version::V0x13,
        standard_params,
    );

    for (name, password) in &passwords {
        group.bench_with_input(
            BenchmarkId::new("argon2id_hash", name),
            password,
            |b, pwd| {
                b.iter(|| {
                    let salt = SaltString::generate(&mut OsRng);
                    argon2_standard
                        .hash_password(black_box(pwd.as_bytes()), &salt)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark password verification
fn bench_password_verification(c: &mut Criterion) {
    use argon2::{
        password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
        Argon2, Params,
    };
    use rand::rngs::OsRng;

    let mut group = c.benchmark_group("password_verification");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    let password = "MySecurePassword123!";
    let params = Params::new(19456, 2, 1, None).unwrap();
    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);

    // Pre-generate hash for verification benchmark
    let salt = SaltString::generate(&mut OsRng);
    let hash = argon2.hash_password(password.as_bytes(), &salt).unwrap();
    let hash_string = hash.to_string();

    group.bench_function("argon2id_verify_correct", |b| {
        b.iter(|| {
            let parsed_hash = PasswordHash::new(&hash_string).unwrap();
            argon2
                .verify_password(black_box(password.as_bytes()), &parsed_hash)
                .is_ok()
        });
    });

    group.bench_function("argon2id_verify_incorrect", |b| {
        let wrong_password = "WrongPassword123!";
        b.iter(|| {
            let parsed_hash = PasswordHash::new(&hash_string).unwrap();
            argon2
                .verify_password(black_box(wrong_password.as_bytes()), &parsed_hash)
                .is_err()
        });
    });

    group.finish();
}

/// Benchmark JWT token operations
fn bench_jwt_operations(c: &mut Criterion) {
    use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct Claims {
        sub: String,
        exp: usize,
        iat: usize,
        role: String,
        permissions: Vec<String>,
    }

    let mut group = c.benchmark_group("jwt_operations");
    group.measurement_time(Duration::from_secs(10));

    let secret = "super_secret_key_for_jwt_signing_must_be_at_least_32_bytes";
    let encoding_key = EncodingKey::from_secret(secret.as_bytes());
    let decoding_key = DecodingKey::from_secret(secret.as_bytes());

    // Standard claims
    let standard_claims = Claims {
        sub: "user-123e4567-e89b-12d3-a456-426614174000".to_string(),
        exp: 9999999999,
        iat: 1700000000,
        role: "admin".to_string(),
        permissions: vec![
            "read:agents".to_string(),
            "write:agents".to_string(),
            "delete:agents".to_string(),
        ],
    };

    // Large claims (many permissions)
    let large_claims = Claims {
        sub: "user-123e4567-e89b-12d3-a456-426614174000".to_string(),
        exp: 9999999999,
        iat: 1700000000,
        role: "superadmin".to_string(),
        permissions: (0..50).map(|i| format!("permission:{}", i)).collect(),
    };

    // Benchmark token generation
    group.bench_function("jwt_encode_standard", |b| {
        b.iter(|| {
            encode(
                black_box(&Header::default()),
                black_box(&standard_claims),
                black_box(&encoding_key),
            )
            .unwrap()
        });
    });

    group.bench_function("jwt_encode_large_claims", |b| {
        b.iter(|| {
            encode(
                black_box(&Header::default()),
                black_box(&large_claims),
                black_box(&encoding_key),
            )
            .unwrap()
        });
    });

    // Pre-generate tokens for decode benchmarks
    let standard_token = encode(&Header::default(), &standard_claims, &encoding_key).unwrap();
    let large_token = encode(&Header::default(), &large_claims, &encoding_key).unwrap();

    group.bench_function("jwt_decode_standard", |b| {
        b.iter(|| {
            decode::<Claims>(
                black_box(&standard_token),
                black_box(&decoding_key),
                black_box(&Validation::default()),
            )
            .unwrap()
        });
    });

    group.bench_function("jwt_decode_large_claims", |b| {
        b.iter(|| {
            decode::<Claims>(
                black_box(&large_token),
                black_box(&decoding_key),
                black_box(&Validation::default()),
            )
            .unwrap()
        });
    });

    // Benchmark validation only (no decode)
    group.bench_function("jwt_validate_signature", |b| {
        let validation = Validation::default();
        b.iter(|| {
            decode::<Claims>(
                black_box(&standard_token),
                black_box(&decoding_key),
                black_box(&validation),
            )
            .is_ok()
        });
    });

    group.finish();
}

/// Benchmark API key operations
fn bench_api_key_operations(c: &mut Criterion) {
    use sha2::{Digest, Sha256};

    let mut group = c.benchmark_group("api_key_operations");
    group.measurement_time(Duration::from_secs(10));

    // Simulate API key generation
    group.bench_function("api_key_generate", |b| {
        b.iter(|| {
            let key_bytes: [u8; 32] = rand::random();
            let key = hex::encode(black_box(key_bytes));
            key
        });
    });

    // Simulate API key hashing (for storage)
    let api_key = "nrctmy_live_1234567890abcdef1234567890abcdef";

    group.bench_function("api_key_hash_sha256", |b| {
        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(api_key.as_bytes()));
            let result = hasher.finalize();
            hex::encode(result)
        });
    });

    // Simulate API key validation (constant-time comparison)
    let stored_hash = {
        let mut hasher = Sha256::new();
        hasher.update(api_key.as_bytes());
        hex::encode(hasher.finalize())
    };

    group.bench_function("api_key_validate", |b| {
        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(api_key.as_bytes()));
            let computed = hex::encode(hasher.finalize());
            // Constant-time comparison
            computed == stored_hash
        });
    });

    // Benchmark API key format validation
    group.bench_function("api_key_format_check", |b| {
        let keys = vec![
            "nrctmy_live_1234567890abcdef1234567890abcdef",
            "nrctmy_test_1234567890abcdef1234567890abcdef",
            "invalid_key",
        ];
        b.iter(|| {
            for key in &keys {
                let valid = key.starts_with("nrctmy_") && key.len() == 46;
                black_box(valid);
            }
        });
    });

    group.finish();
}

/// Benchmark session token operations
fn bench_session_operations(c: &mut Criterion) {
    use sha2::{Digest, Sha256};

    let mut group = c.benchmark_group("session_operations");
    group.measurement_time(Duration::from_secs(10));

    // Session ID generation
    group.bench_function("session_id_generate", |b| {
        b.iter(|| {
            let id = uuid::Uuid::new_v4();
            black_box(id.to_string())
        });
    });

    // Session token generation (more complex)
    group.bench_function("session_token_generate", |b| {
        b.iter(|| {
            let session_id = uuid::Uuid::new_v4();
            let random_bytes: [u8; 16] = rand::random();
            let timestamp = chrono::Utc::now().timestamp();

            let mut hasher = Sha256::new();
            hasher.update(session_id.as_bytes());
            hasher.update(&random_bytes);
            hasher.update(&timestamp.to_le_bytes());

            let token = hex::encode(hasher.finalize());
            black_box(token)
        });
    });

    // Session lookup simulation (hash table lookup)
    use std::collections::HashMap;

    let mut sessions: HashMap<String, String> = HashMap::new();
    for i in 0..1000 {
        sessions.insert(uuid::Uuid::new_v4().to_string(), format!("user_{}", i));
    }

    let existing_key = sessions.keys().next().unwrap().clone();
    let nonexistent_key = uuid::Uuid::new_v4().to_string();

    group.bench_function("session_lookup_existing", |b| {
        b.iter(|| sessions.get(black_box(&existing_key)));
    });

    group.bench_function("session_lookup_nonexistent", |b| {
        b.iter(|| sessions.get(black_box(&nonexistent_key)));
    });

    group.finish();
}

/// Benchmark permission checking
fn bench_permission_operations(c: &mut Criterion) {
    use std::collections::HashSet;

    let mut group = c.benchmark_group("permission_operations");
    group.measurement_time(Duration::from_secs(10));

    // User permissions (typical user)
    let user_permissions: HashSet<&str> = [
        "read:agents",
        "write:agents",
        "read:executions",
        "read:workflows",
    ]
    .into_iter()
    .collect();

    // Admin permissions (many permissions)
    let admin_permissions: HashSet<&str> = (0..50)
        .map(|i| match i % 5 {
            0 => "read:agents",
            1 => "write:agents",
            2 => "delete:agents",
            3 => "admin:users",
            _ => "admin:system",
        })
        .collect();

    // Required permissions for an action
    let required_single = ["read:agents"];
    let required_multiple = ["read:agents", "write:agents", "read:executions"];

    group.bench_function("permission_check_single", |b| {
        b.iter(|| {
            required_single
                .iter()
                .all(|p| user_permissions.contains(black_box(p)))
        });
    });

    group.bench_function("permission_check_multiple", |b| {
        b.iter(|| {
            required_multiple
                .iter()
                .all(|p| user_permissions.contains(black_box(p)))
        });
    });

    group.bench_function("permission_check_admin", |b| {
        b.iter(|| {
            required_multiple
                .iter()
                .all(|p| admin_permissions.contains(black_box(p)))
        });
    });

    // Role-based permission lookup
    use std::collections::HashMap;

    let role_permissions: HashMap<&str, Vec<&str>> = [
        ("viewer", vec!["read:agents", "read:executions"]),
        (
            "editor",
            vec![
                "read:agents",
                "write:agents",
                "read:executions",
                "write:executions",
            ],
        ),
        (
            "admin",
            vec![
                "read:agents",
                "write:agents",
                "delete:agents",
                "read:executions",
                "write:executions",
                "delete:executions",
                "admin:users",
            ],
        ),
    ]
    .into_iter()
    .collect();

    group.bench_function("role_permission_lookup", |b| {
        let roles = ["viewer", "editor", "admin"];
        b.iter(|| {
            for role in &roles {
                black_box(role_permissions.get(black_box(role)));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_password_hashing,
    bench_password_verification,
    bench_jwt_operations,
    bench_api_key_operations,
    bench_session_operations,
    bench_permission_operations,
);

criterion_main!(benches);
