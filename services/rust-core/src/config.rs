//! Application Configuration
//!
//! Handles loading configuration from environment variables and config files.

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub neo4j: Neo4jConfig,
    pub redis: RedisConfig,
    pub nats: NatsConfig,
    pub auth: AuthConfig,
    pub features: FeatureFlags,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub env: String,
    #[serde(default)]
    pub jwt_secret: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,

    // Computed URLs for the db module
    #[serde(skip)]
    pub postgres_url: String,
    #[serde(skip)]
    pub neo4j_url: String,
    #[serde(skip)]
    pub redis_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RedisConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NatsConfig {
    pub url: String,
    pub cluster_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    pub jwt_secret: String,
    pub jwt_expires_in: String,
    pub refresh_token_expires_in: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeatureFlags {
    pub enable_3d_engine: bool,
    pub enable_4d_timeline: bool,
    pub enable_ai_copilot: bool,
    pub enable_discovery_engine: bool,
    pub enable_legal_fortress: bool,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        // Load .env file if present
        dotenvy::dotenv().ok();

        let config = config::Config::builder()
            // Server defaults
            .set_default("server.host", "0.0.0.0")?
            .set_default("server.port", 8080)?
            .set_default("server.env", "development")?
            // Database defaults
            .set_default("database.max_connections", 10)?
            .set_default("database.min_connections", 2)?
            // Feature flags defaults
            .set_default("features.enable_3d_engine", true)?
            .set_default("features.enable_4d_timeline", true)?
            .set_default("features.enable_ai_copilot", true)?
            .set_default("features.enable_discovery_engine", true)?
            .set_default("features.enable_legal_fortress", false)?
            // Load from environment
            .add_source(
                config::Environment::default()
                    .prefix("NEURECTOMY")
                    .separator("__")
                    .try_parsing(true),
            )
            // Also support standard env vars
            .set_override_option("database.url", std::env::var("DATABASE_URL").ok())?
            .set_override_option("neo4j.uri", std::env::var("NEO4J_URI").ok())?
            .set_override_option("neo4j.user", std::env::var("NEO4J_USER").ok())?
            .set_override_option("neo4j.password", std::env::var("NEO4J_PASSWORD").ok())?
            .set_override_option("redis.url", std::env::var("REDIS_URL").ok())?
            .set_override_option("nats.url", std::env::var("NATS_URL").ok())?
            .set_override_option("nats.cluster_id", std::env::var("NATS_CLUSTER_ID").ok())?
            .set_override_option("auth.jwt_secret", std::env::var("JWT_SECRET").ok())?
            .set_override_option("auth.jwt_expires_in", std::env::var("JWT_EXPIRES_IN").ok())?
            .set_override_option(
                "auth.refresh_token_expires_in",
                std::env::var("REFRESH_TOKEN_EXPIRES_IN").ok(),
            )?
            .build()?;

        let mut app_config: AppConfig = config.try_deserialize()?;

        // Set computed database URLs
        app_config.database.postgres_url = app_config.database.url.clone();
        app_config.database.neo4j_url = format!(
            "neo4j://{}:{}@{}",
            app_config.neo4j.user,
            app_config.neo4j.password,
            app_config
                .neo4j
                .uri
                .trim_start_matches("bolt://")
                .trim_start_matches("neo4j://")
        );
        app_config.database.redis_url = app_config.redis.url.clone();

        Ok(app_config)
    }
}
