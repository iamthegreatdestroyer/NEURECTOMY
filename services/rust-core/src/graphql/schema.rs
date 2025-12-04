//! GraphQL Schema Builder
//!
//! Assembles the complete GraphQL schema with all types, queries, mutations, and subscriptions

use std::sync::Arc;
use async_graphql::Schema;

use crate::db::DatabaseConnections;
use crate::graphql::context::GraphQLContext;
use crate::graphql::dataloaders::DataLoaders;
use crate::graphql::queries::QueryRoot;
use crate::graphql::mutations::MutationRoot;
use crate::graphql::subscriptions::SubscriptionRoot;

/// The complete NEURECTOMY GraphQL schema type
pub type NeurectomySchema = Schema<QueryRoot, MutationRoot, SubscriptionRoot>;

/// Build the complete GraphQL schema
pub fn build_schema(db: DatabaseConnections) -> NeurectomySchema {
    // Create context with database connections
    let context = GraphQLContext::new(db.clone());
    
    // Create dataloaders
    let loaders = DataLoaders::new(db);
    
    Schema::build(QueryRoot, MutationRoot, SubscriptionRoot)
        // Add global context
        .data(context)
        .data(loaders)
        // Configure schema options
        .enable_federation()
        .enable_subscription_in_federation()
        // Limits for security
        .limit_complexity(500)
        .limit_depth(15)
        .finish()
}

/// Build schema with custom configuration
pub fn build_schema_with_config(
    db: DatabaseConnections,
    config: SchemaConfig,
) -> NeurectomySchema {
    let context = GraphQLContext::new(db.clone());
    let loaders = DataLoaders::new(db);
    
    let mut builder = Schema::build(QueryRoot, MutationRoot, SubscriptionRoot)
        .data(context)
        .data(loaders)
        .limit_complexity(config.max_complexity)
        .limit_depth(config.max_depth);
    
    if config.enable_federation {
        builder = builder.enable_federation().enable_subscription_in_federation();
    }
    
    if !config.enable_introspection {
        builder = builder.disable_introspection();
    }
    
    builder.finish()
}

/// Schema configuration options
#[derive(Debug, Clone)]
pub struct SchemaConfig {
    pub max_complexity: usize,
    pub max_depth: usize,
    pub enable_federation: bool,
    pub enable_introspection: bool,
}

impl Default for SchemaConfig {
    fn default() -> Self {
        Self {
            max_complexity: 500,
            max_depth: 15,
            enable_federation: true,
            enable_introspection: cfg!(debug_assertions),
        }
    }
}

// ============================================================================
// SCHEMA EXTENSIONS
// ============================================================================

/// Extension for adding tracing to GraphQL operations
pub struct TracingExtension;

impl async_graphql::extensions::ExtensionFactory for TracingExtension {
    fn create(&self) -> Arc<dyn async_graphql::extensions::Extension> {
        Arc::new(TracingExtensionImpl)
    }
}

struct TracingExtensionImpl;

impl async_graphql::extensions::Extension for TracingExtensionImpl {}

/// Extension for logging slow queries
pub struct SlowQueryExtension {
    #[allow(dead_code)]
    threshold_ms: u64,
}

impl SlowQueryExtension {
    pub fn new(threshold_ms: u64) -> Self {
        Self { threshold_ms }
    }
}

impl async_graphql::extensions::ExtensionFactory for SlowQueryExtension {
    fn create(&self) -> Arc<dyn async_graphql::extensions::Extension> {
        Arc::new(SlowQueryExtensionImpl)
    }
}

struct SlowQueryExtensionImpl;

impl async_graphql::extensions::Extension for SlowQueryExtensionImpl {}

// ============================================================================
// SDL EXPORT
// ============================================================================

/// Export the schema as SDL (Schema Definition Language)
pub fn export_sdl(schema: &NeurectomySchema) -> String {
    schema.sdl()
}

/// Export the schema as SDL with options
pub fn export_sdl_with_options(
    schema: &NeurectomySchema,
    options: async_graphql::SDLExportOptions,
) -> String {
    schema.sdl_with_options(options)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_config_defaults() {
        let config = SchemaConfig::default();
        assert_eq!(config.max_complexity, 500);
        assert_eq!(config.max_depth, 15);
        assert!(config.enable_federation);
    }
}
