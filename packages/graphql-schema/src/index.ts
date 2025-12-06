/**
 * @neurectomy/graphql-schema
 *
 * Unified GraphQL schema package for NEURECTOMY.
 * Contains schema definitions, generated types, and utilities.
 */

// Re-export generated types when available
export * from "../generated/typescript";

// Schema utilities
export { gql } from "graphql-tag";

// Version info
export const SCHEMA_VERSION = "0.1.0";

// Schema governance version
export const GOVERNANCE_VERSION = "1.0.0";

// Schema file paths for tooling
export const SCHEMA_PATHS = {
  root: "./schema",
  subscriptions: "./schema/subscriptions",
  mutations: "./schema/mutations",
  types: "./schema/types",
  inputs: "./schema/inputs",
  versioning: "./schema/versioning.graphql",
  reliableDelivery:
    "./schema/subscriptions/reliable-delivery.subscription.graphql",
} as const;

// Feature flags for schema capabilities
export const SCHEMA_FEATURES = {
  subscriptions: true,
  reliableDelivery: true,
  schemaVersioning: true,
  deprecationTracking: true,
  backpressure: true,
} as const;
