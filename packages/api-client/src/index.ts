/**
 * @neurectomy/api-client - GraphQL & REST API Client
 *
 * Provides type-safe API clients for communicating with Neurectomy services,
 * including GraphQL queries/mutations and REST endpoints.
 */

// Export GraphQL client
export {
  GraphQLClient,
  createGraphQLClient,
  type GraphQLClientConfig,
} from "./graphql";

// Export REST client
export { RestClient, createRestClient, type RestClientConfig } from "./rest";

// Export hooks for React integration
export {
  useQuery,
  useMutation,
  useSubscription,
  type QueryOptions,
  type MutationOptions,
  type SubscriptionOptions,
} from "./hooks";

// Export utilities
export { handleApiError, retryRequest, type ApiError } from "./utils";

// Re-export types
export type { Agent, Task, Workflow, ExecutionResult } from "./types";
