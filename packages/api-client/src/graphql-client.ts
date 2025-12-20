/**
 * GraphQL API Client
 *
 * Centralized client for GraphQL queries, mutations, and subscriptions.
 * Includes request interceptors, error handling, and retry logic.
 */

// import {
//   Client,
//   cacheExchange,
//   fetchExchange,
//   subscriptionExchange,
//   CombinedError,
// } from "urql";
// import { createClient as createWSClient } from "graphql-ws";
// import { retryExchange } from "@urql/exchange-retry";
// import { authExchange } from "@urql/exchange-auth";
// import { devtoolsExchange } from "@urql/devtools";

// Environment configuration
const GRAPHQL_ENDPOINT =
  import.meta.env.VITE_GRAPHQL_ENDPOINT || "http://localhost:16080/graphql";
const WS_ENDPOINT =
  import.meta.env.VITE_WS_ENDPOINT || "ws://localhost:16080/graphql";

/**
 * WebSocket Client for Subscriptions
 */
const wsClient = createWSClient({
  url: WS_ENDPOINT,
  connectionParams: async () => {
    const token = await getAuthToken();
    return {
      authorization: token ? `Bearer ${token}` : "",
    };
  },
  retryAttempts: 5,
  shouldRetry: () => true,
  on: {
    connected: () => console.log("[GraphQL WS] Connected"),
    closed: () => console.log("[GraphQL WS] Closed"),
    error: (error) => console.error("[GraphQL WS] Error:", error),
  },
});

/**
 * Get authentication token from storage
 */
async function getAuthToken(): Promise<string | null> {
  // TODO: Implement token retrieval from Tauri store
  return localStorage.getItem("auth_token");
}

/**
 * Add authentication token to requests
 */
const auth = authExchange(async (utils) => {
  const token = await getAuthToken();

  return {
    addAuthToOperation(operation) {
      if (!token) return operation;

      return utils.appendHeaders(operation, {
        Authorization: `Bearer ${token}`,
      });
    },
    didAuthError(error: CombinedError) {
      // Check if error is authentication related
      return error.graphQLErrors.some(
        (e) => e.extensions?.code === "UNAUTHENTICATED"
      );
    },
    async refreshAuth() {
      // TODO: Implement token refresh logic
      console.warn("[GraphQL] Token refresh not implemented");
    },
  };
});

/**
 * Retry configuration for failed requests
 */
const retry = retryExchange({
  initialDelayMs: 1000,
  maxDelayMs: 15000,
  randomDelay: true,
  maxNumberAttempts: 3,
  retryIf: (error) => {
    // Retry on network errors or 5xx server errors
    return !!(error && error.networkError);
  },
});

/**
 * Create URQL Client
 */
export const graphqlClient = new Client({
  url: GRAPHQL_ENDPOINT,
  exchanges: [
    devtoolsExchange,
    cacheExchange,
    retry,
    auth,
    fetchExchange,
    subscriptionExchange({
      forwardSubscription(request) {
        const input = { ...request, query: request.query || "" };
        return {
          subscribe(sink) {
            const unsubscribe = wsClient.subscribe(input, sink);
            return { unsubscribe };
          },
        };
      },
    }),
  ],
  fetchOptions: () => {
    return {
      headers: {
        "Content-Type": "application/json",
      },
    };
  },
});

/**
 * Query helper with error handling
 */
export async function query<TData = any, TVariables = object>(
  query: string,
  variables?: TVariables
): Promise<{ data: TData | null; error: CombinedError | null }> {
  try {
    const result = await graphqlClient.query(query, variables).toPromise();

    if (result.error) {
      console.error("[GraphQL Query Error]:", result.error);
      return { data: null, error: result.error };
    }

    return { data: result.data as TData, error: null };
  } catch (error) {
    console.error("[GraphQL Query Exception]:", error);
    return { data: null, error: error as CombinedError };
  }
}

/**
 * Mutation helper with error handling
 */
export async function mutation<TData = any, TVariables = object>(
  mutation: string,
  variables?: TVariables
): Promise<{ data: TData | null; error: CombinedError | null }> {
  try {
    const result = await graphqlClient
      .mutation(mutation, variables)
      .toPromise();

    if (result.error) {
      console.error("[GraphQL Mutation Error]:", result.error);
      return { data: null, error: result.error };
    }

    return { data: result.data as TData, error: null };
  } catch (error) {
    console.error("[GraphQL Mutation Exception]:", error);
    return { data: null, error: error as CombinedError };
  }
}

/**
 * Subscription helper
 */
export function subscribe<TData = any, TVariables = object>(
  subscription: string,
  variables?: TVariables,
  onData?: (data: TData) => void,
  onError?: (error: CombinedError) => void
) {
  const { unsubscribe } = graphqlClient
    .subscription(subscription, variables)
    .subscribe((result) => {
      if (result.error) {
        console.error("[GraphQL Subscription Error]:", result.error);
        onError?.(result.error);
        return;
      }

      if (result.data) {
        onData?.(result.data as TData);
      }
    });

  return unsubscribe;
}

/**
 * Close WebSocket connection
 */
export function closeWebSocket() {
  wsClient.dispose();
}

/**
 * Reconnect WebSocket
 */
export function reconnectWebSocket() {
  wsClient.dispose();
  // WebSocket will automatically reconnect
}

// Export client for direct use
export { graphqlClient as client };
