import { GraphQLClient as GQLClient, Variables } from "graphql-request";
import { createClient as createWSClient, Client as WSClient } from "graphql-ws";

export interface GraphQLClientConfig {
  endpoint: string;
  wsEndpoint?: string;
  headers?: Record<string, string>;
  timeout?: number;
}

/**
 * Type-safe GraphQL client with subscription support.
 */
export class GraphQLClient {
  private client: GQLClient;
  private wsClient: WSClient | null = null;
  private config: GraphQLClientConfig;

  constructor(config: GraphQLClientConfig) {
    this.config = config;
    this.client = new GQLClient(config.endpoint, {
      headers: config.headers,
      // Note: timeout is handled via fetch options or AbortController if needed
    });

    if (config.wsEndpoint) {
      this.wsClient = createWSClient({
        url: config.wsEndpoint,
        connectionParams: {
          headers: config.headers,
        },
      });
    }
  }

  /**
   * Execute a GraphQL query.
   */
  async query<TData, TVariables extends Variables = Variables>(
    document: string,
    variables?: TVariables
  ): Promise<TData> {
    return this.client.request<TData>(document, variables);
  }

  /**
   * Execute a GraphQL mutation.
   */
  async mutate<TData, TVariables extends Variables = Variables>(
    document: string,
    variables?: TVariables
  ): Promise<TData> {
    return this.client.request<TData>(document, variables);
  }

  /**
   * Subscribe to a GraphQL subscription.
   */
  subscribe<TData, TVariables extends Variables = Variables>(
    document: string,
    onData: (data: TData) => void,
    variables?: TVariables,
    onError?: (error: Error) => void
  ): () => void {
    if (!this.wsClient) {
      throw new Error("WebSocket client not configured");
    }

    const unsubscribe = this.wsClient.subscribe<TData>(
      {
        query: document,
        variables: variables as Record<string, unknown>,
      },
      {
        next: (result) => {
          if (result.data) {
            onData(result.data);
          }
        },
        error: (err) => {
          if (onError) {
            onError(err instanceof Error ? err : new Error(String(err)));
          }
        },
        complete: () => {},
      }
    );

    return unsubscribe;
  }

  /**
   * Set authorization header.
   */
  setAuthToken(token: string): void {
    this.client.setHeader("Authorization", `Bearer ${token}`);
  }

  /**
   * Clear authorization header.
   */
  clearAuthToken(): void {
    this.client.setHeader("Authorization", "");
  }

  /**
   * Dispose of WebSocket connection.
   */
  dispose(): void {
    if (this.wsClient) {
      this.wsClient.dispose();
    }
  }
}

/**
 * Factory function to create a GraphQL client.
 */
export function createGraphQLClient(
  config: GraphQLClientConfig
): GraphQLClient {
  return new GraphQLClient(config);
}
