/**
 * useQuery Hook
 *
 * Custom React hook for GraphQL queries with TypeScript support.
 * Wraps URQL's useQuery with enhanced error handling and loading states.
 */

// import { useQuery as useUrqlQuery, UseQueryArgs, UseQueryState } from "urql";
import { useEffect, useRef } from "react";

export interface UseQueryOptions<TData = any, TVariables = object> {
  query: string;
  variables?: TVariables;
  pause?: boolean;
  requestPolicy?:
    | "cache-first"
    | "cache-only"
    | "network-only"
    | "cache-and-network";
  context?: any;
  onSuccess?: (data: TData) => void;
  onError?: (error: Error) => void;
}

export interface UseQueryResult<TData = any> {
  data: TData | undefined;
  fetching: boolean;
  error: Error | undefined;
  stale: boolean;
  executeQuery: (opts?: Partial<UseQueryArgs<TVariables, TData>>) => void;
  refetch: () => void;
}

/**
 * Custom useQuery hook with enhanced features
 *
 * @example
 * ```tsx
 * const { data, fetching, error, refetch } = useQuery<User>({
 *   query: `
 *     query GetUser($id: ID!) {
 *       user(id: $id) {
 *         id
 *         name
 *         email
 *       }
 *     }
 *   `,
 *   variables: { id: '123' },
 *   onSuccess: (data) => console.log('User loaded:', data.user),
 *   onError: (error) => console.error('Failed to load user:', error),
 * });
 * ```
 */
export function useQuery<TData = any, TVariables extends object = object>({
  query,
  variables,
  pause = false,
  requestPolicy = "cache-first",
  context,
  onSuccess,
  onError,
}: UseQueryOptions<TData, TVariables>): UseQueryResult<TData> {
  // Use URQL's useQuery
  const [result, executeQuery] = useUrqlQuery<TData, TVariables>({
    query,
    variables,
    pause,
    requestPolicy,
    context,
  });

  // Track previous data to detect successful data fetches
  const prevDataRef = useRef<TData | undefined>();

  // Call onSuccess when data changes (successful fetch)
  useEffect(() => {
    if (
      result.data &&
      result.data !== prevDataRef.current &&
      !result.fetching
    ) {
      prevDataRef.current = result.data;
      onSuccess?.(result.data);
    }
  }, [result.data, result.fetching, onSuccess]);

  // Call onError when error occurs
  useEffect(() => {
    if (result.error && !result.fetching) {
      const error = new Error(result.error.message);
      error.name = result.error.name;
      (error as any).graphQLErrors = result.error.graphQLErrors;
      (error as any).networkError = result.error.networkError;
      onError?.(error);
    }
  }, [result.error, result.fetching, onError]);

  // Refetch function
  const refetch = () => {
    executeQuery({ requestPolicy: "network-only" });
  };

  return {
    data: result.data,
    fetching: result.fetching,
    error: result.error as Error | undefined,
    stale: result.stale,
    executeQuery,
    refetch,
  };
}

/**
 * Hook for queries that should run on demand (lazy queries)
 *
 * @example
 * ```tsx
 * const [executeSearch, { data, fetching }] = useLazyQuery<SearchResults>({
 *   query: SEARCH_QUERY,
 * });
 *
 * const handleSearch = (term: string) => {
 *   executeSearch({ variables: { term } });
 * };
 * ```
 */
export function useLazyQuery<TData = any, TVariables extends object = object>({
  query,
  requestPolicy = "network-only",
  context,
  onSuccess,
  onError,
}: Omit<UseQueryOptions<TData, TVariables>, "variables" | "pause">) {
  const result = useQuery<TData, TVariables>({
    query,
    pause: true,
    requestPolicy,
    context,
    onSuccess,
    onError,
  });

  const execute = (options?: { variables?: TVariables }) => {
    result.executeQuery({
      requestPolicy,
      ...options,
    } as any);
  };

  return [execute, result] as const;
}

export type { UseQueryArgs, UseQueryState };
