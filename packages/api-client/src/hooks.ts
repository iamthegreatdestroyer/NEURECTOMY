import {
  useQuery as useRQQuery,
  useMutation as useRQMutation,
  type UseQueryOptions,
  type UseMutationOptions,
} from "@tanstack/react-query";
import { useState, useEffect, useCallback } from "react";

export interface QueryOptions<TData> extends Omit<
  UseQueryOptions<TData>,
  "queryKey" | "queryFn"
> {
  key: string[];
}

export interface MutationOptions<TData, TVariables> extends Omit<
  UseMutationOptions<TData, Error, TVariables>,
  "mutationFn"
> {}

export interface SubscriptionOptions<TData> {
  onData?: (data: TData) => void;
  onError?: (error: Error) => void;
  enabled?: boolean;
}

/**
 * Hook for GraphQL/REST queries with React Query integration.
 */
export function useQuery<TData>(
  queryKey: string[],
  queryFn: () => Promise<TData>,
  options?: Partial<QueryOptions<TData>>
) {
  return useRQQuery<TData>({
    queryKey,
    queryFn,
    ...options,
  });
}

/**
 * Hook for GraphQL/REST mutations with React Query integration.
 */
export function useMutation<TData, TVariables>(
  mutationFn: (variables: TVariables) => Promise<TData>,
  options?: MutationOptions<TData, TVariables>
) {
  return useRQMutation<TData, Error, TVariables>({
    mutationFn,
    ...options,
  });
}

/**
 * Hook for GraphQL subscriptions.
 */
export function useSubscription<TData>(
  subscribe: (
    onData: (data: TData) => void,
    onError: (error: Error) => void
  ) => () => void,
  options?: SubscriptionOptions<TData>
) {
  const [data, setData] = useState<TData | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const handleData = useCallback(
    (newData: TData) => {
      setData(newData);
      setError(null);
      setIsConnected(true);
      options?.onData?.(newData);
    },
    [options]
  );

  const handleError = useCallback(
    (err: Error) => {
      setError(err);
      setIsConnected(false);
      options?.onError?.(err);
    },
    [options]
  );

  useEffect(() => {
    if (options?.enabled === false) {
      return;
    }

    const unsubscribe = subscribe(handleData, handleError);
    setIsConnected(true);

    return () => {
      unsubscribe();
      setIsConnected(false);
    };
  }, [subscribe, handleData, handleError, options?.enabled]);

  return { data, error, isConnected };
}
