/**
 * useSubscription Hook
 *
 * Custom React hook for GraphQL subscriptions with TypeScript support.
 * Wraps URQL's useSubscription with enhanced connection management and error handling.
 */

// import {
//   useSubscription as useUrqlSubscription,
//   UseSubscriptionArgs,
// } from "urql";
import { useEffect, useRef, useState } from "react";

export interface UseSubscriptionOptions<TData = any, TVariables = object> {
  query: string;
  variables?: TVariables;
  pause?: boolean;
  context?: any;
  onData?: (data: TData) => void;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export interface UseSubscriptionResult<TData = any> {
  data: TData | undefined;
  fetching: boolean;
  error: Error | undefined;
  stale: boolean;
  connected: boolean;
  executeSubscription: (
    opts?: Partial<UseSubscriptionArgs<TVariables, TData>>
  ) => void;
}

/**
 * Custom useSubscription hook with enhanced features
 *
 * @example
 * ```tsx
 * const { data, connected, error } = useSubscription<AgentStatusUpdate>({
 *   query: `
 *     subscription OnAgentStatusChange($agentId: ID!) {
 *       agentStatusChanged(agentId: $agentId) {
 *         id
 *         status
 *         metrics {
 *           executionTime
 *           tokensUsed
 *         }
 *         timestamp
 *       }
 *     }
 *   `,
 *   variables: { agentId: '123' },
 *   onData: (data) => {
 *     console.log('Agent status updated:', data.agentStatusChanged);
 *     updateAgentInStore(data.agentStatusChanged);
 *   },
 *   onConnect: () => {
 *     console.log('WebSocket connected');
 *   },
 *   onError: (error) => {
 *     console.error('Subscription error:', error);
 *     toast.error('Lost connection to server');
 *   },
 * });
 * ```
 */
export function useSubscription<
  TData = any,
  TVariables extends object = object,
>({
  query,
  variables,
  pause = false,
  context,
  onData,
  onError,
  onConnect,
  onDisconnect,
}: UseSubscriptionOptions<TData, TVariables>): UseSubscriptionResult<TData> {
  // Track connection state
  const [connected, setConnected] = useState(false);

  // Use URQL's useSubscription with custom handler
  const handleSubscription = useRef(
    (_prev: TData | undefined, data: TData) => data
  ).current;

  const [result, executeSubscription] = useUrqlSubscription<TData, TVariables>(
    {
      query,
      variables,
      pause,
      context,
    },
    handleSubscription
  );

  // Track previous data to detect new subscription data
  const prevDataRef = useRef<TData | undefined>();

  // Call onData when new data arrives
  useEffect(() => {
    if (
      result.data &&
      result.data !== prevDataRef.current &&
      !result.fetching
    ) {
      prevDataRef.current = result.data;
      onData?.(result.data);
    }
  }, [result.data, result.fetching, onData]);

  // Call onError when error occurs
  useEffect(() => {
    if (result.error && !result.fetching) {
      const error = new Error(result.error.message);
      error.name = result.error.name;
      (error as any).graphQLErrors = result.error.graphQLErrors;
      (error as any).networkError = result.error.networkError;
      onError?.(error);
      setConnected(false);
    }
  }, [result.error, result.fetching, onError]);

  // Track connection state based on fetching and error
  useEffect(() => {
    const isConnected = !pause && !result.error && result.fetching;

    if (isConnected !== connected) {
      setConnected(isConnected);

      if (isConnected) {
        onConnect?.();
      } else {
        onDisconnect?.();
      }
    }
  }, [
    pause,
    result.error,
    result.fetching,
    connected,
    onConnect,
    onDisconnect,
  ]);

  return {
    data: result.data,
    fetching: result.fetching,
    error: result.error as Error | undefined,
    stale: result.stale,
    connected,
    executeSubscription,
  };
}

/**
 * Hook for subscriptions with automatic reconnection logic
 *
 * @example
 * ```tsx
 * const { data, connected, reconnecting } = useAutoReconnectSubscription<MetricsUpdate>({
 *   query: METRICS_SUBSCRIPTION,
 *   variables: { clusterId: 'prod-cluster' },
 *   maxReconnectAttempts: 5,
 *   reconnectDelay: 2000,
 *   onData: (data) => updateMetrics(data),
 * });
 *
 * if (reconnecting) {
 *   return <div>Reconnecting to server...</div>;
 * }
 * ```
 */
export interface UseAutoReconnectSubscriptionOptions<
  TData = any,
  TVariables = object,
> extends UseSubscriptionOptions<TData, TVariables> {
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
}

export function useAutoReconnectSubscription<
  TData = any,
  TVariables extends object = object,
>({
  query,
  variables,
  pause = false,
  context,
  maxReconnectAttempts = 5,
  reconnectDelay = 2000,
  onData,
  onError,
  onConnect,
  onDisconnect,
}: UseAutoReconnectSubscriptionOptions<TData, TVariables>) {
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleError = (error: Error) => {
    onError?.(error);

    // Attempt reconnection if under max attempts
    if (reconnectAttempts < maxReconnectAttempts) {
      setIsReconnecting(true);

      reconnectTimeoutRef.current = setTimeout(
        () => {
          setReconnectAttempts((prev) => prev + 1);
          setIsReconnecting(false);
        },
        reconnectDelay * Math.pow(2, reconnectAttempts)
      ); // Exponential backoff
    }
  };

  const handleConnect = () => {
    setReconnectAttempts(0);
    setIsReconnecting(false);
    onConnect?.();
  };

  const result = useSubscription<TData, TVariables>({
    query,
    variables,
    pause,
    context,
    onData,
    onError: handleError,
    onConnect: handleConnect,
    onDisconnect,
  });

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    ...result,
    reconnecting: isReconnecting,
    reconnectAttempts,
  };
}

export type { UseSubscriptionArgs };
