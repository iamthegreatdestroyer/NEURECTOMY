/**
 * @file Neo4j Graph Renderer
 * @description React component and hooks for Neo4j graph visualization
 * @module @neurectomy/3d-engine/visualization/graph/neo4j
 * @agents @VERTEX @CANVAS @APEX
 */

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  useMemo,
} from "react";
import {
  Neo4jGraphAdapter,
  type Neo4jConfig,
  type Neo4jGraphResult,
} from "./adapter";
import {
  Neo4jDataTransformer,
  type TransformOptions,
  type TransformedGraph,
} from "./transformer";
import {
  CypherQueryVisualizer,
  type CypherQuery,
  type QueryVisualization,
} from "./query-visualizer";
import type { GraphNode, GraphEdge } from "../types";

// ============================================================================
// Types
// ============================================================================

export interface Neo4jRenderOptions {
  /** Neo4j connection config */
  connection: Neo4jConfig;
  /** Data transformation options */
  transform?: TransformOptions;
  /** Auto-refresh interval in ms (0 = disabled) */
  refreshInterval?: number;
  /** Maximum nodes to fetch */
  maxNodes?: number;
  /** Enable real-time updates via polling */
  realtime?: boolean;
  /** Callback when data updates */
  onUpdate?: (data: TransformedGraph) => void;
  /** Callback on error */
  onError?: (error: Error) => void;
  /** Callback when connected */
  onConnected?: () => void;
  /** Callback when disconnected */
  onDisconnected?: () => void;
}

export interface Neo4jGraphState {
  /** Current graph data */
  data: TransformedGraph | null;
  /** Loading state */
  loading: boolean;
  /** Error state */
  error: Error | null;
  /** Connection state */
  connected: boolean;
  /** Last update timestamp */
  lastUpdate: number | null;
}

export interface Neo4jGraphActions {
  /** Connect to Neo4j */
  connect: () => Promise<void>;
  /** Disconnect from Neo4j */
  disconnect: () => Promise<void>;
  /** Refresh data */
  refresh: () => Promise<void>;
  /** Run custom Cypher query */
  runQuery: (
    query: string,
    params?: Record<string, unknown>
  ) => Promise<TransformedGraph>;
  /** Fetch subgraph around a node */
  fetchSubgraph: (
    nodeId: string | number,
    depth?: number
  ) => Promise<TransformedGraph>;
  /** Search nodes */
  searchNodes: (
    property: string,
    value: string,
    fuzzy?: boolean
  ) => Promise<GraphNode[]>;
  /** Visualize a query */
  visualizeQuery: (query: CypherQuery) => QueryVisualization;
}

// ============================================================================
// Hook: useNeo4jGraph
// ============================================================================

/**
 * React hook for Neo4j graph integration
 */
export function useNeo4jGraph(
  options: Neo4jRenderOptions
): [Neo4jGraphState, Neo4jGraphActions] {
  const [state, setState] = useState<Neo4jGraphState>({
    data: null,
    loading: false,
    error: null,
    connected: false,
    lastUpdate: null,
  });

  const adapterRef = useRef<Neo4jGraphAdapter | null>(null);
  const transformerRef = useRef<Neo4jDataTransformer | null>(null);
  const visualizerRef = useRef<CypherQueryVisualizer | null>(null);
  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );

  // Initialize adapter and transformer
  useEffect(() => {
    adapterRef.current = new Neo4jGraphAdapter(options.connection);
    transformerRef.current = new Neo4jDataTransformer(options.transform);
    visualizerRef.current = new CypherQueryVisualizer();

    return () => {
      if (adapterRef.current?.connected) {
        adapterRef.current.disconnect();
      }
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [options.connection.uri, options.connection.database]);

  // Update transformer options
  useEffect(() => {
    if (transformerRef.current && options.transform) {
      transformerRef.current.updateOptions(options.transform);
    }
  }, [options.transform]);

  // Connect to Neo4j
  const connect = useCallback(async () => {
    if (!adapterRef.current) return;

    setState((s) => ({ ...s, loading: true, error: null }));

    try {
      await adapterRef.current.connect();
      setState((s) => ({ ...s, connected: true }));
      options.onConnected?.();

      // Initial data fetch
      const result = await adapterRef.current.fetchGraph(
        options.maxNodes ?? 1000
      );
      const transformed = transformerRef.current!.transform(
        result.nodes,
        result.relationships
      );

      setState((s) => ({
        ...s,
        data: transformed,
        loading: false,
        lastUpdate: Date.now(),
      }));

      options.onUpdate?.(transformed);

      // Setup refresh interval if enabled
      if (options.refreshInterval && options.refreshInterval > 0) {
        refreshIntervalRef.current = setInterval(async () => {
          try {
            const newResult = await adapterRef.current!.fetchGraph(
              options.maxNodes ?? 1000
            );
            const newTransformed = transformerRef.current!.transform(
              newResult.nodes,
              newResult.relationships
            );

            setState((s) => ({
              ...s,
              data: newTransformed,
              lastUpdate: Date.now(),
            }));

            options.onUpdate?.(newTransformed);
          } catch (err) {
            console.error("[useNeo4jGraph] Refresh error:", err);
          }
        }, options.refreshInterval);
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      setState((s) => ({ ...s, loading: false, error: err }));
      options.onError?.(err);
    }
  }, [
    options.maxNodes,
    options.refreshInterval,
    options.onConnected,
    options.onUpdate,
    options.onError,
  ]);

  // Disconnect from Neo4j
  const disconnect = useCallback(async () => {
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current);
      refreshIntervalRef.current = null;
    }

    if (adapterRef.current?.connected) {
      await adapterRef.current.disconnect();
    }

    setState((s) => ({
      ...s,
      connected: false,
      data: null,
      lastUpdate: null,
    }));

    options.onDisconnected?.();
  }, [options.onDisconnected]);

  // Refresh data
  const refresh = useCallback(async () => {
    if (!adapterRef.current?.connected) {
      throw new Error("Not connected to Neo4j");
    }

    setState((s) => ({ ...s, loading: true }));

    try {
      const result = await adapterRef.current.fetchGraph(
        options.maxNodes ?? 1000
      );
      const transformed = transformerRef.current!.transform(
        result.nodes,
        result.relationships
      );

      setState((s) => ({
        ...s,
        data: transformed,
        loading: false,
        lastUpdate: Date.now(),
      }));

      options.onUpdate?.(transformed);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      setState((s) => ({ ...s, loading: false, error: err }));
      throw err;
    }
  }, [options.maxNodes, options.onUpdate]);

  // Run custom query
  const runQuery = useCallback(
    async (
      query: string,
      params?: Record<string, unknown>
    ): Promise<TransformedGraph> => {
      if (!adapterRef.current?.connected) {
        throw new Error("Not connected to Neo4j");
      }

      const result = await adapterRef.current.query<{
        n?: {
          identity: string | number;
          labels: string[];
          properties: Record<string, unknown>;
        };
        r?: {
          identity: string | number;
          type: string;
          startNodeId: string | number;
          endNodeId: string | number;
          properties: Record<string, unknown>;
        };
        m?: {
          identity: string | number;
          labels: string[];
          properties: Record<string, unknown>;
        };
      }>(query, params);

      if (result.error) {
        throw result.error;
      }

      // Extract nodes and relationships from query result
      const nodesMap = new Map<
        string,
        {
          identity: string | number;
          labels: string[];
          properties: Record<string, unknown>;
        }
      >();
      const relationships: Array<{
        identity: string | number;
        type: string;
        startNodeId: string | number;
        endNodeId: string | number;
        properties: Record<string, unknown>;
      }> = [];

      for (const record of result.records) {
        if (record.n) {
          nodesMap.set(String(record.n.identity), record.n);
        }
        if (record.m) {
          nodesMap.set(String(record.m.identity), record.m);
        }
        if (record.r) {
          relationships.push(record.r);
        }
      }

      return transformerRef.current!.transform(
        Array.from(nodesMap.values()),
        relationships
      );
    },
    []
  );

  // Fetch subgraph
  const fetchSubgraph = useCallback(
    async (
      nodeId: string | number,
      depth: number = 2
    ): Promise<TransformedGraph> => {
      if (!adapterRef.current?.connected) {
        throw new Error("Not connected to Neo4j");
      }

      const result = await adapterRef.current.fetchSubgraph(nodeId, depth);
      return transformerRef.current!.transform(
        result.nodes,
        result.relationships
      );
    },
    []
  );

  // Search nodes
  const searchNodes = useCallback(
    async (
      property: string,
      value: string,
      fuzzy: boolean = false
    ): Promise<GraphNode[]> => {
      if (!adapterRef.current?.connected) {
        throw new Error("Not connected to Neo4j");
      }

      const neo4jNodes = await adapterRef.current.searchNodes(
        property,
        value,
        fuzzy
      );
      const transformed = transformerRef.current!.transform(neo4jNodes, []);
      return transformed.nodes;
    },
    []
  );

  // Visualize query
  const visualizeQuery = useCallback(
    (query: CypherQuery): QueryVisualization => {
      return visualizerRef.current!.visualize(query);
    },
    []
  );

  const actions: Neo4jGraphActions = useMemo(
    () => ({
      connect,
      disconnect,
      refresh,
      runQuery,
      fetchSubgraph,
      searchNodes,
      visualizeQuery,
    }),
    [
      connect,
      disconnect,
      refresh,
      runQuery,
      fetchSubgraph,
      searchNodes,
      visualizeQuery,
    ]
  );

  return [state, actions];
}

// ============================================================================
// Component: Neo4jGraphRenderer
// ============================================================================

export interface Neo4jGraphRendererProps extends Neo4jRenderOptions {
  /** Render function for graph data */
  children: (
    state: Neo4jGraphState,
    actions: Neo4jGraphActions
  ) => React.ReactNode;
  /** Auto-connect on mount */
  autoConnect?: boolean;
}

/**
 * Render prop component for Neo4j graph visualization
 */
export const Neo4jGraphRenderer: React.FC<Neo4jGraphRendererProps> = ({
  children,
  autoConnect = false,
  ...options
}) => {
  const [state, actions] = useNeo4jGraph(options);

  useEffect(() => {
    if (autoConnect) {
      actions.connect();
    }

    return () => {
      if (state.connected) {
        actions.disconnect();
      }
    };
  }, [autoConnect]);

  return <>{children(state, actions)}</>;
};

// ============================================================================
// Component: Neo4jConnectionStatus
// ============================================================================

export interface Neo4jConnectionStatusProps {
  state: Neo4jGraphState;
  onConnect?: () => void;
  onDisconnect?: () => void;
  className?: string;
}

/**
 * Connection status indicator component
 */
export const Neo4jConnectionStatus: React.FC<Neo4jConnectionStatusProps> = ({
  state,
  onConnect,
  onDisconnect,
  className = "",
}) => {
  const statusColor = state.connected
    ? "#22c55e"
    : state.loading
      ? "#f59e0b"
      : state.error
        ? "#ef4444"
        : "#6b7280";

  const statusText = state.connected
    ? "Connected"
    : state.loading
      ? "Connecting..."
      : state.error
        ? "Error"
        : "Disconnected";

  return (
    <div
      className={`neo4j-connection-status ${className}`}
      style={{ display: "flex", alignItems: "center", gap: "8px" }}
    >
      <div
        style={{
          width: "10px",
          height: "10px",
          borderRadius: "50%",
          backgroundColor: statusColor,
          boxShadow: `0 0 4px ${statusColor}`,
        }}
      />
      <span style={{ fontSize: "12px", color: "#a0a0a0" }}>{statusText}</span>
      {!state.connected && !state.loading && onConnect && (
        <button
          onClick={onConnect}
          style={{
            padding: "4px 8px",
            fontSize: "11px",
            backgroundColor: "#3b82f6",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Connect
        </button>
      )}
      {state.connected && onDisconnect && (
        <button
          onClick={onDisconnect}
          style={{
            padding: "4px 8px",
            fontSize: "11px",
            backgroundColor: "#6b7280",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Disconnect
        </button>
      )}
      {state.lastUpdate && (
        <span style={{ fontSize: "10px", color: "#6b7280" }}>
          Last update: {new Date(state.lastUpdate).toLocaleTimeString()}
        </span>
      )}
    </div>
  );
};

// ============================================================================
// Component: Neo4jQueryInput
// ============================================================================

export interface Neo4jQueryInputProps {
  onExecute: (query: string) => void;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

/**
 * Cypher query input component
 */
export const Neo4jQueryInput: React.FC<Neo4jQueryInputProps> = ({
  onExecute,
  loading = false,
  disabled = false,
  placeholder = "Enter Cypher query...",
  className = "",
}) => {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !loading && !disabled) {
      onExecute(query.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleSubmit(e);
    }
  };

  return (
    <form
      className={`neo4j-query-input ${className}`}
      onSubmit={handleSubmit}
      style={{ display: "flex", flexDirection: "column", gap: "8px" }}
    >
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled || loading}
        style={{
          width: "100%",
          minHeight: "80px",
          padding: "12px",
          fontFamily: "monospace",
          fontSize: "13px",
          backgroundColor: "#1a1a1a",
          color: "#e0e0e0",
          border: "1px solid #333",
          borderRadius: "6px",
          resize: "vertical",
        }}
      />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span style={{ fontSize: "11px", color: "#6b7280" }}>
          Press Ctrl+Enter to execute
        </span>
        <button
          type="submit"
          disabled={disabled || loading || !query.trim()}
          style={{
            padding: "8px 16px",
            fontSize: "13px",
            backgroundColor: loading || disabled ? "#4b5563" : "#3b82f6",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: loading || disabled ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Executing..." : "Run Query"}
        </button>
      </div>
    </form>
  );
};

// ============================================================================
// Component: Neo4jGraphStats
// ============================================================================

export interface Neo4jGraphStatsProps {
  data: TransformedGraph | null;
  className?: string;
}

/**
 * Graph statistics display component
 */
export const Neo4jGraphStats: React.FC<Neo4jGraphStatsProps> = ({
  data,
  className = "",
}) => {
  if (!data) {
    return (
      <div
        className={`neo4j-graph-stats ${className}`}
        style={{ color: "#6b7280" }}
      >
        No data loaded
      </div>
    );
  }

  return (
    <div
      className={`neo4j-graph-stats ${className}`}
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
        gap: "12px",
        padding: "12px",
        backgroundColor: "#1a1a1a",
        borderRadius: "6px",
      }}
    >
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "24px", fontWeight: "bold", color: "#3b82f6" }}>
          {data.metadata.nodeCount}
        </div>
        <div style={{ fontSize: "11px", color: "#6b7280" }}>Nodes</div>
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "24px", fontWeight: "bold", color: "#22c55e" }}>
          {data.metadata.edgeCount}
        </div>
        <div style={{ fontSize: "11px", color: "#6b7280" }}>Edges</div>
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "24px", fontWeight: "bold", color: "#f59e0b" }}>
          {data.metadata.labels.length}
        </div>
        <div style={{ fontSize: "11px", color: "#6b7280" }}>Labels</div>
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "24px", fontWeight: "bold", color: "#a855f7" }}>
          {data.metadata.relationshipTypes.length}
        </div>
        <div style={{ fontSize: "11px", color: "#6b7280" }}>Rel Types</div>
      </div>
    </div>
  );
};
