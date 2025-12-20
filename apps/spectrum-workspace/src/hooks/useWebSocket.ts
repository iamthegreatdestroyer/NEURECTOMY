/**
 * WebSocket Hook
 * Real-time connection to Rust Core backend
 */

import { useEffect, useRef, useCallback, useState } from "react";
import { useAppStore } from "../stores/app.store";

// WebSocket message types
interface WSMessage {
  type: string;
  payload: unknown;
  timestamp: string;
  requestId?: string;
}

// Connection states
type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

// Hook options
interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onMessage?: (message: WSMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

// Default WebSocket URL
const DEFAULT_WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:16083/ws";

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = DEFAULT_WS_URL,
    autoConnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );

  const [connectionState, setConnectionState] =
    useState<ConnectionState>("disconnected");
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);

  const setConnectionStatus = useAppStore((state) => state.setConnectionStatus);
  const addNotification = useAppStore((state) => state.addNotification);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState("connecting");

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setConnectionState("connected");
        setConnectionStatus(true);
        reconnectAttemptsRef.current = 0;

        addNotification({
          type: "success",
          title: "Connected",
          message: "Real-time connection established",
        });

        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onclose = () => {
        setConnectionState("disconnected");
        setConnectionStatus(false);
        wsRef.current = null;

        onDisconnect?.();

        // Attempt reconnection
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else {
          addNotification({
            type: "error",
            title: "Connection Lost",
            message: "Unable to reconnect to server",
          });
        }
      };

      ws.onerror = (error) => {
        setConnectionState("error");
        setConnectionStatus(false);
        onError?.(error);
      };

      wsRef.current = ws;
    } catch (error) {
      setConnectionState("error");
      setConnectionStatus(false);
      console.error("WebSocket connection error:", error);
    }
  }, [
    url,
    maxReconnectAttempts,
    reconnectInterval,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
    setConnectionStatus,
    addNotification,
  ]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState("disconnected");
    setConnectionStatus(false);
  }, [setConnectionStatus]);

  // Send message
  const sendMessage = useCallback(
    (type: string, payload: unknown, requestId?: string) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        console.error("WebSocket is not connected");
        return false;
      }

      const message: WSMessage = {
        type,
        payload,
        timestamp: new Date().toISOString(),
        requestId: requestId || crypto.randomUUID(),
      };

      try {
        wsRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error("Failed to send WebSocket message:", error);
        return false;
      }
    },
    []
  );

  // Subscribe to specific message types
  const subscribe = useCallback(
    (type: string, handler: (payload: unknown) => void) => {
      const messageHandler = (message: WSMessage) => {
        if (message.type === type) {
          handler(message.payload);
        }
      };

      // This is a simplified implementation
      // In production, you'd want a proper pub/sub system
      return () => {
        // Unsubscribe logic
      };
    },
    []
  );

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    connectionState,
    isConnected: connectionState === "connected",
    lastMessage,
    connect,
    disconnect,
    sendMessage,
    subscribe,
  };
}

// Typed message hooks for specific message types
export function useAgentStatusUpdates(onUpdate: (status: unknown) => void) {
  const { subscribe, isConnected } = useWebSocket({
    onMessage: (message) => {
      if (message.type === "agent_status") {
        onUpdate(message.payload);
      }
    },
  });

  return { isConnected };
}

export function useContainerEvents(onEvent: (event: unknown) => void) {
  const { subscribe, isConnected } = useWebSocket({
    onMessage: (message) => {
      if (message.type === "container_event") {
        onEvent(message.payload);
      }
    },
  });

  return { isConnected };
}

export function useTrainingProgress(
  jobId: string,
  onProgress: (progress: unknown) => void
) {
  const { sendMessage, isConnected } = useWebSocket({
    onMessage: (message) => {
      if (
        message.type === "training_progress" &&
        (message.payload as any)?.jobId === jobId
      ) {
        onProgress(message.payload);
      }
    },
    onConnect: () => {
      // Subscribe to job updates when connected
      sendMessage("subscribe_training", { jobId });
    },
  });

  return { isConnected };
}
