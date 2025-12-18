/**
 * useAIChat Hook - React integration for Ryot LLM
 *
 * Provides stateful chat management with streaming support
 * for the AI Panel component.
 *
 * @module @neurectomy/hooks
 * @author @LINGUA @TENSOR
 */

import { useState, useCallback, useRef } from "react";
import {
  streamChatCompletion,
  isRyotAvailable,
  ChatMessage as ServiceChatMessage,
  StreamDelta,
} from "@/services/ryot-service";
import type { ChatMessage, MessageContext } from "@/components/shell/AIPanel";

// ============================================================================
// Types
// ============================================================================

export interface UseAIChatOptions {
  initialMessages?: ChatMessage[];
  systemPrompt?: string;
  model?: string;
  temperature?: number;
  onError?: (error: Error) => void;
}

export interface UseAIChatReturn {
  messages: ChatMessage[];
  isStreaming: boolean;
  isLoading: boolean;
  error: Error | null;
  ryotAvailable: boolean;
  sendMessage: (content: string, context?: MessageContext[]) => Promise<void>;
  cancelStream: () => void;
  clearChat: () => void;
  regenerateResponse: (messageId: string) => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useAIChat(options: UseAIChatOptions = {}): UseAIChatReturn {
  const {
    initialMessages = [],
    systemPrompt = "You are a helpful AI coding assistant integrated into the NEURECTOMY IDE. You help developers write better code, debug issues, and explain complex concepts. Be concise and provide code examples when helpful.",
    model,
    temperature = 0.7,
    onError,
  } = options;

  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [ryotAvailable, setRyotAvailable] = useState(false);

  const cancelFnRef = useRef<(() => void) | null>(null);
  const currentAssistantIdRef = useRef<string | null>(null);

  // Check Ryot availability on mount
  useState(() => {
    isRyotAvailable().then(setRyotAvailable);
  });

  // Generate unique message ID
  const generateId = () =>
    `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

  // Convert UI messages to service format
  const toServiceMessages = (
    msgs: ChatMessage[],
    systemMsg?: string
  ): ServiceChatMessage[] => {
    const result: ServiceChatMessage[] = [];

    // Add system prompt if provided
    if (systemMsg) {
      result.push({ role: "system", content: systemMsg });
    }

    // Convert messages
    for (const msg of msgs) {
      let content = msg.content;

      // Include context if present
      if (msg.context && msg.context.length > 0) {
        const contextStr = msg.context
          .map((ctx) => `[${ctx.type}: ${ctx.name}]\n${ctx.preview || ""}`)
          .join("\n\n");
        content = `${contextStr}\n\n${content}`;
      }

      result.push({
        role: msg.role,
        content,
      });
    }

    return result;
  };

  // Send message
  const sendMessage = useCallback(
    async (content: string, context?: MessageContext[]) => {
      if (!content.trim() || isStreaming) return;

      setError(null);

      // Create user message
      const userMessage: ChatMessage = {
        id: generateId(),
        role: "user",
        content: content.trim(),
        timestamp: new Date(),
        context,
      };

      // Create placeholder assistant message
      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
        streaming: true,
      };

      currentAssistantIdRef.current = assistantMessage.id;

      // Add messages to state
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);
      setIsLoading(true);

      try {
        // Get all messages including new user message
        const allMessages = [...messages, userMessage];
        const serviceMessages = toServiceMessages(allMessages, systemPrompt);

        // Start streaming
        cancelFnRef.current = await streamChatCompletion(
          {
            messages: serviceMessages,
            temperature,
            model,
            stream: true,
          },
          // On delta
          (delta: StreamDelta) => {
            setIsLoading(false);
            if (delta.content) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === currentAssistantIdRef.current
                    ? { ...msg, content: msg.content + delta.content }
                    : msg
                )
              );
            }
          },
          // On complete
          () => {
            setIsStreaming(false);
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === currentAssistantIdRef.current
                  ? { ...msg, streaming: false }
                  : msg
              )
            );
            cancelFnRef.current = null;
            currentAssistantIdRef.current = null;
          },
          // On error
          (err: Error) => {
            setIsStreaming(false);
            setIsLoading(false);
            setError(err);
            onError?.(err);

            // Update message with error
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === currentAssistantIdRef.current
                  ? {
                      ...msg,
                      content: `Error: ${err.message}. Please try again.`,
                      streaming: false,
                    }
                  : msg
              )
            );

            cancelFnRef.current = null;
            currentAssistantIdRef.current = null;
          }
        );
      } catch (err) {
        const error = err as Error;
        setIsStreaming(false);
        setIsLoading(false);
        setError(error);
        onError?.(error);
      }
    },
    [messages, isStreaming, systemPrompt, temperature, model, onError]
  );

  // Cancel streaming
  const cancelStream = useCallback(() => {
    if (cancelFnRef.current) {
      cancelFnRef.current();
      cancelFnRef.current = null;
    }

    setIsStreaming(false);
    setIsLoading(false);

    // Mark current message as not streaming
    if (currentAssistantIdRef.current) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === currentAssistantIdRef.current
            ? {
                ...msg,
                streaming: false,
                content: msg.content + " [Cancelled]",
              }
            : msg
        )
      );
      currentAssistantIdRef.current = null;
    }
  }, []);

  // Clear chat
  const clearChat = useCallback(() => {
    cancelStream();
    setMessages([]);
    setError(null);
  }, [cancelStream]);

  // Regenerate response
  const regenerateResponse = useCallback(
    (messageId: string) => {
      // Find the message and the user message before it
      const messageIndex = messages.findIndex((m) => m.id === messageId);
      if (messageIndex <= 0) return;

      // Find the previous user message
      let userMessageIndex = messageIndex - 1;
      while (
        userMessageIndex >= 0 &&
        messages[userMessageIndex].role !== "user"
      ) {
        userMessageIndex--;
      }

      if (userMessageIndex < 0) return;

      const userMessage = messages[userMessageIndex];

      // Remove messages from the regeneration point
      setMessages((prev) => prev.slice(0, userMessageIndex));

      // Resend the user message
      setTimeout(() => {
        sendMessage(userMessage.content, userMessage.context);
      }, 100);
    },
    [messages, sendMessage]
  );

  return {
    messages,
    isStreaming,
    isLoading,
    error,
    ryotAvailable,
    sendMessage,
    cancelStream,
    clearChat,
    regenerateResponse,
  };
}

export default useAIChat;
