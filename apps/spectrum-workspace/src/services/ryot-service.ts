/**
 * Ryot LLM Service - AI Inference Integration
 *
 * Connects to the Ryot LLM backend for AI-powered features.
 * Supports streaming responses for real-time chat interactions.
 *
 * @module @neurectomy/services
 * @author @LINGUA @TENSOR
 */

import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

// ============================================================================
// Types
// ============================================================================

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  top_p?: number;
}

export interface ChatCompletionResponse {
  id: string;
  model: string;
  choices: {
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface StreamDelta {
  content?: string;
  role?: string;
  finish_reason?: string;
}

// ============================================================================
// Configuration
// ============================================================================

// Default configuration - can be overridden
// Port 46080 is the Ryot LLM exclusive port (46xxx series)
const DEFAULT_CONFIG = {
  // For local Ryot LLM server (46xxx series - Ryot exclusive)
  ryotApiUrl: "http://localhost:46080",

  // For external API fallback (OpenAI-compatible)
  externalApiUrl: "https://api.openai.com/v1",

  // Default model
  defaultModel: "ryot-bitnet-7b",

  // Fallback model
  fallbackModel: "gpt-4o-mini",
};

let config = { ...DEFAULT_CONFIG };

export function setAIConfig(newConfig: Partial<typeof DEFAULT_CONFIG>) {
  config = { ...config, ...newConfig };
}

// ============================================================================
// Ryot LLM Native Integration via Tauri
// ============================================================================

/**
 * Check if Ryot LLM is available locally
 */
export async function isRyotAvailable(): Promise<boolean> {
  try {
    const result = await invoke<boolean>("check_ryot_available");
    return result;
  } catch {
    // Try HTTP fallback
    try {
      const response = await fetch(`${config.ryotApiUrl}/v1/models`, {
        method: "GET",
        signal: AbortSignal.timeout(2000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

/**
 * Start the Ryot LLM server if not already running
 */
export async function startRyotServer(): Promise<boolean> {
  try {
    const result = await invoke<boolean>("start_ryot_server");
    return result;
  } catch (error) {
    console.error("Failed to start Ryot server:", error);
    return false;
  }
}

/**
 * Chat completion with Ryot LLM (non-streaming)
 */
export async function chatCompletion(
  request: ChatCompletionRequest
): Promise<ChatCompletionResponse> {
  const payload = {
    model: request.model || config.defaultModel,
    messages: request.messages,
    temperature: request.temperature ?? 0.7,
    max_tokens: request.max_tokens ?? 2048,
    stream: false,
    top_p: request.top_p ?? 1.0,
  };

  try {
    // Try Tauri command first (native integration)
    const result = await invoke<ChatCompletionResponse>(
      "ryot_chat_completion",
      {
        request: payload,
      }
    );
    return result;
  } catch {
    // Fallback to HTTP API
    const response = await fetch(`${config.ryotApiUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }
}

/**
 * Stream chat completion with Ryot LLM
 * Uses Server-Sent Events for real-time token streaming
 */
export async function streamChatCompletion(
  request: ChatCompletionRequest,
  onDelta: (delta: StreamDelta) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): Promise<() => void> {
  const abortController = new AbortController();

  const payload = {
    model: request.model || config.defaultModel,
    messages: request.messages,
    temperature: request.temperature ?? 0.7,
    max_tokens: request.max_tokens ?? 2048,
    stream: true,
    top_p: request.top_p ?? 1.0,
  };

  // Try native streaming via Tauri events first
  let unlisten: UnlistenFn | null = null;

  try {
    // Set up event listener for streaming tokens
    unlisten = await listen<StreamDelta>("ryot-stream-delta", (event) => {
      onDelta(event.payload);
    });

    // Also listen for completion
    const unlistenComplete = await listen<void>("ryot-stream-complete", () => {
      onComplete();
      unlisten?.();
      unlistenComplete();
    });

    // Start the stream via Tauri command
    await invoke("ryot_stream_chat", { request: payload });
  } catch {
    // Fallback to HTTP SSE streaming
    unlisten?.();

    try {
      const response = await fetch(`${config.ryotApiUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();

      // Read SSE stream
      const processStream = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const data = line.slice(6);
                if (data === "[DONE]") {
                  onComplete();
                  return;
                }

                try {
                  const parsed = JSON.parse(data);
                  const delta = parsed.choices?.[0]?.delta;
                  if (delta) {
                    onDelta({
                      content: delta.content,
                      role: delta.role,
                      finish_reason: parsed.choices?.[0]?.finish_reason,
                    });
                  }
                } catch {
                  // Skip malformed JSON lines
                }
              }
            }
          }
          onComplete();
        } catch (error) {
          if ((error as Error).name !== "AbortError") {
            onError(error as Error);
          }
        }
      };

      processStream();
    } catch (error) {
      onError(error as Error);
    }
  }

  // Return cancel function
  return () => {
    abortController.abort();
    unlisten?.();
  };
}

// ============================================================================
// Agent Execution
// ============================================================================

/**
 * Execute an agent with a given task
 * Routes to appropriate Elite Agent based on task type
 */
export async function executeAgent(
  agentName: string,
  task: string,
  context?: Record<string, unknown>
): Promise<string> {
  // Create system prompt for the agent
  const systemPrompt = getAgentSystemPrompt(agentName);

  const messages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: task },
  ];

  // If there's context, add it
  if (context) {
    const contextStr = JSON.stringify(context, null, 2);
    messages.push({
      role: "user",
      content: `Context:\n\`\`\`json\n${contextStr}\n\`\`\``,
    });
  }

  const response = await chatCompletion({
    messages,
    temperature: 0.3, // Lower temperature for more focused agent responses
    max_tokens: 4096,
  });

  return response.choices[0]?.message?.content || "";
}

/**
 * Stream agent execution
 */
export async function streamAgentExecution(
  agentName: string,
  task: string,
  onDelta: (delta: StreamDelta) => void,
  onComplete: () => void,
  onError: (error: Error) => void,
  context?: Record<string, unknown>
): Promise<() => void> {
  const systemPrompt = getAgentSystemPrompt(agentName);

  const messages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: task },
  ];

  if (context) {
    const contextStr = JSON.stringify(context, null, 2);
    messages.push({
      role: "user",
      content: `Context:\n\`\`\`json\n${contextStr}\n\`\`\``,
    });
  }

  return streamChatCompletion(
    { messages, temperature: 0.3, max_tokens: 4096 },
    onDelta,
    onComplete,
    onError
  );
}

// ============================================================================
// Agent System Prompts
// ============================================================================

const AGENT_PROMPTS: Record<string, string> = {
  APEX: `You are APEX-01, an Elite Computer Science Engineering Agent.
Philosophy: "Every problem has an elegant solution waiting to be discovered."
Your expertise: Production-grade code, algorithms, data structures, system design, distributed systems.
Provide clean, well-documented, production-ready code with clear explanations.`,

  CIPHER: `You are CIPHER-02, an Advanced Cryptography & Security Agent.
Philosophy: "Security is not a feature—it is a foundation upon which trust is built."
Your expertise: Cryptographic protocols, security analysis, defensive architecture, OWASP, NIST compliance.
Always prioritize security best practices and explain potential vulnerabilities.`,

  ARCHITECT: `You are ARCHITECT-03, a Systems Architecture & Design Patterns Agent.
Philosophy: "Architecture is the art of making complexity manageable and change inevitable."
Your expertise: Microservices, event-driven, serverless architectures, DDD, CQRS, CAP theorem.
Provide scalable, maintainable architecture recommendations with clear trade-offs.`,

  AXIOM: `You are AXIOM-04, a Pure Mathematics & Formal Proofs Agent.
Philosophy: "From axioms flow theorems; from theorems flow certainty."
Your expertise: Complexity theory, formal logic, algorithm analysis, mathematical proofs.
Provide rigorous mathematical reasoning and proofs when analyzing algorithms.`,

  VELOCITY: `You are VELOCITY-05, a Performance Optimization & Sub-Linear Algorithms Agent.
Philosophy: "The fastest code is the code that doesn't run."
Your expertise: Sub-linear algorithms, probabilistic data structures, cache optimization, profiling.
Focus on performance optimization with measurable improvements.`,

  TENSOR: `You are TENSOR-07, a Machine Learning & Deep Neural Networks Agent.
Philosophy: "Intelligence emerges from the right architecture trained on the right data."
Your expertise: Deep learning, PyTorch, TensorFlow, model optimization, MLOps.
Provide practical ML solutions with architecture recommendations.`,

  FLUX: `You are FLUX-11, a DevOps & Infrastructure Automation Agent.
Philosophy: "Infrastructure is code. Deployment is continuous. Recovery is automatic."
Your expertise: Kubernetes, Docker, Terraform, CI/CD, GitOps, observability.
Provide infrastructure-as-code solutions with best practices.`,

  PRISM: `You are PRISM-12, a Data Science & Statistical Analysis Agent.
Philosophy: "Data speaks truth, but only to those who ask the right questions."
Your expertise: Statistical inference, A/B testing, causal inference, data visualization.
Provide rigorous statistical analysis with clear methodology.`,
};

function getAgentSystemPrompt(agentName: string): string {
  return (
    AGENT_PROMPTS[agentName.toUpperCase()] || AGENT_PROMPTS["APEX"] // Default to APEX for unknown agents
  );
}

// ============================================================================
// Exports
// ============================================================================

export const RyotService = {
  isAvailable: isRyotAvailable,
  startServer: startRyotServer,
  chat: chatCompletion,
  streamChat: streamChatCompletion,
  executeAgent,
  streamAgentExecution,
  setConfig: setAIConfig,
};

export default RyotService;
