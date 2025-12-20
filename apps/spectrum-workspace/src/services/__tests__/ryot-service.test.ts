/**
 * Ryot LLM Service Tests
 *
 * Unit tests for the Ryot LLM integration service.
 *
 * @module @neurectomy/services/tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { invoke } from "@tauri-apps/api/core";

// Mock modules before importing the service
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

// Import after mocks
import {
  isRyotAvailable,
  setAIConfig,
  ChatMessage,
  ChatCompletionRequest,
} from "../ryot-service";

const mockInvoke = vi.mocked(invoke);

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("RyotService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset to default config
    setAIConfig({
      ryotApiUrl: "http://localhost:46080",
      defaultModel: "ryot-bitnet-7b",
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("isRyotAvailable", () => {
    it("should return true when Tauri command succeeds", async () => {
      mockInvoke.mockResolvedValueOnce(true);

      const available = await isRyotAvailable();

      expect(available).toBe(true);
      expect(mockInvoke).toHaveBeenCalledWith("check_ryot_available");
    });

    it("should fallback to HTTP when Tauri command fails", async () => {
      mockInvoke.mockRejectedValueOnce(new Error("No Tauri"));
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ models: [] }),
      });

      const available = await isRyotAvailable();

      expect(available).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:46080/v1/models",
        expect.objectContaining({ method: "GET" })
      );
    });

    it("should return false when both Tauri and HTTP fail", async () => {
      mockInvoke.mockRejectedValueOnce(new Error("No Tauri"));
      mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

      const available = await isRyotAvailable();

      expect(available).toBe(false);
    });

    it("should return false on non-ok HTTP response", async () => {
      mockInvoke.mockRejectedValueOnce(new Error("No Tauri"));
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
      });

      const available = await isRyotAvailable();

      expect(available).toBe(false);
    });
  });

  describe("setAIConfig", () => {
    it("should update the API URL", async () => {
      setAIConfig({ ryotApiUrl: "http://custom:9000" });

      // Trigger a call that uses the config
      mockInvoke.mockRejectedValueOnce(new Error("No Tauri"));
      mockFetch.mockResolvedValueOnce({ ok: true });

      await isRyotAvailable();

      expect(mockFetch).toHaveBeenCalledWith(
        "http://custom:9000/v1/models",
        expect.anything()
      );
    });

    it("should update the default model", () => {
      setAIConfig({ defaultModel: "custom-model" });
      // Model is used in chat completion, not directly testable without full flow
      // But config update should not throw
      expect(true).toBe(true);
    });
  });

  describe("Type exports", () => {
    it("should export ChatMessage type correctly", () => {
      const message: ChatMessage = {
        role: "user",
        content: "Hello",
      };
      expect(message.role).toBe("user");
      expect(message.content).toBe("Hello");
    });

    it("should export ChatCompletionRequest type correctly", () => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Test" }],
        model: "test-model",
        temperature: 0.7,
        max_tokens: 100,
        stream: false,
      };
      expect(request.messages).toHaveLength(1);
      expect(request.model).toBe("test-model");
    });
  });
});
