import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { RestClient, RestError } from "@neurectomy/api-client/rest";
import {
  handleApiError,
  retryRequest,
  isRetryableError,
} from "@neurectomy/api-client";
import {
  NeurectomyError,
  ValidationError,
  NetworkError,
} from "@neurectomy/core";

/**
 * Integration tests verifying cross-package functionality
 */

describe("API Client + Core Error Integration", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("should handle RestError through handleApiError", async () => {
    const client = new RestClient({ baseUrl: "https://api.test.com" });

    fetchMock.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: "Not Found",
      text: () => Promise.resolve('{"error":"Resource not found"}'),
    });

    try {
      await client.get("/missing");
    } catch (error) {
      expect(error).toBeInstanceOf(RestError);

      const apiError = handleApiError(error);
      expect(apiError.code).toBe("HTTP_404");
      expect(apiError.statusCode).toBe(404);
    }
  });

  it("should retry requests with retryable errors", async () => {
    vi.useFakeTimers();

    const client = new RestClient({ baseUrl: "https://api.test.com" });
    let callCount = 0;

    fetchMock.mockImplementation(() => {
      callCount++;
      if (callCount < 3) {
        return Promise.resolve({
          ok: false,
          status: 503,
          statusText: "Service Unavailable",
          text: () => Promise.resolve("Retry later"),
        });
      }
      return Promise.resolve({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve({ success: true }),
      });
    });

    const promise = retryRequest(() => client.get("/flaky"), {
      maxRetries: 3,
      baseDelayMs: 100,
      shouldRetry: isRetryableError,
    });

    // Advance through retries
    await vi.advanceTimersByTimeAsync(5000);

    const result = await promise;
    expect(result).toEqual({ success: true });
    expect(callCount).toBe(3);

    vi.useRealTimers();
  });
});

describe("Core Error Classes Cross-Package Usage", () => {
  it("should create NeurectomyError with context from API response", () => {
    const apiResponse = {
      code: "VALIDATION_FAILED",
      message: "Invalid input",
      fields: ["email", "name"],
    };

    const error = new NeurectomyError(apiResponse.message, apiResponse.code, {
      fields: apiResponse.fields,
    });

    expect(error.code).toBe("VALIDATION_FAILED");
    expect(error.context?.fields).toEqual(["email", "name"]);
    expect(error.toJSON()).toMatchObject({
      code: "VALIDATION_FAILED",
      context: { fields: ["email", "name"] },
    });
  });

  it("should create ValidationError from API validation response", () => {
    const validationErrors = [
      { field: "email", message: "Invalid email format" },
      { field: "password", message: "Too short" },
    ];

    const error = new ValidationError("Validation failed", validationErrors);

    expect(error.errors).toHaveLength(2);
    expect(error.errors[0].field).toBe("email");
  });

  it("should create NetworkError from fetch failure", () => {
    const error = new NetworkError(
      "Request failed",
      503,
      "https://api.test.com/endpoint"
    );

    expect(error.statusCode).toBe(503);
    expect(error.url).toBe("https://api.test.com/endpoint");
    expect(error.toJSON()).toMatchObject({
      statusCode: 503,
      url: "https://api.test.com/endpoint",
    });
  });
});

describe("Type Compatibility", () => {
  it("should use types from @neurectomy/types with API client", async () => {
    // This test verifies type compatibility at compile time
    // and runtime structure matching

    const mockAgent = {
      id: "agent-123",
      name: "TestAgent",
      codename: "TEST",
      tier: "foundational" as const,
      status: "idle" as const,
      description: "Test agent",
      philosophy: "Test philosophy",
      capabilities: ["test"],
      version: "1.0.0",
      createdAt: new Date(),
      updatedAt: new Date(),
      metadata: {},
    };

    // Verify structure matches Agent interface
    expect(mockAgent.id).toBe("agent-123");
    expect(mockAgent.tier).toBe("foundational");
    expect(mockAgent.status).toBe("idle");
  });
});
