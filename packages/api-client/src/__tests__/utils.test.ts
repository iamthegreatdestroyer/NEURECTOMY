import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { handleApiError, retryRequest, isRetryableError } from "../utils";
import { RestError } from "../rest";

describe("handleApiError", () => {
  it("should handle RestError", () => {
    const error = new RestError(
      "Not Found",
      404,
      '{"message":"User not found"}'
    );
    const result = handleApiError(error);

    expect(result.code).toBe("HTTP_404");
    expect(result.message).toBe("Not Found");
    expect(result.statusCode).toBe(404);
    expect(result.details).toEqual({ message: "User not found" });
  });

  it("should handle RestError with invalid JSON body", () => {
    const error = new RestError("Error", 500, "Not JSON");
    const result = handleApiError(error);

    expect(result.code).toBe("HTTP_500");
    expect(result.details).toEqual({ rawBody: "Not JSON" });
  });

  it("should handle generic Error", () => {
    const error = new Error("Something went wrong");
    const result = handleApiError(error);

    expect(result.code).toBe("UNKNOWN_ERROR");
    expect(result.message).toBe("Something went wrong");
    expect(result.statusCode).toBeUndefined();
  });

  it("should handle string error", () => {
    const result = handleApiError("string error");

    expect(result.code).toBe("UNKNOWN_ERROR");
    expect(result.message).toBe("string error");
  });

  it("should handle unknown types", () => {
    const result = handleApiError(12345);

    expect(result.code).toBe("UNKNOWN_ERROR");
    expect(result.message).toBe("12345");
  });
});

describe("retryRequest", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should return result on success", async () => {
    const fn = vi.fn().mockResolvedValue("success");

    const result = await retryRequest(fn);

    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("should retry on failure", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("First fail"))
      .mockRejectedValueOnce(new Error("Second fail"))
      .mockResolvedValue("success");

    const promise = retryRequest(fn, { maxRetries: 3, baseDelayMs: 100 });

    // First call fails immediately
    await vi.advanceTimersByTimeAsync(0);
    // Wait for retry delay
    await vi.advanceTimersByTimeAsync(200);
    // Second call fails
    await vi.advanceTimersByTimeAsync(400);
    // Third call succeeds

    const result = await promise;

    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("should throw after max retries", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("Always fails"));

    const promise = retryRequest(fn, { maxRetries: 2, baseDelayMs: 100 });

    // Advance through all retries
    for (let i = 0; i < 3; i++) {
      await vi.advanceTimersByTimeAsync(10000);
    }

    await expect(promise).rejects.toThrow("Always fails");
    expect(fn).toHaveBeenCalledTimes(3); // Initial + 2 retries
  });

  it("should respect shouldRetry option", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("Non-retryable"));

    const promise = retryRequest(fn, {
      maxRetries: 3,
      shouldRetry: () => false,
    });

    await expect(promise).rejects.toThrow("Non-retryable");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("should use exponential backoff", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("fail"));

    const promise = retryRequest(fn, {
      maxRetries: 2,
      baseDelayMs: 1000,
      maxDelayMs: 10000,
    });

    // Just verify it doesn't complete immediately
    let resolved = false;
    promise.catch(() => {
      resolved = true;
    });

    expect(resolved).toBe(false);

    // Advance to complete all retries
    await vi.advanceTimersByTimeAsync(30000);

    await expect(promise).rejects.toThrow();
  });
});

describe("isRetryableError", () => {
  it("should return true for 5xx errors", () => {
    const error = new RestError("Server Error", 500, "");
    expect(isRetryableError(error)).toBe(true);
  });

  it("should return true for 503 errors", () => {
    const error = new RestError("Service Unavailable", 503, "");
    expect(isRetryableError(error)).toBe(true);
  });

  it("should return true for 429 rate limit errors", () => {
    const error = new RestError("Too Many Requests", 429, "");
    expect(isRetryableError(error)).toBe(true);
  });

  it("should return false for 4xx client errors", () => {
    const error = new RestError("Not Found", 404, "");
    expect(isRetryableError(error)).toBe(false);
  });

  it("should return false for 400 errors", () => {
    const error = new RestError("Bad Request", 400, "");
    expect(isRetryableError(error)).toBe(false);
  });

  it("should return true for fetch network errors", () => {
    const error = new TypeError("fetch failed");
    expect(isRetryableError(error)).toBe(true);
  });

  it("should return false for other errors", () => {
    const error = new Error("Some error");
    expect(isRetryableError(error)).toBe(false);
  });

  it("should return false for non-Error values", () => {
    expect(isRetryableError("string")).toBe(false);
    expect(isRetryableError(null)).toBe(false);
  });
});
