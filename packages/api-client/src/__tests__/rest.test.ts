import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { RestClient, RestError, createRestClient } from "../rest";

describe("RestClient", () => {
  let client: RestClient;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    client = new RestClient({ baseUrl: "https://api.test.com" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("constructor", () => {
    it("should strip trailing slash from baseUrl", () => {
      const c = new RestClient({ baseUrl: "https://api.test.com/" });
      fetchMock.mockResolvedValue({
        ok: true,
        headers: new Map([["content-type", "application/json"]]),
        json: () => Promise.resolve({}),
      });

      c.get("/test");
      expect(fetchMock).toHaveBeenCalledWith(
        "https://api.test.com/test",
        expect.any(Object)
      );
    });

    it("should set default Content-Type header", () => {
      fetchMock.mockResolvedValue({
        ok: true,
        headers: new Map([["content-type", "application/json"]]),
        json: () => Promise.resolve({}),
      });

      client.get("/test");
      expect(fetchMock).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            "Content-Type": "application/json",
          }),
        })
      );
    });
  });

  describe("get", () => {
    it("should make GET request", async () => {
      const mockResponse = { id: 1, name: "Test" };
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.get("/users/1");

      expect(fetchMock).toHaveBeenCalledWith(
        "https://api.test.com/users/1",
        expect.objectContaining({ method: "GET" })
      );
      expect(result).toEqual(mockResponse);
    });

    it("should append query params", async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve([]),
      });

      await client.get("/users", { page: 1, limit: 10 });

      expect(fetchMock).toHaveBeenCalledWith(
        "https://api.test.com/users?page=1&limit=10",
        expect.any(Object)
      );
    });
  });

  describe("post", () => {
    it("should make POST request with body", async () => {
      const body = { name: "New User" };
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve({ id: 1, ...body }),
      });

      await client.post("/users", body);

      expect(fetchMock).toHaveBeenCalledWith(
        "https://api.test.com/users",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify(body),
        })
      );
    });
  });

  describe("put", () => {
    it("should make PUT request", async () => {
      const body = { name: "Updated" };
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve(body),
      });

      await client.put("/users/1", body);

      expect(fetchMock).toHaveBeenCalledWith(
        "https://api.test.com/users/1",
        expect.objectContaining({ method: "PUT" })
      );
    });
  });

  describe("patch", () => {
    it("should make PATCH request", async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve({}),
      });

      await client.patch("/users/1", { name: "Patched" });

      expect(fetchMock).toHaveBeenCalledWith(
        "https://api.test.com/users/1",
        expect.objectContaining({ method: "PATCH" })
      );
    });
  });

  describe("delete", () => {
    it("should make DELETE request", async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve({}),
      });

      await client.delete("/users/1");

      expect(fetchMock).toHaveBeenCalledWith(
        "https://api.test.com/users/1",
        expect.objectContaining({ method: "DELETE" })
      );
    });
  });

  describe("error handling", () => {
    it("should throw RestError on non-ok response", async () => {
      fetchMock.mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
        text: () => Promise.resolve('{"error":"User not found"}'),
      });

      await expect(client.get("/users/999")).rejects.toThrow(RestError);
    });

    it("should include status code in RestError", async () => {
      fetchMock.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        text: () => Promise.resolve("Server error"),
      });

      try {
        await client.get("/error");
      } catch (error) {
        expect(error).toBeInstanceOf(RestError);
        expect((error as RestError).statusCode).toBe(500);
      }
    });
  });

  describe("auth token", () => {
    it("should set authorization header", async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve({}),
      });

      client.setAuthToken("test-token");
      await client.get("/protected");

      expect(fetchMock).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: "Bearer test-token",
          }),
        })
      );
    });

    it("should clear authorization header", async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        headers: { get: () => "application/json" },
        json: () => Promise.resolve({}),
      });

      client.setAuthToken("test-token");
      client.clearAuthToken();
      await client.get("/public");

      const callHeaders = fetchMock.mock.calls[0][1].headers;
      expect(callHeaders.Authorization).toBeUndefined();
    });
  });
});

describe("RestError", () => {
  it("should store status code and response body", () => {
    const error = new RestError(
      "Not Found",
      404,
      '{"error":"Resource not found"}'
    );

    expect(error.message).toBe("Not Found");
    expect(error.statusCode).toBe(404);
    expect(error.responseBody).toBe('{"error":"Resource not found"}');
    expect(error.name).toBe("RestError");
  });

  it("should be instanceof Error", () => {
    const error = new RestError("Error", 500, "");
    expect(error).toBeInstanceOf(Error);
  });
});

describe("createRestClient", () => {
  it("should create RestClient instance", () => {
    const client = createRestClient({ baseUrl: "https://api.test.com" });
    expect(client).toBeInstanceOf(RestClient);
  });
});
